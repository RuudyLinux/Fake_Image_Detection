# Fake Image Detection System — Project Specification

Status: v1 built and shipped. This doc reflects what's actually running,
not the original plan — see [Honest Status](#honest-status--known-limitations)
for what's fixed, what's traded off, and what's still open.

## Goal

Dual-approach, multi-modal system to detect fake images with explainable,
calibrated, honestly-uncertain output.

## Detection Modes

- **Blind detection**: single image, no original reference. CNN ensemble +
  FFT/ELA heuristics vote a 3-class verdict.
- **Comparative detection**: original + suspected image. pHash/dHash distance
  + ELA diff localizes manipulation between the pair.

## Architecture (as built)

### Models — 3, ensembled

| Model | Weight | Input | Training | Role |
| --- | --- | --- | --- | --- |
| EfficientNet-B4 | 0.5 | 380px | fine-tuned | primary |
| ConvNeXt-Tiny | 0.3 | 224px | fine-tuned | secondary — deliberately kept architecturally distinct so the pair disagrees on genuinely ambiguous images instead of sharing blind spots |
| CLIP ViT-L/14 + linear probe | 0.2 | 224px | **backbone frozen**, only the head trains | generalization to generators outside this project's training data — the two fine-tuned CNNs learn *this project's* generators' artifacts, not a general synthetic-vs-real concept (measured directly: both shared an identical blind spot on Stable Diffusion before it was added to training — see [generate_diffusion_data.py](backend/training/generate_diffusion_data.py)) |

Ensemble ([backend/models/ensemble.py](backend/models/ensemble.py)) combines
weighted logits, applies fitted temperature scaling
([backend/checkpoints/calibration.json](backend/checkpoints/calibration.json)),
and abstains ("uncertain") when confidence is low or members disagree —
honest non-answer beats a confident wrong one. Calibration currently only
covers the two CNN members (`fit_calibration.py`'s loading loop is CNN-only);
the CLIP member runs under that same shared temperature as an approximation —
known gap, flagged in code.

### Supporting detection tools — 4, non-trained

- **blind_detector.py** — FFT spectral analysis (GAN periodicity), ELA,
  noise/texture heuristics; fallback 3-class signal.
- **comparative_detector.py** — pHash + dHash distance, plus a
  **localization ratio** (mean diff inside flagged regions vs. outside) that
  separates genuine edits from harmless re-compression — SSIM magnitude alone
  can't (see Honest Status).
- **fusion.py** — blends CNN ensemble probabilities with heuristic scores
  into the final verdict.
- **gradcam.py** — Grad-CAM heatmap for explainability (doesn't affect the verdict).

`tta.py` — test-time augmentation feeding the ensemble; views must be
in-distribution for the classifier (plain resample-free views aren't enough —
see git history, fix 5da4bae).

## Outputs

- Prediction: REAL, AI_GENERATED, or EDITED — or **abstain (uncertain)**.
- Calibrated confidence score (temperature-scaled, not raw softmax).
- Grad-CAM heatmap.
- Comparative mode: pHash/dHash distance + ELA diff map.
- PDF report, history dashboard.

## Data and Training

### Current dataset composition

```
data/
  train/  real=13,597  ai_generated=8,800  edited=4,835   total=27,232
  val/    real=3,401   ai_generated=2,200  edited=1,209   total=6,810
```

`real` and `ai_generated` each mix a face-photo majority with an injected
non-face sub-domain (`casia_au_*`, `sd14_*`, plus `indoor67_*`/`textocr_*`
merged into the `casia_au_ind`/`casia_au_txt` buckets). `edited` (CASIA v2 Tp)
has its own internal tamper-type split (`Tp_S`/`Tp_D`/`Sp_S`/`Sp_D`). All of
these are balanced by [training/dataset.py](backend/training/dataset.py)'s
`make_sampler()` — see [Honest Status](#honest-status--known-limitations) for
why that had to be fixed three times, at increasing granularity each time.

### Training setup

- Platform: either Google Colab (T4) via
  [colab_fake_image_detection.ipynb](colab_fake_image_detection.ipynb), or
  local GPU via [backend/training/train.py](backend/training/train.py)
  directly (verified working on a 4.3GB laptop GPU with
  `--batch-size 8 --grad-accum-steps 4`).
- Framework: PyTorch, AMP (`autocast`/`GradScaler`).
- Two-phase: freeze backbone (head-only) → unfreeze, fine-tune at lower LR
  with cosine annealing.
- Loss: CrossEntropyLoss, label smoothing 0.1.
- Optimizer: AdamW, weight decay 1e-4.
- Augmentation: flip, rotation, color jitter, grayscale, **random downscale**
  and **Gaussian blur** (added after robustness testing showed both were the
  worst-surviving perturbations — accuracy dropped ~15-20pts without them in
  training), JPEG re-compression, Gaussian noise.
- `--resume` re-measures baseline on the **current** val set before deciding
  whether a new epoch is actually better — comparing against a checkpoint's
  stored metric from a stale val set once already discarded a genuinely
  better checkpoint (see train.py comments).
- `--save-all-dir` (added this session): saves every epoch's checkpoint, not
  just ones beating the aggregate val macro_f1 gate. Needed when the gate is
  blind to the minority sub-population a fix targets — see Honest Status.

## Evaluation

- [training/evaluate.py](backend/training/evaluate.py) — single-checkpoint
  3-class accuracy/F1/AUC + confusion matrix + inference latency.
- [training/evaluate_ensemble.py](backend/training/evaluate_ensemble.py) —
  fused-ensemble accuracy with/without TTA, abstention coverage.
- [training/evaluate_by_category.py](backend/training/evaluate_by_category.py)
  (added this session) — breaks val accuracy down by the sampler's own
  sub-domain groups (`real/txt`, `real/ind`, `edited/Tp_S`, ...). Exists
  because the aggregate metrics above can't see a minority category moving —
  see Honest Status for why that mattered in practice.
- [training/robustness_eval.py](backend/training/robustness_eval.py) —
  accuracy under compression/blur/resize/noise perturbation.

Metric JSONs are tracked in git despite `backend/checkpoints/` being
gitignored (`.gitignore` carves out `*_by_category.json`, `*.metrics.json`,
`calibration.json`) — otherwise "we improved X→Y" claims aren't diffable
against a prior commit.

## Website

### Frontend

React + Vite + TypeScript + Tailwind, dark theme. Drag-and-drop upload, two
modes (blind / comparative), result dashboard with Grad-CAM heatmap,
history view, PDF export.

### Backend

FastAPI. Routes in [backend/routers/detect.py](backend/routers/detect.py).
SQLite history ([backend/history.db](backend/history.db)).

## Honest Status — known limitations

This section exists because "it works" claims in this project have
repeatedly turned out to be measurement gaps, not fixed bugs — recorded
here so the next person (or session) doesn't have to rediscover them.

1. **Non-face REAL photos were misclassified EDITED at ~88% confidence,
   every time.** Root cause: REAL and AI_GENERATED were 100% face closeups;
   EDITED (CASIA v2) spans general photography. Model learned "not a face →
   EDITED" as a shortcut. Fixed by injecting CASIA's own `Au` (authentic,
   non-face) split into REAL, with sampler rebalancing so the minority
   sub-domain isn't drowned out by faces 2:1.

2. **That fix was itself class-only, not category-only** — `casia_au_` was
   balanced as one group vs. faces, but the object categories inside it
   (`ani/arc/art/cha/ind/nat/pla/sec/txt`) are wildly uneven in CASIA's own
   data (txt=79, ind=186 vs. 491-588 for the rest). Symptom: scene/text/indoor
   false-EDITED rate stayed elevated after fix #1. Fixed by extending the
   sampler to balance per-category, and separately per-tamper-type inside
   EDITED (`Tp_S`/`Tp_D`/`Sp_S`/`Sp_D`, previously never balanced at all).

3. **Fix #2 is a real trade, not a free win, and it doesn't land the same
   on both ensemble members:**

   | | EfficientNet-B4 (promoted) | ConvNeXt-Tiny (not promoted) |
   | --- | --- | --- |
   | real/txt | 0.150 → 0.350 | 0.150 → best 0.250 |
   | real/ind | 0.255 → 0.787 | 0.277 → best 0.553 |
   | real/arc | 0.455 → 0.713 | 0.706 → best 0.748 |
   | edited/Tp_S | 0.866 → 0.665 | 0.873 → 0.790 (worse in later epochs) |
   | edited/Sp_D | 0.914 → 0.901 | 0.988 → 0.951 |
   | **verdict** | **net win, promoted live** | **net loss on every epoch tried, kept baseline** |

   B4's baseline had more shortcut to remove and less accurate EDITED
   signal to lose; ConvNeXt's baseline was already strong on EDITED
   (Sp_D 0.988) with less shortcut reliance, so the identical fix cost more
   than it bought there. Ensemble now deliberately runs one updated member +
   one unchanged member — forcing parity would reduce the disagreement signal
   abstention depends on, for no accuracy gain.

   (B4's own numbers above are from that point in the session, not current —
   see item 5 for where they ended up after the next retrain.)

4. **txt (79) and ind (186) were CASIA's entire authentic pool for those
   categories — not a subsample.** Fixed by growing the source data itself:
   MIT Indoor67 (+798) and TextOCR (+800) injected as `indoor67_`/`textocr_`,
   merged into the same sampler bucket as `casia_au_ind`/`casia_au_txt`
   (not a new parallel group — see `dataset.py`'s `_CATEGORY_ALIASES`).
   Retrained EfficientNet-B4 on the enlarged pool (epoch9 promoted): real/txt
   0.623→0.886, real/ind 0.595→0.915 (measured against the old model
   zero-shot on the new data as baseline, since the val set grew too).
   Unlike the category-balancing retrain, every one of the 10 saved epochs
   beat baseline this round, and the aggregate val macro_f1 gate agreed with
   the per-category verdict for once.

5. **EDITED's majority tamper type keeps eroding, across both rounds of the
   non-face fix, and it hasn't leveled off yet:**
   edited/Tp_S (647 val images, the largest EDITED group) — 0.866 (original,
   shortcut-inflated) → 0.665 (after category-balancing) → **0.573** (after
   growing txt/ind). The model's own CNN signal now misses ~43% of the most
   common splice type. `fusion.py`'s ELA/pHash heuristics backstop EDITED
   detection independent of the CNN, so this isn't the whole system's EDITED
   recall — but the CNN member's contribution has roughly halved across two
   promotions moving in the same direction, and nothing in this arc has
   addressed the actual cause (EDITED's own training diversity, all from
   CASIA's synthetic splice ops). This is now the top open risk, not a
   footnote — see Immediate Next Steps #1.

6. Live model checkpoints (`backend/checkpoints/*.pth`) are **not** tracked
   in git (too large). The only rollback path after a promotion is whatever
   local backup file was kept at promotion time
   (e.g. `efficientnet_b4_fake_pre_v3_backup.pth`) — these backups don't
   survive a fresh clone.

7. **Comparative mode called 4 of 5 unedited images "manipulated."**
   `is_manipulated` was an OR of three loose thresholds
   (`score < 0.95 OR any diff region OR phash > 10`), each individually
   tripped by a plain JPEG re-save. Measured on 250 authentic CASIA images vs.
   their own re-compressed copies: **81.6% false-alarm rate**, 54.7%
   precision. Root cause: SSIM *magnitude* does not separate re-compression
   from editing at all — a genuine copy-move splice measured SSIM 0.955 while
   a clean q=30 re-save of the *same* unedited image measured 0.933, worse.
   Fixed by measuring the *spatial structure* of the difference instead — a
   localization ratio (mean diff inside flagged regions vs. outside).
   Re-compression spreads error diffusely; editing concentrates it. Validated
   on 3396 real ground-truth pairs recovered from CASIA's own filenames:
   recall 98.4%→95.6%, false-alarm 81.6%→**0.0%**, F1 0.703→**0.978**.

8. **CLIP ensemble member's generalization claim is validated same-family
   only.** The controlled A/B test (`clip_member_ab_test.metrics.json`) uses
   a held-out Stable Diffusion set — SD1.4 is already in the CNN members'
   training distribution after fix #4/#1's diffusion work, so this measures a
   regression check (no harm from adding CLIP, small real gain: 29/30+1
   abstain → 30/30) more than it measures the actual claim CLIP was added
   for. Generalization to generator families absent from training entirely
   (Flux, Midjourney v7, SDXL, Firefly) remains **unverified** — no local
   access to sample images from those generators. Don't cite the SD1.4 result
   as evidence of that broader claim; it isn't.

9. **Unseen-generator accuracy is at or below chance — now measured twice,
   scripted, not anecdotal.** The 0/30 figure in
   `generate_diffusion_data.py`'s docstring was real but is no longer
   reproducible: no checkpoint on disk predates SD1.4 entering training
   (the `*_pre_v*_backup.pth` files all score 97%+ on `sd14`, so they
   post-date it). Two fresh leave-one-generator-out runs replaced it:

   | Model | Held-out generator | Zero-shot accuracy |
   | --- | --- | --- |
   | EfficientNet-B4 (`--exclude-prefix sd14_`) | SD 1.4 | **42.5%** (85/200; 109/200 confidently REAL) |
   | Generation Fingerprint v1 (`--exclude-domain defactify_dalle3`) | DALL·E 3 | **40.0%** (150/375) |

   Both skew toward the dangerous direction (false REAL). The B4 run is a
   *diagnostic artifact, not a promotable checkpoint* — see item 10.

10. **A partial research-data download contaminated a baseline run, and the
    sampler was blind to it.** ~3,900 `defactify_*` images left on disk by a
    crashed download were picked up by the next training run: `--exclude-prefix`
    only knew about `sd14_`, and `make_sampler`'s `group_key` had no case for
    `defactify_`/`openfake_`, so ~7,500 new AI_GENERATED images landed against
    1,500 new REAL with no group balancing. Measured damage: **real/face
    accuracy 25.3%** (1,492/2,000 authentic face photos called AI_GENERATED),
    real/defactify_coco 32.5%. Fixed in `dataset.py::group_key` and
    `evaluate_by_category.py::category_of` (both now bucket per generator).
    The affected checkpoint (`efficientnet_b4_logo_nosd14.pth`) is kept for
    its LOGO number only and must not be promoted.

11. **The download script itself caused an OOM-class crash.** The first
    version of `download_research_data.py` buffered every accepted image as
    a decoded PIL object until the scan finished — 7.8GB RSS, system free RAM
    to 0.4GB, which killed a concurrent training run mid-epoch (torch shared
    memory, Windows error 1455 "paging file is too small"). Rewritten around
    `StreamingBucket`, which writes each image on acceptance: peak memory is
    now O(1) per bucket instead of O(target).

12. **A reverse id→name lookup fabricated a research result.** Several domain
    keys deliberately share one integer id (cross-dataset generator merges).
    `evaluate_fingerprint_logo.py` reversed that map to label its output,
    silently renaming the held-out `defactify_dalle3` to `openfake_dalle3` —
    a generator that was never downloaded — so the LOGO filter matched nothing
    and reported **`logo_accuracy: 0.0`**. Not a real result; the true figure
    was 40.0%. Fixed at the source: `domain_key` is now a first-class field
    carried through dataset → collate → eval, never reconstructed.

13. **Provenance layer added as Tier A** (`models/provenance.py`, wired into
    `blind_detector.analyze`). Two invariants worth restating because
    violating either is a real vulnerability: absence of provenance never
    raises confidence (else stripping a manifest *improves* an attacker's
    score), and a detected manifest is reported `present, unverified` because
    signature verification is not shipped. A laundering gate rides on the same
    analysis — verified flipping one test image from `REAL @ 0.919` to
    `UNCERTAIN` after a q=12 re-save.

## Immediate Next Steps

1. **EDITED/Tp_S recall has dropped in the same direction across two
   consecutive promotions (0.866→0.665→0.573) — don't promote a third
   round in this direction without addressing it first.** Likely needs
   EDITED's own training diversity grown the same way real/txt and real/ind
   just were (CASIA's Tp split is 100% synthetic splice/copy-move — real-world
   tampering, e.g. Photoshop retouch or AI inpainting, may not share the same
   cues its own held-out set rewards). Growing that pool, not further
   real-side rebalancing, is the next real fix.
2. ConvNeXt-tiny is still the original, untouched checkpoint (never promoted —
   see Honest Status #3). Worth a third attempt now that real/txt and
   real/ind have more source diversity, not just more oversampling of the
   same thin pool that made ConvNeXt's trade-off unfavorable last time.
3. Re-run `evaluate_by_category.py` after any retrain, against a **freshly
   re-measured** baseline if the val set composition changed — the aggregate
   metrics have already hidden a real result once, hidden a real non-result
   once, and once actually agreed with the per-category verdict. Don't
   assume which one it'll be.
4. Recalibrate (`fit_calibration.py`) after every checkpoint promotion —
   stale calibration on a shifted model silently miscalibrates confidence.
5. `subprocess.run(["kaggle", ...])` and Unicode status characters
   (`✓ → ─`) in print output both crashed background runs this session on
   Windows (bare command not on PATH; console codepage can't encode the
   characters when stdout is redirected to a log file). Fixed in
   `download_data.py`, `train.py`, `create_kaggle_dataset.py` — check any
   *new* training/data script for the same two patterns before backgrounding
   it, they aren't obviously related until they bite.
