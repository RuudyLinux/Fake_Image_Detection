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

### Trained models — 2, ensembled

| Model | Weight | Input | Role |
| --- | --- | --- | --- |
| EfficientNet-B4 | 0.6 | 380px | primary |
| ConvNeXt-Tiny | 0.4 | 224px | secondary — deliberately kept architecturally distinct so the pair disagrees on genuinely ambiguous images instead of sharing blind spots |

Ensemble ([backend/models/ensemble.py](backend/models/ensemble.py)) combines
weighted logits, applies fitted temperature scaling
([backend/checkpoints/calibration.json](backend/checkpoints/calibration.json)),
and abstains ("uncertain") when confidence is low or members disagree —
honest non-answer beats a confident wrong one.

### Supporting detection tools — 4, non-trained

- **blind_detector.py** — FFT spectral analysis (GAN periodicity), ELA,
  noise/texture heuristics; fallback 3-class signal.
- **comparative_detector.py** — pHash + dHash distance, ELA diff, for
  original-vs-suspect pair comparison.
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
