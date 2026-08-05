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
  train/  real=11,999  ai_generated=8,800  edited=4,835   total=25,634
  val/    real=3,001   ai_generated=2,200  edited=1,209   total=6,410
```

`real` and `ai_generated` each mix a face-photo majority with an injected
non-face sub-domain (`casia_au_*`, `sd14_*`). `edited` (CASIA v2 Tp) has its
own internal tamper-type split (`Tp_S`/`Tp_D`/`Sp_S`/`Sp_D`). All of these are
balanced by [training/dataset.py](backend/training/dataset.py)'s
`make_sampler()` — see [Honest Status](#honest-status--known-limitations) for
why that had to be fixed twice, at two different granularities.

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

This section exists because "it works" claims in this project have twice
turned out to be measurement gaps, not fixed bugs — recorded here so the
next person (or session) doesn't have to rediscover them.

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

4. **txt (79 unique images) and ind (186) are CASIA's entire authentic pool
   for those categories — not a subsample.** Oversampling already extracts
   what's available (each txt image now sampled ~7.4x more per epoch); the
   real ceiling is more source images, not more training. This is the next
   open item — see below.

5. **EDITED's F1 dropped 0.997→0.819 after fix #1, and further on B4 after
   fix #2** (edited/Tp_S 0.866→0.665). The original high number was inflated
   by the same shortcut that broke non-face REAL — non-face-topic was a free
   tell for CASIA's tampered set too. The model now has to learn actual
   splice cues instead of guessing by subject matter; it's measurably worse
   at that than the old, shortcut-inflated number suggested.

6. Live model checkpoints (`backend/checkpoints/*.pth`) are **not** tracked
   in git (too large). The only rollback path after a promotion is whatever
   local backup file was kept at promotion time
   (e.g. `efficientnet_b4_fake_pre_v2_backup.pth`) — these backups don't
   survive a fresh clone.

## Immediate Next Steps

1. **Grow txt/ind source data.** CASIA's Au split is exhausted for these
   categories; need a new authentic-photo source (candidates: MIT
   Indoor67 / Places365 subset for `ind`, ICDAR-style scene-text photos for
   `txt`) via Kaggle, integrated the same way CASIA's Au was — new
   `SUBDOMAIN_PREFIXES` entry, sampler-balanced, then retrained and
   evaluated per-category before promoting.
2. Re-run `evaluate_by_category.py` after any retrain — the aggregate
   metrics have already hidden a real result once (EfficientNet-B4 epoch3)
   and a real non-result once (ConvNeXt); don't trust the aggregate alone
   again.
3. Consider whether EDITED needs real-world (non-CASIA-synthetic) tampering
   examples — CASIA's splice ops may not represent what an actual malicious
   edit (Photoshop retouch, AI inpaint) looks like as well as its own held-out
   set suggests.
4. Recalibrate (`fit_calibration.py`) after every checkpoint promotion —
   stale calibration on a shifted model silently miscalibrates confidence.
