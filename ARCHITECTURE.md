# FakeGuard AI — System Architecture

How the system actually works, end to end. For what's known-broken or
traded-off, see [PROJECT_SPEC.md](PROJECT_SPEC.md)'s Honest Status section —
this file explains the mechanism, that one tracks its current limitations.

## 1. What it does

Takes an image and answers one of two questions:

- **Blind detection** — "is this image REAL, AI_GENERATED, or EDITED?"
  No reference needed, single upload.
- **Comparative detection** — "what changed between these two images?"
  Original + suspected pair, localizes the difference.

Every blind verdict can also come back **UNCERTAIN** — the system is built
to refuse a guess rather than hand back a confident wrong answer.

## 2. Top-level flow

```mermaid
flowchart TB
    User(["User"]) -->|upload| FE["Frontend\nReact + Vite + TS + Tailwind"]
    FE -->|POST /detect/blind\nor /detect/compare| API["FastAPI backend\nbackend/main.py"]
    API --> Router["routers/detect.py"]
    Router -->|blind mode| Blind["blind_detector.analyze()"]
    Router -->|comparative mode| Comp["comparative_detector.compare()"]
    Blind --> Result1["verdict + confidence\n+ heatmap + reasons"]
    Comp --> Result2["manipulated? + diff map\n+ hash distances"]
    Result1 --> DB[("SQLite\nhistory.db")]
    Result2 --> DB
    Result1 --> FE
    Result2 --> FE
    Router -->|.../report| PDF["reportlab PDF export"]
```

## 3. Blind detection pipeline

This is the core of the system — two independent signal sources are fused,
then gated by an abstention check before anything is returned as a verdict.

```mermaid
flowchart TB
    Img["input image"] --> Ens["① CNN ENSEMBLE"]
    Img --> Heur["② FORENSIC HEURISTICS"]

    subgraph Ens[" "]
        direction TB
        EnsTitle["① CNN Ensemble — models/ensemble.py"]
        B4["EfficientNet-B4 · 380px\nweight 0.6"]
        CN["ConvNeXt-Tiny · 224px\nweight 0.4"]
        TTA["TTA: resample-free views\nmodels/tta.py"]
        Cal["temperature scaling\ncheckpoints/calibration.json"]
        B4 --> TTA
        CN --> TTA
        TTA --> Cal
        Cal --> Unc["uncertainty decomposition\naleatoric + epistemic (MI)"]
    end

    subgraph Heur[" "]
        direction TB
        HeurTitle["② Forensic Heuristics — models/blind_detector.py"]
        H1["spectral decay"]
        H2["GAN periodicity (FFT)"]
        H3["noise residual"]
        H4["texture uniformity"]
        H5["edge statistics"]
        H6["ELA (JPEG error level)"]
    end

    Unc --> Fuse["③ FUSION — models/fusion.py\ncnn_weight=0.85, heuristic_weight=0.15"]
    Heur --> Fuse

    Fuse --> Gate{"④ ABSTAIN?\nconfidence < 0.55\nOR disagreement (MI) > 0.35\nOR below user threshold"}
    Gate -->|yes| Uncertain["verdict = UNCERTAIN"]
    Gate -->|no| Verdict["verdict = REAL / AI_GENERATED / EDITED"]

    Verdict --> Cam{"is_fake AND\nenable_gradcam?"}
    Cam -->|yes| GC["⑤ Grad-CAM on B4\nmodels/gradcam.py"]
    Cam -->|no| Out
    Uncertain --> Out["response: prediction, confidence,\nper-class probs, member agreement,\nheuristic breakdown, heatmap url"]
    GC --> Out
```

### ① CNN Ensemble — the primary signal

Two independently-trained models, deliberately different architectures at
different input resolutions, so they fail on different inputs instead of
sharing blind spots:

| Model | Resolution | Weight | Why this weight |
|---|---|---|---|
| EfficientNet-B4 | 380px | 0.6 | finer input detail → more forensic signal retained |
| ConvNeXt-Tiny | 224px | 0.4 | secondary vote, catches what B4 misses |

Each member runs **test-time augmentation** (resample-free views only — a
lesson learned the hard way: augmented views that resample the image push it
out of the classifier's input distribution and *reduce* accuracy, see git
history `5da4bae`). Logits are weight-averaged, then passed through a fitted
**temperature scaling** factor (`calibration.json`) so the reported
confidence reflects real accuracy, not raw softmax overconfidence.

**Uncertainty isn't one number.** The ensemble decomposes predictive entropy
into:

- **Aleatoric** — the image is genuinely ambiguous; more models wouldn't help.
- **Epistemic** (mutual information between members) — the members
  *disagree*, the signature of an input unlike anything in training. This is
  the dangerous kind — a single model asked to extrapolate is often
  confidently wrong.

High epistemic uncertainty is what drives abstention, not just low confidence
— an input can have a "confident" softmax score from each individual member
while the members confidently disagree with each other.

### ② Forensic heuristics — the explainable cross-check

Six signal-based scores computed independently of any trained model
(`models/blind_detector.py`):

| Signal | What it catches |
|---|---|
| Spectral decay | unnatural frequency falloff from generative upsampling |
| GAN periodicity (FFT) | periodic peaks at N/2 multiples — a GAN upsampling fingerprint |
| Noise residual | inconsistent sensor-noise pattern across the image |
| Texture uniformity | over-smooth regions typical of generated content |
| Edge statistics | gradient anomalies at manipulation boundaries |
| ELA (error level analysis) | JPEG re-compression inconsistency at edited regions |

These don't have a training set to overfit to, so they act as an
independent cross-check on the CNN rather than a competing vote — see ③.

### ③ Fusion

`models/fusion.py` blends the two probability distributions:

```
fused[class] = 0.85 × cnn_prob[class] + 0.15 × heuristic_prob[class]
```

The CNN dominates (it's the far stronger signal — val macro-F1 in the 0.85-
0.90 range vs. heuristics alone); heuristics mainly matter when they
*disagree* with the CNN, which is surfaced to the user as a scrutiny flag
even though the fused verdict still follows the CNN.

### ④ Abstention gate

A verdict is downgraded to **UNCERTAIN** if any of:

- fused confidence < 0.55 (`MIN_CONFIDENCE`)
- member disagreement (mutual information) > 0.35 nats (`MAX_DISAGREEMENT`,
  out of ln(3) ≈ 1.0986 max possible for 3 classes)
- fused confidence < the caller-supplied `confidence_threshold` (default 0.5)

This is a deliberate design stance: an honest non-answer is preferred over a
confident wrong one.

### ⑤ Grad-CAM

Computed from the **EfficientNet-B4 member specifically**, not the fused
output — a fused score has no single feature map to attribute against, and
B4's 380px input gives a finer activation grid than ConvNeXt's 224px one.
Generated even when the verdict is UNCERTAIN, since that's exactly when a
human reviewer needs to look closest.

## 4. Comparative detection pipeline

```mermaid
flowchart LR
    O["original image"] --> Hash["pHash + dHash\ndistance"]
    S["suspected image"] --> Hash
    O --> ELA["ELA diff map"]
    S --> ELA
    Hash --> Dec{"manipulated?\nscore < 0.95 OR\nsignificant regions OR\nphash_distance > 10"}
    ELA --> Dec
    Dec --> Out["is_manipulated, similarity_score,\nphash/dhash distance, diff/highlight map"]
```

No trained model here — pure signal comparison
(`models/comparative_detector.py`). Perceptual hash distance catches
structural changes; the ELA diff localizes *where* pixels were
re-compressed differently, which is what actually gets drawn as the
highlighted region.

## 5. Serving stack

```mermaid
flowchart LR
    subgraph Frontend
        RC["React components\nUploadZone, ResultPanels,\nConfidenceBar, History"]
        Pages["pages: Home, BlindDetection,\nComparativeDetection, History"]
        API_client["api/detection.ts (axios)"]
    end
    subgraph Backend
        FastAPI["FastAPI app (main.py)"]
        Router2["routers/detect.py"]
        Models["models/ package\n(ensemble, blind_detector,\ncomparative_detector, fusion, gradcam)"]
        SQLite[("history.db")]
        Static["static/ (thumbnails, heatmaps)"]
    end
    RC --> Pages --> API_client -->|HTTP| FastAPI
    FastAPI --> Router2 --> Models
    Router2 --> SQLite
    Router2 --> Static
```

- **Frontend**: React + Vite + TypeScript + Tailwind, dark theme.
- **Backend**: FastAPI, CORS-scoped to the local dev frontend ports.
- **History**: every detection (blind or comparative) is persisted to SQLite
  with its thumbnail, retrievable via `/detect/history`.
- **Reports**: `.../report` endpoints render the same analysis into a PDF
  via `reportlab`.

## 6. Training pipeline (offline, not part of the request path)

```mermaid
flowchart TB
    Sources["Data sources\nFFHQ/140k faces · StyleGAN faces · SD1.4\nCASIA v2 (Tp + Au) · Indoor67 · TextOCR"]
    Sources --> DS["training/dataset.py\nFakeImageDataset"]
    DS --> Sampler["make_sampler()\nbalances: class, then\nsub-domain, then per-category\n(real/txt, real/ind, ...) and\nper-tamper-type (edited/Tp_S, ...)"]
    Sampler --> Train["training/train.py\ntwo-phase: freeze head → unfreeze,\nfine-tune with cosine LR"]
    Train --> Ckpt["checkpoints/*_fake.pth"]
    Ckpt --> EvalCat["evaluate_by_category.py\nper-subdomain accuracy —\ncatches what the aggregate\nmetric can't see"]
    Ckpt --> EvalEns["evaluate_ensemble.py\nfused accuracy + abstention coverage"]
    Ckpt --> Robust["robustness_eval.py\naccuracy under blur/compress/\nresize/noise"]
    EvalCat --> Decision{"beats baseline on\nthe metric that\nactually matters?"}
    Decision -->|yes| Promote["promote → live checkpoint\n+ fit_calibration.py"]
    Decision -->|no| Keep["keep current live checkpoint"]
```

The sampler is the load-bearing piece here — it's been extended three times
this project's life as thinner and thinner sub-populations turned out to be
invisible to coarser balancing (class-only → sub-domain → per-category /
per-tamper-type). See PROJECT_SPEC.md's Honest Status for the specific bugs
each extension fixed.

## 7. Directory map

```
backend/
  main.py                 FastAPI app entrypoint
  routers/detect.py       /detect/blind, /detect/compare, /history, .../report
  db.py                   SQLite history persistence
  models/
    ensemble.py           2-model weighted ensemble, calibration, abstention
    blind_detector.py     orchestrates heuristics + ensemble → verdict
    comparative_detector.py   pHash/dHash + ELA pairwise comparison
    fusion.py              CNN + heuristic probability blending
    gradcam.py             Grad-CAM heatmap generation
    tta.py                  test-time augmentation views
    cnn_model.py            model architectures (EfficientNet-B4, ConvNeXt-Tiny)
    calibration.py          temperature scaling load/apply
  training/
    dataset.py              FakeImageDataset + balanced sampler
    train.py                two-phase fine-tuning loop
    evaluate.py, evaluate_ensemble.py, evaluate_by_category.py,
    robustness_eval.py       metrics tooling
    fit_calibration.py       temperature fitting
    download_data.py         Kaggle dataset acquisition + organization
  checkpoints/               *.pth (gitignored) + calibration.json,
                              *_by_category.json (tracked — see .gitignore)
  reports/pdf_report.py       PDF export
frontend/
  src/pages/                 Home, BlindDetection, ComparativeDetection, History
  src/components/             UploadZone, ResultPanels, ConfidenceBar, Navbar
  src/api/detection.ts         axios client
```

## 8. Tech stack

| Layer | Tech |
|---|---|
| Model training | PyTorch, AMP, torchvision |
| Model architectures | EfficientNet-B4, ConvNeXt-Tiny (both ImageNet-pretrained, fine-tuned) |
| Backend | FastAPI, Uvicorn, SQLite |
| Image / signal processing | Pillow, OpenCV, scikit-image, SciPy, ImageHash |
| Reports | ReportLab (PDF) |
| Frontend | React, Vite, TypeScript, Tailwind CSS, axios |
| Dataset acquisition | Kaggle API |
