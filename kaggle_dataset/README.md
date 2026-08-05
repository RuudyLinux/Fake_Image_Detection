# Fake Image Detection Dataset

3-class dataset for FakeGuard AI training.

## Classes
| Label | Folder | Description |
|-------|--------|-------------|
| 0 | `real/` | Authentic photographs — Flickr-Faces (`*.jpg`) + CASIA v2 Au, general photography, prefixed `casia_au_*` |
| 1 | `ai_generated/` | GAN / diffusion generated — StyleGAN (`*.jpg`) + Stable Diffusion, prefixed `sd14_*` |
| 2 | `edited/` | Copy-move, splicing, retouching — CASIA v2 Tp |

`real` and `ai_generated` each mix two sub-domains with a large size gap
(thousands of face photos vs. hundreds of injected images). training/dataset.py's
`make_sampler()` gives every sub-domain equal sampling weight within its class —
without that, the minority sub-domain is proportionally drowned out even
though the class itself is balanced. This was a real, measured bug: `real`
being 100% face photos with no non-face authentic examples caused any
non-face photo (a building, an animal, a landscape) to be misclassified as
EDITED with ~88% confidence, because `edited` (CASIA v2) spans general
photography while `real` did not.

## Structure
```
train/
  real/           (11,999 images — 8,000 faces + 3,999 casia_au_*)
  ai_generated/   (8,800 images — 8,000 StyleGAN + 800 sd14_*)
  edited/         (4,835 images)
val/
  real/           (3,001 images — 2,000 faces + 1,001 casia_au_*)
  ai_generated/   (2,200 images — 2,000 StyleGAN + 200 sd14_*)
  edited/         (1,209 images)
```

## Usage
Populated via Colab training notebook:
https://github.com/RuudyLinux/Fake-Image-Detections/blob/main/colab_fake_image_detection.ipynb

Diffusion images: `training/generate_diffusion_data.py`
CASIA Au/Tp split: `training/download_data.py` or `training/create_kaggle_dataset.py`

## Model
2-model calibrated ensemble (EfficientNet-B4 + ConvNeXt-Tiny), 3-class
CrossEntropyLoss, fine-tuned with a sub-domain-balanced sampler.
Full-val macro-F1: ~0.90 (this benchmark got substantially harder once `real`
stopped being face-only — see project history for the honest before/after).
