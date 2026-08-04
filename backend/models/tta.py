"""
Test-time augmentation for the forensic detector.

IMPORTANT — two constraints stack here, and missing either one breaks TTA.

1. RESAMPLE-FREE. This model decides using *forensic* evidence — spectral
   decay, GAN upsampling periodicity, sensor noise residual, compression
   error. Any augmentation that RESAMPLES the pixel grid (scaling, rotation
   by non-multiples of 90 deg, perspective warp) runs its own interpolation
   filter over the image, smoothing high-frequency content and rewriting
   exactly the artifacts being measured.

2. IN-DISTRIBUTION FOR THE CLASSIFIER. Resample-free is necessary but not
   sufficient: a view the network was never trained to be invariant to is
   simply a novel, out-of-distribution input, whatever its forensic
   properties. Measured directly — see training/evaluate_ensemble.py —
   adding vflip and 180-degree rotation to the TTA set dropped forced
   accuracy from 99.78% to 90.22%, because training/dataset.py only ever
   applies RandomHorizontalFlip (never vertical flip or 180-degree rotation).
   The network has learned invariance to a horizontal mirror and nothing
   else; showing it an upside-down face is not a robustness test, it is a
   different distribution, and averaging its confidently-wrong output into
   the ensemble actively hurts.

So TTA here is restricted to the one pixel-permuting, resample-free view the
model was actually trained to be invariant to: horizontal flip.
"""

import torch


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _hflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-1])


# Matches training augmentation exactly — see module docstring before adding
# vflip/rot180 back. They were tried and measured harmful.
TTA_TRANSFORMS = [_identity, _hflip]


@torch.no_grad()
def tta_logits(model, tensor: torch.Tensor, transforms=None) -> torch.Tensor:
    """
    Mean logits across resample-free views of `tensor` ([1, C, H, W]).

    Averaging in logit space (rather than probability space) keeps the
    downstream temperature calibration meaningful — the calibrated temperature
    is fitted against logits, so the ensemble must stay in that space.
    """
    transforms = transforms or TTA_TRANSFORMS
    accumulated = None
    for transform in transforms:
        logits = model(transform(tensor))
        accumulated = logits if accumulated is None else accumulated + logits
    return accumulated / len(transforms)
