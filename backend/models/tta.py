"""
Test-time augmentation for the forensic detector.

IMPORTANT domain constraint: most standard TTA recipes are wrong here.

This model decides using *forensic* evidence — spectral decay, GAN upsampling
periodicity, sensor noise residual, compression error. Any augmentation that
RESAMPLES the pixel grid (scaling, rotation by non-multiples of 90 deg,
perspective warp) runs its own interpolation filter over the image, which
smooths high-frequency content and rewrites exactly the artifacts being
measured. Averaging over resampled views does not reduce variance — it
averages in evidence about the resampler instead of the image.

So TTA here is restricted to transforms that permute pixels without
interpolating: horizontal flip, vertical flip, and 180 deg rotation. These
preserve the magnitude spectrum and the noise-residual statistics while still
giving the network genuinely different inputs.
"""

import torch


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _hflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-1])


def _vflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-2])


def _rot180(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-2, -1])


# Resample-free only — see module docstring.
TTA_TRANSFORMS = [_identity, _hflip, _vflip, _rot180]


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
