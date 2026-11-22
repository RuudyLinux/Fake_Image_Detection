"""
Multi-representation extraction — proposal §4.1.

Turns one RGB image into the 6 representations the fingerprint extractor
consumes: RGB, FFT magnitude, DCT magnitude, noise residual, texture (LBP),
edge (Sobel magnitude/orientation). All deterministic, non-trained, cheap
(no network weights) — same "no training set to overfit to" property the
existing blind_detector.py heuristics have, kept as a virtue here too:
these are the *inputs* to a trained encoder, not classifiers themselves.

Honest simplification vs. the proposal text: §4.1 describes DCT as a
block-histogram (JPEG-grid-aligned). This implementation uses a full-image
2D DCT log-magnitude instead — cheaper, no 8x8 blocking/histogram step, and
still captures the same "does energy concentrate where a real photo's JPEG
pipeline would put it" signal that the block-histogram version targets.
Revisit if the ablation in proposal §10 shows DCT contributing less than
FFT — that would be a real, reportable signal about which formulation
matters, not just an implementation detail.
"""

import numpy as np
import cv2
from scipy.fft import dctn
from skimage.feature import local_binary_pattern

REPRESENTATION_NAMES = ["rgb", "freq", "dct", "noise", "texture", "edge"]

# Channels per representation, for building each SmallEncoder with the right
# input depth.
REPRESENTATION_CHANNELS = {
    "rgb": 3,
    "freq": 1,
    "dct": 1,
    "noise": 3,
    "texture": 1,
    "edge": 2,  # magnitude + orientation
}

_LBP_RADIUS = 2
_LBP_POINTS = 8 * _LBP_RADIUS


def _to_gray(img_uint8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)


def _fft_magnitude(gray: np.ndarray) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    mag = np.log1p(np.abs(f))
    mag -= mag.min()
    denom = mag.max()
    if denom > 1e-8:
        mag /= denom
    return mag.astype(np.float32)


def _dct_magnitude(gray: np.ndarray) -> np.ndarray:
    d = dctn(gray.astype(np.float32), norm="ortho")
    mag = np.log1p(np.abs(d))
    mag -= mag.min()
    denom = mag.max()
    if denom > 1e-8:
        mag /= denom
    return mag.astype(np.float32)


def _noise_residual(img_uint8: np.ndarray) -> np.ndarray:
    """High-pass residual: image minus a mild Gaussian blur of itself. Same
    family as blind_detector.py's noise heuristic, kept as a full spatial
    residual map here rather than collapsed to a scalar — the fingerprint
    encoder needs spatial structure, a global score would throw it away."""
    blurred = cv2.GaussianBlur(img_uint8, (5, 5), sigmaX=1.0)
    residual = img_uint8.astype(np.float32) - blurred.astype(np.float32)
    # Center at 0.5, scale into [0, 1] — residuals are small (~±30 typical).
    residual = residual / 255.0 + 0.5
    return np.clip(residual, 0.0, 1.0).astype(np.float32)


def _texture_lbp(gray: np.ndarray) -> np.ndarray:
    lbp = local_binary_pattern(gray, P=_LBP_POINTS, R=_LBP_RADIUS, method="uniform")
    lbp = lbp / max(lbp.max(), 1.0)
    return lbp.astype(np.float32)


def _edge_sobel(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag /= max(mag.max(), 1e-8)
    orientation = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)  # -> [0, 1]
    return np.stack([mag, orientation], axis=0).astype(np.float32)


def extract_representations(img_uint8: np.ndarray) -> dict[str, np.ndarray]:
    """
    img_uint8: HxWx3 RGB, uint8.
    Returns dict of representation_name -> CxHxW float32 array in [0, 1],
    ready to torch.from_numpy() and batch.
    """
    assert img_uint8.ndim == 3 and img_uint8.shape[2] == 3, \
        f"expected HxWx3 RGB uint8, got shape {img_uint8.shape}"

    gray = _to_gray(img_uint8)
    rgb = (img_uint8.astype(np.float32) / 255.0).transpose(2, 0, 1)
    noise = _noise_residual(img_uint8).transpose(2, 0, 1)

    return {
        "rgb": rgb,
        "freq": _fft_magnitude(gray)[None, ...],
        "dct": _dct_magnitude(gray)[None, ...],
        "noise": noise,
        "texture": _texture_lbp(gray)[None, ...],
        "edge": _edge_sobel(gray),
    }
