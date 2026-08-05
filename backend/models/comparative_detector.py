"""
Comparative detection: SSIM + pixel diff + contour-based region localization.
"""

import io
import uuid
import numpy as np
from PIL import Image
import cv2
import imagehash
from skimage.metrics import structural_similarity as ssim
from pathlib import Path

STATIC_DIR = Path(__file__).parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

MAX_DIM = 1024

# See the decision block in compare() for the measurements behind this value.
LOCALIZATION_RATIO_THRESHOLD = 20.0


def _localization_ratio(diff_gray: np.ndarray, regions: list) -> float:
    """
    How concentrated the difference is: mean absolute difference inside the
    flagged regions divided by the mean outside them.

    ~1 means the difference is spread evenly over the whole image (the
    signature of re-compression or resizing). A large value means it is
    confined to specific regions (the signature of local editing).
    """
    if not regions:
        return 0.0
    mask = np.zeros_like(diff_gray)
    cv2.drawContours(mask, regions, -1, 255, -1)
    inside_px = diff_gray[mask > 0]
    outside_px = diff_gray[mask == 0]
    if inside_px.size == 0 or outside_px.size == 0:
        return 0.0
    return float(inside_px.mean() / (outside_px.mean() + 1e-6))


def _load_resize(data: bytes, target_size: tuple | None = None) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if target_size:
        img = img.resize(target_size, Image.LANCZOS)
    else:
        w, h = img.size
        scale = min(MAX_DIM / w, MAX_DIM / h, 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img)


def _save_image(img: np.ndarray, prefix: str) -> str:
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.jpg"
    out_path = STATIC_DIR / filename
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return f"/static/{filename}"


def _hash_distances(original_data: bytes, suspected_data: bytes) -> tuple[int, int]:
    """
    Perceptual hash distance (Hamming, 0-64): near-0 means structurally identical
    even after resize/recompression; large jumps flag content-level changes that
    SSIM (pixel-aligned) can miss on shifted/rescaled/recompressed images.
    """
    orig_pil = Image.open(io.BytesIO(original_data)).convert("RGB")
    susp_pil = Image.open(io.BytesIO(suspected_data)).convert("RGB")
    phash_dist = imagehash.phash(orig_pil) - imagehash.phash(susp_pil)
    dhash_dist = imagehash.dhash(orig_pil) - imagehash.dhash(susp_pil)
    return int(phash_dist), int(dhash_dist)


def compare(original_data: bytes, suspected_data: bytes) -> dict:
    orig = _load_resize(original_data)
    h, w = orig.shape[:2]
    susp = _load_resize(suspected_data, target_size=(w, h))

    orig_gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    susp_gray = cv2.cvtColor(susp, cv2.COLOR_RGB2GRAY)

    # SSIM
    score, ssim_map = ssim(orig_gray, susp_gray, full=True)
    score = float(score)

    phash_distance, dhash_distance = _hash_distances(original_data, suspected_data)

    # Pixel diff map
    diff = cv2.absdiff(orig, susp)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    diff_norm = (diff_gray.astype(np.float32) / (diff_gray.max() + 1e-8) * 255).astype(np.uint8)
    diff_colored = cv2.applyColorMap(diff_norm, cv2.COLORMAP_HOT)
    diff_map_url = _save_image(cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB), "diff")

    # Highlight manipulated regions on suspected image.
    # Minimum region area scales with image size: a fixed 200px area is a large
    # fraction of a thumbnail and pure noise in a 1024px image.
    min_region_area = max(200, int(0.0005 * h * w))
    _, thresh = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    significant = [c for c in contours if cv2.contourArea(c) > min_region_area]
    highlighted = susp.copy()
    cv2.drawContours(highlighted, significant, -1, (255, 50, 50), 2)
    for c in significant:
        x, y, rw, rh = cv2.boundingRect(c)
        cv2.rectangle(highlighted, (x, y), (x + rw, y + rh), (255, 80, 80), 1)

    highlighted_url = _save_image(highlighted, "highlighted") if significant else None

    localization = _localization_ratio(diff_gray, significant)

    # Decision. Previously this was `score < 0.95 or len(significant) > 0 or
    # phash_distance > 10` — an OR of three loose thresholds, which reported
    # almost any re-compressed but unedited image as manipulated.
    #
    # SSIM magnitude cannot fix that, because it does not separate the two
    # cases at all. Measured on real images: a genuine copy-move splice scored
    # SSIM 0.955 while a q=30 re-save of the SAME unedited image scored 0.933 —
    # the clean image scored worse. No threshold on SSIM can work.
    #
    # What does separate them is the *spatial structure* of the difference.
    # Re-compression and resizing spread error diffusely over every pixel; a
    # splice concentrates it in a compact region and leaves the rest clean. The
    # localization ratio (mean difference inside the flagged regions vs. the
    # background difference) measures exactly that, and separates cleanly:
    #     re-compress q30 : 0.00 - 4.07
    #     0.5x resize     : 2.52 - 16.19
    #     copy-move splice: 79.57 - 290.78
    # Threshold set at 20 — above every clean case measured, ~4x below the
    # weakest edit, leaving margin for the blended (lower-ratio) splices that a
    # crude paste in the probe above does not represent.
    #
    # Caveat: calibrated on a 6-image synthetic probe, not a held-out benchmark.
    # Worth re-validating against CASIA's real Tp/Au source pairs, whose
    # filenames encode the authentic images each tampered image was built from.
    hash_evidence = phash_distance > 10 or dhash_distance > 12
    localized_evidence = bool(significant) and localization >= LOCALIZATION_RATIO_THRESHOLD

    is_manipulated = hash_evidence or localized_evidence

    reasons = []
    if hash_evidence:
        reasons.append(
            f"perceptual hash distance indicates content change "
            f"(pHash={phash_distance}, dHash={dhash_distance})"
        )
    if localized_evidence:
        reasons.append(
            f"{len(significant)} concentrated difference region(s); difference inside them is "
            f"{localization:.0f}x the background level, consistent with local editing"
        )
    if not is_manipulated and (significant or score < 0.95):
        reasons.append(
            f"differences are diffuse (localization ratio {localization:.1f}), "
            f"consistent with re-compression or resizing rather than editing"
        )

    return {
        "similarity_score": round(score, 4),
        "is_manipulated": bool(is_manipulated),
        "diff_map_url": diff_map_url,
        "highlighted_url": highlighted_url,
        "regions_count": len(significant),
        "phash_distance": phash_distance,
        "dhash_distance": dhash_distance,
        "localization_ratio": round(localization, 2),
        "evidence": {
            "hash_evidence": bool(hash_evidence),
            "localized_evidence": bool(localized_evidence),
            "reasons": reasons,
        },
    }
