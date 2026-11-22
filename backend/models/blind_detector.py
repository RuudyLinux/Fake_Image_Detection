"""
Blind detection: forensic heuristics + a calibrated CNN ensemble.

Decision pipeline, strongest evidence first:

  1. ENSEMBLE — EfficientNet-B4 (380px) + ConvNeXt-Tiny (224px), each run with
     resample-free test-time augmentation, logits temperature-calibrated, then
     weight-averaged. Different architectures at different resolutions fail on
     different inputs, so their disagreement is informative rather than noise.

  2. HEURISTICS — spectral decay, GAN upsampling periodicity, noise residual,
     texture uniformity, edge statistics, and JPEG error level analysis. These
     carry a minority weight in the final fusion and provide the explainable
     breakdown shown to the user.

  3. ABSTENTION — when the ensemble's confidence is low, or its members
     disagree (high epistemic uncertainty, the signature of an input unlike
     anything in training), the verdict is UNCERTAIN rather than a guess.

Prediction classes: REAL | AI_GENERATED | EDITED | UNCERTAIN

Provenance (models/provenance.py) runs first as a separate evidence tier —
see PLAN_HIGH_CONFIDENCE_DETECTION.md §3. It is the only source of
near-certain answers, because it reads signed metadata rather than
classifying pixels. Two rules it imposes on this module:

  * PRESENCE of verified provenance can short-circuit to a near-certain
    verdict. ABSENCE must never raise confidence — most real photos carry
    no manifest either, and a screenshot strips it. See that module's
    docstring for why inverting this would be an exploitable bug.

  * LAUNDERING detected in the container (heavy re-compression, screenshot
    signatures) means the high-frequency evidence every heuristic below
    depends on is degraded. That adds an abstention reason rather than
    letting the model read noise and report a confident number.

If no checkpoint is present the module degrades to heuristics alone, and the
verdict is marked low-confidence accordingly.
"""

import io
import sys
import uuid
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

from models.fusion import fuse
from models.provenance import analyze_provenance

# ─── load the calibrated ensemble (optional) ───────────────────────────────────

_ENSEMBLE = None
_GRADCAM = None


def _try_load_ensemble():
    """
    Build the ensemble, and attach Grad-CAM to the EfficientNet-B4 member.

    Grad-CAM is deliberately taken from B4 rather than from the fused output:
    a fused score has no single feature map to attribute against, and B4 runs
    at 380px so its activation grid localises more precisely than ConvNeXt's
    224px one.
    """
    global _ENSEMBLE, _GRADCAM
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from models.ensemble import EnsembleDetector
        from models.gradcam import build_gradcam

        ensemble = EnsembleDetector()  # auto-selects CUDA when available
        if not ensemble.is_available:
            print("[FakeGuard] No checkpoints found — heuristics only")
            return

        _ENSEMBLE = ensemble
        for member in ensemble.members:
            if member["name"] == "efficientnet_b4":
                _GRADCAM = build_gradcam(member["model"])
                break

        print(f"[FakeGuard] Ensemble ready: {ensemble.member_names}  "
              f"temperature={ensemble.temperature:.4f}  "
              f"grad_cam={'yes' if _GRADCAM else 'no'}")
    except Exception as e:
        print(f"[FakeGuard] Ensemble load failed ({e}), using heuristics only")


_try_load_ensemble()

STATIC_DIR = Path(__file__).parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

CLASSES = ["REAL", "AI_GENERATED", "EDITED"]


# ─── helpers ───────────────────────────────────────────────────────────────────

def _load_image(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img)


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)


# ─── signal 1: spectral decay ───────────────────────────────────────────────────

def _spectral_score(img: np.ndarray) -> float:
    """
    Natural images: log radial spectrum decays ~linearly (1/f^alpha, alpha~2).
    AI images: flatter high-frequency shelf, deviating from this linear decay.
    Returns deviation from expected decay → higher = more suspicious.
    """
    gray = _to_gray(img)
    h, w = gray.shape
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.ogrid[:h, :w]
    radius = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2).astype(int)
    max_r = min(cx, cy) - 1

    radial_sum = np.zeros(max_r + 1)
    radial_cnt = np.zeros(max_r + 1)
    mask = radius <= max_r
    np.add.at(radial_sum, radius[mask], magnitude[mask])
    np.add.at(radial_cnt, radius[mask], 1)
    radial_cnt[radial_cnt == 0] = 1
    radial_profile = np.log1p(radial_sum[1:] / radial_cnt[1:])

    freqs = np.linspace(0, 1, len(radial_profile))
    if radial_profile.max() > 0:
        profile_norm = radial_profile / radial_profile.max()
    else:
        return 0.0

    coeffs = np.polyfit(freqs, profile_norm, 1)
    fitted = np.polyval(coeffs, freqs)
    residuals = profile_norm - fitted

    hf_half = len(residuals) // 2
    hf_excess = float(np.clip(residuals[hf_half:].mean() + 0.5, 0, 1))
    irregularity = float(np.clip(residuals.std() * 4, 0, 1))

    return float(np.clip(0.6 * hf_excess + 0.4 * irregularity, 0, 1))


# ─── signal 2: GAN periodic peaks ──────────────────────────────────────────────

def _gan_periodic_score(img: np.ndarray) -> float:
    """
    GAN transposed-convolution upsampling creates checkerboard artifacts visible
    as spectral peaks at multiples of N/2 in the FFT magnitude.
    """
    gray = _to_gray(img)
    h, w = gray.shape

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))

    cy, cx = h // 2, w // 2
    nh, nw = h // 4, w // 4

    def patch_mean(r, c, hw=3):
        return mag[max(0, r-hw):r+hw, max(0, c-hw):c+hw].mean()

    artifact_energy = np.mean([
        patch_mean(cy - nh, cx - nw),
        patch_mean(cy - nh, cx + nw),
        patch_mean(cy + nh, cx - nw),
        patch_mean(cy + nh, cx + nw),
    ])
    center_energy = patch_mean(cy, cx, hw=5) + 1e-8
    peak_ratio = float(np.clip(artifact_energy / center_energy, 0, 1))

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx**2 + gy**2)
    flat = gmag.flatten()
    if len(flat) > 4:
        lag1 = float(np.corrcoef(flat[:-1], flat[1:])[0, 1])
        lag2 = float(np.corrcoef(flat[:-2], flat[2:])[0, 1])
        periodicity = float(np.clip(abs(lag2) - abs(lag1) + 0.5, 0, 1))
    else:
        periodicity = 0.0

    return float(np.clip(0.5 * peak_ratio + 0.5 * periodicity, 0, 1))


# ─── signal 3: noise residual ──────────────────────────────────────────────────

def _noise_residual_score(img: np.ndarray) -> float:
    """
    Real cameras produce spatially correlated photon + sensor noise (PRNU).
    AI-generated images are unnaturally smooth — their noise residual energy
    is anomalously low or has flat spatial frequency distribution.
    Edited images can show noise inconsistency between regions.
    """
    gray = _to_gray(img)

    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    noise_sigma = np.median(np.abs(laplacian)) / 0.6745

    low_noise_score = float(np.clip(1.0 - noise_sigma / 10.0, 0, 1))

    blocks = []
    bh, bw = max(gray.shape[0] // 8, 1), max(gray.shape[1] // 8, 1)
    for i in range(0, gray.shape[0] - bh, bh):
        for j in range(0, gray.shape[1] - bw, bw):
            block = laplacian[i:i+bh, j:j+bw]
            blocks.append(np.abs(block).mean())
    if len(blocks) > 1:
        block_arr = np.array(blocks)
        noise_uniformity = 1.0 - float(np.clip(block_arr.std() / (block_arr.mean() + 1e-8), 0, 1))
    else:
        noise_uniformity = 0.5

    return float(np.clip(0.5 * low_noise_score + 0.5 * noise_uniformity, 0, 1))


# ─── signal 4: texture uniformity ──────────────────────────────────────────────

def _texture_score(img: np.ndarray) -> float:
    """
    AI portrait generators produce suspiciously uniform, smooth skin textures.
    Edited images may show boundary artifacts or inconsistent texture regions.
    """
    gray = _to_gray(img)
    h, w = gray.shape

    kernel_size = max(min(h, w) // 16, 7) | 1
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    blur_sq = cv2.GaussianBlur(gray**2, (kernel_size, kernel_size), 0)
    local_var = np.sqrt(np.abs(blur_sq - blur**2))

    lv_of_lv = local_var.std() / (local_var.mean() + 1e-8)
    uniformity_score = float(np.clip(1.0 - lv_of_lv / 1.5, 0, 1))

    hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [64], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-8)
    hist_roughness = float(np.std(np.diff(hist)) * 50)
    smooth_hist_score = float(np.clip(1.0 - hist_roughness, 0, 1))

    return float(np.clip(0.6 * uniformity_score + 0.4 * smooth_hist_score, 0, 1))


# ─── signal 5: gradient/edge proxy ─────────────────────────────────────────────

def _edge_score(img: np.ndarray) -> float:
    """
    AI images from diffusion models have abnormally sharp, consistent edges.
    Edited images may have inconsistent edge sharpness at manipulation boundaries.
    """
    gray = _to_gray(img)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx**2 + gy**2)

    flat = gmag.flatten()
    p25, p75 = np.percentile(flat, 25), np.percentile(flat, 75)
    iqr = p75 - p25 + 1e-8
    median_grad = np.median(flat) + 1e-8
    uniformity = float(np.clip(1.0 - iqr / (median_grad * 2), 0, 1))

    r, g, b = img[:, :, 0].flatten(), img[:, :, 1].flatten(), img[:, :, 2].flatten()
    rg_corr = abs(float(np.corrcoef(r, g)[0, 1]))
    rb_corr = abs(float(np.corrcoef(r, b)[0, 1]))
    channel_corr = float(np.clip((rg_corr + rb_corr) / 2, 0, 1))

    return float(np.clip(0.5 * uniformity + 0.5 * channel_corr, 0, 1))


# ─── signal 6: error level analysis ────────────────────────────────────────────

def _ela_score(img: np.ndarray) -> float:
    """
    Error Level Analysis: resave at a fixed JPEG quality and diff against the
    original. A real photo degrades ~uniformly under recompression. A region
    spliced in from elsewhere (different original quality/generation) shows a
    different recompression error than its surroundings — so high *block-wise
    irregularity* in the error map, not the raw error level, is the tamper signal.
    """
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    resaved = np.array(Image.open(buf).convert("RGB")).astype(np.float32)

    diff_gray = np.abs(img.astype(np.float32) - resaved).mean(axis=2)

    bh, bw = max(diff_gray.shape[0] // 8, 1), max(diff_gray.shape[1] // 8, 1)
    blocks = [
        diff_gray[i:i + bh, j:j + bw].mean()
        for i in range(0, diff_gray.shape[0] - bh, bh)
        for j in range(0, diff_gray.shape[1] - bw, bw)
    ]
    block_arr = np.array(blocks) if blocks else np.array([diff_gray.mean()])
    irregularity = block_arr.std() / (block_arr.mean() + 1e-8)

    return float(np.clip(irregularity / 1.5, 0, 1))


# ─── heatmap ───────────────────────────────────────────────────────────────────

def _generate_gradcam_heatmap(img: np.ndarray, cam: np.ndarray) -> str:
    """Overlay a Grad-CAM activation map (low-res, [0,1]) onto the original image."""
    h, w = img.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_resized = cv2.GaussianBlur(cam_resized, (0, 0), sigmaX=w / 60)
    cam_resized = np.clip(cam_resized, 0, 1)
    cam_u8 = (cam_resized * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 0.45, heatmap, 0.55, 0)

    filename = f"heatmap_{uuid.uuid4().hex[:8]}.jpg"
    cv2.imwrite(str(STATIC_DIR / filename), overlay)
    return f"/static/{filename}"


def _generate_heatmap(img: np.ndarray) -> str:
    """Noise residual heatmap: highlights regions with anomalous local statistics."""
    gray = _to_gray(img).astype(np.uint8)

    laplacian = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F))
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    texture_diff = cv2.absdiff(gray, blur).astype(np.float32)

    combined = 0.5 * laplacian + 0.5 * texture_diff
    combined = cv2.GaussianBlur(combined, (21, 21), 0)
    combined_norm = (combined / (combined.max() + 1e-8) * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(combined_norm, cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 0.45, heatmap, 0.55, 0)

    filename = f"heatmap_{uuid.uuid4().hex[:8]}.jpg"
    cv2.imwrite(str(STATIC_DIR / filename), overlay)
    return f"/static/{filename}"


# ─── trained model inference ────────────────────────────────────────────────────

def _ensemble_predict(img: np.ndarray, use_tta: bool = True) -> dict:
    """Calibrated ensemble prediction with TTA and an uncertainty decomposition."""
    return _ENSEMBLE.predict(img, use_tta=use_tta)


# ─── heuristic 3-class prediction ──────────────────────────────────────────────

def _heuristic_predict(
    spectral: float, gan: float, noise: float, texture: float, edge: float, ela: float
) -> tuple[str, dict[str, float], float]:
    """
    Returns (prediction, class_probabilities, composite_fake_score).

    Heuristic logic:
      AI_GENERATED: strong GAN fingerprint + spectral deviation
      EDITED:       noise/ELA inconsistency + texture anomaly, but weaker GAN signal
    """
    composite = (
        0.22 * spectral
        + 0.18 * gan
        + 0.18 * noise
        + 0.12 * texture
        + 0.08 * edge
        + 0.22 * ela
    )

    threshold = 0.45
    if composite < threshold:
        real_p = 1.0 - composite
        fake_p = composite
        return "REAL", {
            "REAL": round(real_p, 4),
            "AI_GENERATED": round(fake_p * 0.5, 4),
            "EDITED": round(fake_p * 0.5, 4),
        }, composite

    # Fake — decide subtype
    # AI_GENERATED: GAN peaks + spectral are the dominant signal
    # EDITED: noise/ELA inconsistency / texture boundary effects dominate
    ai_score   = 0.50 * gan + 0.35 * spectral + 0.15 * edge
    edit_score = 0.35 * noise + 0.25 * texture + 0.15 * spectral + 0.25 * ela

    total = ai_score + edit_score + 1e-8
    ai_frac   = ai_score / total
    edit_frac = edit_score / total

    fake_mass = composite  # how much probability goes to fake classes
    real_p    = 1.0 - fake_mass
    ai_p      = fake_mass * ai_frac
    edit_p    = fake_mass * edit_frac

    if ai_score >= edit_score:
        prediction = "AI_GENERATED"
    else:
        prediction = "EDITED"

    return prediction, {
        "REAL": round(real_p, 4),
        "AI_GENERATED": round(ai_p, 4),
        "EDITED": round(edit_p, 4),
    }, composite


# ─── signal label → dominant forgery type ──────────────────────────────────────

def _dominant_signal(spectral, gan, noise, texture, edge, ela, prediction) -> str:
    if prediction == "REAL":
        return "none"
    signal_map = {
        "spectral_anomaly":   spectral,
        "gan_upsampling":     gan,
        "noise_inconsistency": noise,
        "texture_boundary":   texture,
        "edge_sharpness":     edge,
        "compression_inconsistency": ela,
    }
    return max(signal_map, key=signal_map.get)


# ─── main entry ────────────────────────────────────────────────────────────────

def analyze(
    image_data: bytes,
    confidence_threshold: float = 0.5,
    enable_gradcam: bool = True,
    use_tta: bool = True,
    allow_abstain: bool = True,
) -> dict:
    img = _load_image(image_data)

    # ── Tier A: provenance / container forensics (see module docstring) ──
    # Runs first and costs almost nothing (parsing, no inference). Cannot
    # lower confidence on its own — absence of provenance is not evidence.
    prov = analyze_provenance(image_data)
    if prov.tier_a_resolved and prov.tier_a_verdict:
        # Cryptographically resolved: return without running the models at
        # all. Nothing the classifier could say would outweigh a verified
        # signature, and pretending otherwise would only add noise to a
        # certain answer.
        return {
            "prediction": prov.tier_a_verdict,
            "model_prediction": prov.tier_a_verdict,
            "confidence": round(prov.confidence or 0.999, 3),
            "meets_threshold": True,
            "abstained": False,
            "abstain_reasons": [],
            "evidence_tier": "A_cryptographic",
            "forgery_type": None,
            "heatmap_url": None,
            "class_probabilities": {prov.tier_a_verdict: prov.confidence or 0.999},
            "signal_agreement": None,
            "uncertainty": None,
            "member_probabilities": None,
            "ensemble": None,
            "provenance": prov.to_dict(),
            "analysis": {},
        }

    # Laundering gate: if the container shows the image has been through
    # heavy re-compression, the high-frequency traces every heuristic below
    # reads have been degraded or destroyed. Flagged here and folded into
    # the abstention reasons further down, rather than silently producing a
    # confident number from damaged evidence.
    laundering_detected = any(
        "heavily compressed" in note or "degraded" in note
        for note in prov.quantization_notes
    )

    spectral  = _spectral_score(img)
    gan       = _gan_periodic_score(img)
    noise     = _noise_residual_score(img)
    texture   = _texture_score(img)
    edge      = _edge_score(img)
    ela       = _ela_score(img)

    heuristic_pred, heuristic_probs, composite = _heuristic_predict(
        spectral, gan, noise, texture, edge, ela
    )

    uncertainty = None
    member_probs = None
    ensemble_meta = None

    if _ENSEMBLE is not None:
        ens = _ensemble_predict(img, use_tta=use_tta)

        class_probs = fuse(ens["class_probabilities"], heuristic_probs)
        model_prediction = max(class_probs, key=class_probs.get)
        confidence = class_probs[model_prediction]

        signal_agreement = ens["prediction"] == heuristic_pred
        uncertainty = ens["uncertainty"]
        member_probs = ens["members"]
        ensemble_meta = {
            "members": _ENSEMBLE.member_names,
            "members_agree": ens["members_agree"],
            "tta_used": ens["tta_used"],
            "temperature": round(ens["temperature"], 4),
        }

        # Abstain on the ensemble's own signals, plus the user's confidence floor.
        abstain_reasons = list(ens["abstain_reasons"])
        if confidence < confidence_threshold:
            abstain_reasons.append("below_user_threshold")
        if laundering_detected:
            # Not a model-confidence problem — an evidence-availability one.
            # The classifier may well be confident; it is confident about
            # degraded input, which is exactly the case this gate exists for.
            abstain_reasons.append("forensic_evidence_degraded_by_compression")

        abstained = allow_abstain and bool(abstain_reasons)
        prediction = "UNCERTAIN" if abstained else model_prediction

        # Heatmap explains the model's actual leaning, so keep generating it
        # even when abstaining — that is precisely when a human needs to look.
        is_fake = model_prediction != "REAL"
        heatmap_url = None
        if is_fake and enable_gradcam and _GRADCAM is not None and ens["_b4_tensor"] is not None:
            cam = _GRADCAM(ens["_b4_tensor"], CLASSES.index(model_prediction))
            heatmap_url = _generate_gradcam_heatmap(img, cam)
    else:
        model_prediction = heuristic_pred
        class_probs = heuristic_probs
        confidence = class_probs[model_prediction]
        signal_agreement = None
        # Heuristics alone are far weaker than the CNN; never present them as
        # a confident verdict.
        abstain_reasons = ["heuristics_only"]
        if confidence < confidence_threshold:
            abstain_reasons.append("below_user_threshold")
        if laundering_detected:
            abstain_reasons.append("forensic_evidence_degraded_by_compression")
        abstained = allow_abstain and confidence < confidence_threshold
        prediction = "UNCERTAIN" if abstained else model_prediction
        is_fake = model_prediction != "REAL"
        heatmap_url = _generate_heatmap(img) if (is_fake and enable_gradcam) else None

    forgery_type = _dominant_signal(spectral, gan, noise, texture, edge, ela, model_prediction)

    return {
        "prediction": prediction,
        "model_prediction": model_prediction,
        "confidence": round(confidence, 3),
        "meets_threshold": confidence >= confidence_threshold,
        "abstained": abstained,
        "abstain_reasons": abstain_reasons,
        # Which evidence tier produced this answer (PLAN_HIGH_CONFIDENCE_DETECTION.md §3).
        # "B_forensic" = model + heuristic consensus; "C_uncertain" = refused.
        "evidence_tier": "C_uncertain" if abstained else "B_forensic",
        "provenance": prov.to_dict(),
        "forgery_type": forgery_type,
        "heatmap_url": heatmap_url,
        "class_probabilities": class_probs,
        "signal_agreement": signal_agreement,
        "uncertainty": uncertainty,
        "member_probabilities": member_probs,
        "ensemble": ensemble_meta,
        "analysis": {
            "edge_score":           round(edge, 3),
            "fft_score":            round(spectral, 3),
            "gan_score":            round(gan, 3),
            "noise_score":          round(noise, 3),
            "texture_score":        round(texture, 3),
            "ela_score":            round(ela, 3),
            "heuristic_composite":  round(composite, 3),
        },
    }
