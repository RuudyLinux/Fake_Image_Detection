"""
Tier A — provenance and container forensics.
See PLAN_HIGH_CONFIDENCE_DETECTION.md §3 for why this tier exists.

This is the only path in the entire system to *near-certain* answers,
because it does not classify anything: it reads signed metadata and
container structure. Math and parsing, not statistics. A valid C2PA
signature identifies an image's origin with cryptographic confidence; no
model inference is involved and no accuracy number applies.

──────────────────────────────────────────────────────────────────────────
THE ASYMMETRY — the single most important rule in this module
──────────────────────────────────────────────────────────────────────────
    PRESENCE of a valid provenance manifest  -> STRONG evidence.
    ABSENCE of one                            -> PROVES NOTHING.

Most tools emit no manifest at all, and any manifest is destroyed by a
screenshot, a re-save, or a social-media upload. Treating absence as
evidence of authenticity would be a serious and trivially exploitable bug:
an attacker strips the manifest and the system upgrades its confidence.
Every function below returns findings that can only ever *raise* certainty,
never lower it, and `absence_is_not_evidence` is returned explicitly so
callers cannot accidentally invert the logic.
──────────────────────────────────────────────────────────────────────────

Scope note, stated honestly: full C2PA *signature verification* requires the
c2pa library and a trust list, which this project does not currently ship.
What this module does today is detect a manifest's PRESENCE (JUMBF/C2PA
markers in the container) and report it as `unverified_manifest_present` —
useful evidence, but explicitly NOT the cryptographic guarantee that a
verified signature would provide. Do not report it to the user as verified.
Same for SynthID: detection requires Google's verifier, which is not
publicly available as a library — the hook is defined here and disabled,
rather than faked.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field, asdict

from PIL import Image, ExifTags

# C2PA embeds its manifest in a JUMBF (JPEG Universal Metadata Box Format)
# box. These byte markers identify one without parsing the full structure.
_C2PA_MARKERS = (b"c2pa", b"jumb", b"urn:uuid:", b"c2pa.assertions")
# Software tags that indicate a generative tool touched the file. Presence is
# suggestive, not conclusive — these strings are trivially forged or absent.
_GENERATIVE_SOFTWARE_HINTS = (
    "stable diffusion", "midjourney", "dall-e", "dalle", "flux",
    "firefly", "generative", "comfyui", "automatic1111", "invokeai",
    "novelai", "imagen", "gpt-image",
)
_CAMERA_EXIF_TAGS = ("Make", "Model", "LensModel", "ExposureTime", "FNumber",
                     "ISOSpeedRatings", "FocalLength", "DateTimeOriginal")


@dataclass
class ProvenanceFindings:
    """Evidence only. Never a verdict on its own — see module docstring."""
    # Tier A: cryptographic / signed
    c2pa_manifest_present: bool = False
    c2pa_verified: bool = False          # stays False until real signature verification ships
    synthid_checked: bool = False        # stays False — no public verifier available
    # Container / metadata forensics
    has_exif: bool = False
    camera_evidence: dict = field(default_factory=dict)
    camera_evidence_strength: str = "none"      # none | weak | moderate | strong
    software_tags: list = field(default_factory=list)
    generative_software_hint: str | None = None
    jpeg_quantization_fingerprint: str | None = None
    quantization_notes: list = field(default_factory=list)
    thumbnail_mismatch: bool | None = None
    # Interpretation
    tier_a_resolved: bool = False
    tier_a_verdict: str | None = None    # "AI_GENERATED" | "REAL" | None
    confidence: float | None = None
    reasons: list = field(default_factory=list)
    absence_is_not_evidence: bool = True  # see module docstring; never flip this

    def to_dict(self) -> dict:
        return asdict(self)


def _scan_c2pa(data: bytes) -> bool:
    """Presence check only — NOT signature verification. See module docstring.
    Scans the first 256KB, where a JUMBF box sits in practice; scanning the
    whole file on every upload is wasted work for a marker that is always
    near the front."""
    head = data[:262144]
    hits = sum(1 for m in _C2PA_MARKERS if m in head)
    # Require 2+ distinct markers: "jumb" alone appears in unrelated binary
    # data often enough to produce false positives on ordinary photos.
    return hits >= 2


def _read_exif(img: Image.Image) -> tuple[dict, list]:
    exif_raw = img.getexif()
    if not exif_raw:
        return {}, []
    tag_names = {v: k for k, v in ExifTags.TAGS.items()}
    exif = {}
    for tag_id, value in exif_raw.items():
        name = ExifTags.TAGS.get(tag_id, str(tag_id))
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8", errors="replace").strip("\x00")
            except Exception:
                value = f"<{len(value)} bytes>"
        exif[name] = value

    software = []
    for key in ("Software", "ProcessingSoftware", "HostComputer"):
        if key in exif and exif[key]:
            software.append(f"{key}={exif[key]}")
    return exif, software


def _camera_strength(exif: dict) -> tuple[dict, str]:
    """How much genuine capture evidence is present. A real camera writes a
    coherent cluster of these; a generator writes none, and a stripped photo
    also writes none — which is exactly why absence is not evidence."""
    found = {t: exif[t] for t in _CAMERA_EXIF_TAGS if t in exif and exif[t] not in (None, "")}
    n = len(found)
    if n == 0:
        strength = "none"
    elif n <= 2:
        strength = "weak"
    elif n <= 5:
        strength = "moderate"
    else:
        strength = "strong"
    return found, strength


def _quantization_fingerprint(img: Image.Image) -> tuple[str | None, list]:
    """JPEG quantization tables are a device/encoder fingerprint. Standard
    camera encoders and known software encoders use characteristic tables;
    a mismatch between the claimed EXIF Make/Model and the actual table is a
    classical forensic red flag. Reported here as a fingerprint string plus
    observations — full table-to-device matching needs a reference database
    this project does not ship, so no device claim is made from it."""
    notes = []
    q = getattr(img, "quantization", None)
    if not q:
        return None, notes

    # Compact, comparable signature: first 8 coefficients of each table.
    parts = []
    for table_id in sorted(q.keys()):
        coeffs = list(q[table_id])[:8]
        parts.append(f"{table_id}:" + ",".join(str(c) for c in coeffs))
    fingerprint = "|".join(parts)

    luma = list(q[sorted(q.keys())[0]])
    if all(c == 1 for c in luma[:16]):
        notes.append("quantization table is near-lossless (q~100) — unusual for a "
                     "camera original, common for a re-saved or generated file")
    elif max(luma[:16]) > 40:
        notes.append("coarse quantization — heavily compressed; low-level forensic "
                     "evidence is likely degraded (see laundering gate)")
    if len(q) == 1:
        notes.append("single quantization table (grayscale or unusual encoder)")
    return fingerprint, notes


def _thumbnail_mismatch(img: Image.Image, data: bytes) -> bool | None:
    """An EXIF thumbnail is written at capture and not always regenerated on
    edit — a thumbnail that disagrees with the full image is classical
    evidence of manipulation. Returns None when there is no thumbnail to
    compare, which is the common case and is NOT suspicious."""
    exif = img.getexif()
    if not exif:
        return None
    thumb_bytes = None
    try:
        thumb_bytes = exif.get_ifd(0x5100)  # thumbnail IFD, when present
    except Exception:
        return None
    if not thumb_bytes:
        return None
    try:
        # Only a coarse comparison — a proper check would compare downsampled
        # pixel statistics; this flags gross mismatch only.
        thumb = Image.open(io.BytesIO(bytes(thumb_bytes))).convert("RGB")
        full = img.convert("RGB").resize(thumb.size)
        import numpy as np
        diff = float(np.abs(np.asarray(thumb, dtype=float) -
                            np.asarray(full, dtype=float)).mean())
        return diff > 40.0
    except Exception:
        return None


def analyze_provenance(data: bytes) -> ProvenanceFindings:
    """
    Entry point. Returns evidence, never a standalone verdict except in the
    one case where the evidence is cryptographic (Tier A resolved).

    Callers MUST respect `absence_is_not_evidence`: when nothing is found,
    this function returns `tier_a_resolved=False` and the caller falls
    through to Tier B. It must NOT interpret an empty result as "real".
    """
    f = ProvenanceFindings()

    try:
        img = Image.open(io.BytesIO(data))
    except Exception as exc:
        f.reasons.append(f"could not parse image container: {exc}")
        return f

    # ── C2PA ────────────────────────────────────────────────────────────
    f.c2pa_manifest_present = _scan_c2pa(data)
    if f.c2pa_manifest_present:
        # Deliberately NOT setting c2pa_verified — presence != verified.
        # Verification needs the c2pa library plus a trust list; claiming
        # verification here would be a false cryptographic guarantee.
        f.reasons.append(
            "C2PA/JUMBF manifest detected in container (PRESENCE only — signature "
            "NOT verified; this system does not yet ship signature verification)"
        )

    # ── EXIF / software ─────────────────────────────────────────────────
    exif, software = _read_exif(img)
    f.has_exif = bool(exif)
    f.software_tags = software
    blob = " ".join(software).lower() + " " + str(exif.get("ImageDescription", "")).lower()
    for hint in _GENERATIVE_SOFTWARE_HINTS:
        if hint in blob:
            f.generative_software_hint = hint
            f.reasons.append(f"metadata names a generative tool ({hint!r}) — suggestive, "
                             f"but such tags are trivially forged or stripped")
            break

    # ── camera evidence ─────────────────────────────────────────────────
    f.camera_evidence, f.camera_evidence_strength = _camera_strength(exif)
    if f.camera_evidence_strength in ("moderate", "strong"):
        f.reasons.append(
            f"{f.camera_evidence_strength} camera capture evidence "
            f"({len(f.camera_evidence)} EXIF capture fields present)"
        )
    elif not f.has_exif:
        f.reasons.append("no EXIF present — NOT evidence of AI generation; every "
                         "screenshot, social-media upload and re-save strips it")

    # ── container forensics ─────────────────────────────────────────────
    f.jpeg_quantization_fingerprint, f.quantization_notes = _quantization_fingerprint(img)
    f.reasons.extend(f.quantization_notes)
    f.thumbnail_mismatch = _thumbnail_mismatch(img, data)
    if f.thumbnail_mismatch:
        f.reasons.append("EXIF thumbnail disagrees with full image — classical "
                         "manipulation indicator")

    # ── Tier A resolution ───────────────────────────────────────────────
    # ONLY cryptographically-verified evidence resolves Tier A. Everything
    # else, however suggestive, is passed to Tier B as evidence. Today
    # nothing sets c2pa_verified=True (no verification shipped), so Tier A
    # never resolves — correct and deliberate: an unverified manifest must
    # not be sold to the user as certainty.
    if f.c2pa_verified:
        f.tier_a_resolved = True
        f.confidence = 0.999
        f.reasons.append("verified C2PA signature — cryptographic provenance")

    return f
