"""
CLIP-feature detector — a third ensemble member built on a FROZEN foundation
model instead of a fine-tuned supervised CNN.

Why this exists
---------------
The other two members (EfficientNet-B4, ConvNeXt-Tiny) are ImageNet-pretrained
CNNs fine-tuned end-to-end on this project's training set. That set contains
StyleGAN faces and Stable Diffusion 1.4/2.1 output. Fine-tuning teaches those
models *those specific generators'* artifacts — upsampling signatures, texture
statistics — which are architecture-specific and change with every generator
generation. The features do not transfer to generators that did not exist at
training time.

This was already measured in this project, twice:

  * training/generate_diffusion_data.py records StyleGAN-trained models scoring
    0/30 on Stable Diffusion faces, 25/30 CONFIDENTLY WRONG as REAL, with
    near-zero member disagreement — both members shared the blind spot, so the
    ensemble's MI-based abstention never fired.

  * A 2026 zero-shot benchmark of 16 SOTA detection methods across 291
    generators reports 18-30% accuracy on Flux Dev / Firefly v4 / Midjourney
    v7. Those methods had more generator diversity in training than this one.

The published counter-approach (UnivFD, Ojha et al. CVPR 2023, and the
CLIP/DINO line of work through 2026) is to NOT fine-tune: take a large
vision-language model trained on hundreds of millions of natural images, freeze
it, and fit only a small head on its features. The frozen features encode a
broad prior over what real photographs look like rather than one generator's
fingerprint, so the decision boundary transfers to unseen generators far
better.

Design notes
------------
  * The backbone is FROZEN. Only the linear probe trains. This is the whole
    point — fine-tuning the backbone would re-introduce the generator-specific
    overfitting this module exists to avoid.
  * Because it fails very differently from two supervised CNNs, adding it also
    makes the ensemble's mutual-information (epistemic uncertainty) term more
    informative, which improves abstention — see ensemble.py.
  * Feature extraction is deterministic for a given image, so features can be
    cached and the probe retrained in seconds.
"""

from pathlib import Path

import torch
import torch.nn as nn

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
CLIP_INPUT_SIZE = 224
FEATURE_DIM = 768  # openai/clip-vit-large-patch14 projection_dim, verified at runtime

CLASSES = ["REAL", "AI_GENERATED", "EDITED"]


class CLIPProbe(nn.Module):
    """
    Linear probe over frozen CLIP vision features.

    Deliberately small. UnivFD's central result is that a *linear* classifier on
    frozen CLIP features generalizes better across generators than a deep
    network trained end-to-end — capacity here is a liability, not an asset,
    because it lets the head memorize generator-specific structure that the
    frozen features were chosen to avoid.
    """

    def __init__(self, feature_dim: int = FEATURE_DIM, num_classes: int = len(CLASSES),
                 hidden: int | None = None, dropout: float = 0.1):
        super().__init__()
        if hidden:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes),
            )
        self.feature_dim = feature_dim
        self.num_classes = num_classes

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)


class CLIPFeatureExtractor:
    """
    Frozen CLIP vision tower + preprocessing. Loaded lazily so importing this
    module (e.g. from the API at startup) does not pull ~1.7GB of weights
    unless a CLIP checkpoint is actually being used.
    """

    def __init__(self, model_id: str = CLIP_MODEL_ID, device: str | None = None):
        self.model_id = model_id
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

        self._processor = CLIPImageProcessor.from_pretrained(self.model_id)
        model = CLIPVisionModelWithProjection.from_pretrained(self.model_id)
        model.eval().to(self.device)
        for p in model.parameters():
            p.requires_grad = False  # frozen — see module docstring
        self._model = model

    @property
    def output_dim(self) -> int:
        self._ensure_loaded()
        return int(self._model.config.projection_dim)

    @torch.no_grad()
    def encode_pil(self, images) -> torch.Tensor:
        """Encode a PIL image or list of them into L2-normalized feature vectors."""
        self._ensure_loaded()
        if not isinstance(images, (list, tuple)):
            images = [images]
        inputs = self._processor(images=list(images), return_tensors="pt").to(self.device)
        out = self._model(**inputs).image_embeds
        return torch.nn.functional.normalize(out, dim=-1)


class CLIPDetector(nn.Module):
    """
    Inference-time wrapper presenting the same interface as the CNN members
    (forward(image_tensor) -> logits), so ensemble.py can treat it uniformly.

    Note it consumes PIL images rather than a pre-normalized tensor, because
    CLIP has its own preprocessing (different normalization constants and
    resize behaviour from the ImageNet pipeline the CNN members use). The
    ensemble passes the raw image alongside the tensor for this reason.
    """

    def __init__(self, probe: CLIPProbe, extractor: CLIPFeatureExtractor):
        super().__init__()
        self.probe = probe
        self.extractor = extractor

    @torch.no_grad()
    def forward_pil(self, pil_image) -> torch.Tensor:
        feats = self.extractor.encode_pil(pil_image)
        return self.probe(feats.to(next(self.probe.parameters()).device))


def load_clip_detector(checkpoint_path: str | Path, device: str | None = None) -> CLIPDetector:
    """Rebuild a trained probe + frozen extractor from a checkpoint."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = torch.load(checkpoint_path, map_location=dev)

    probe = CLIPProbe(
        feature_dim=state.get("feature_dim", FEATURE_DIM),
        num_classes=len(state.get("classes", CLASSES)),
        hidden=state.get("hidden"),
    )
    probe.load_state_dict(state["probe"])
    probe.eval().to(dev)

    extractor = CLIPFeatureExtractor(state.get("clip_model_id", CLIP_MODEL_ID), device=str(dev))
    return CLIPDetector(probe, extractor)
