"""
Ensemble inference with test-time augmentation, calibrated confidence, and
principled abstention.

Design rationale
----------------
A single softmax score conflates two very different kinds of doubt:

  * ALEATORIC uncertainty — the image genuinely sits near a class boundary.
    More models will not help; the evidence is ambiguous.

  * EPISTEMIC uncertainty — the image is unlike anything in training, so the
    model is extrapolating. This is the dangerous one: a lone network will
    often extrapolate *confidently* and be confidently wrong.

Averaging softmax scores hides that distinction. This module keeps it, by
decomposing total predictive uncertainty:

    total entropy  =  expected member entropy  +  mutual information
    (predictive)      (aleatoric)                 (epistemic / disagreement)

Mutual information is high exactly when members individually commit to
*different* answers — the signature of an out-of-distribution input. That is
the signal used to abstain, which is why ensemble members are deliberately
different architectures at different resolutions; identical members agree even
when they are all wrong, and would make this term uselessly small.

Verdicts below the confidence floor, or above the disagreement ceiling, are
returned as UNCERTAIN rather than forced into a class.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.cnn_model import build_model, CLASSES
from models.calibration import load_temperature
from models.tta import tta_logits

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"

# Members are listed with the weight applied to their logits. EfficientNet-B4
# carries more weight: it sees 380px vs 224px, so it retains finer forensic
# detail. Weights are renormalised over whichever members actually load.
MEMBER_SPECS = [
    {"arch": "efficientnet_b4", "checkpoint": "efficientnet_b4_fake.pth", "weight": 0.6},
    {"arch": "convnext_tiny",   "checkpoint": "convnext_tiny_fake.pth",   "weight": 0.4},
]

# Abstain thresholds. MI is in nats; ln(3) ~ 1.0986 is maximum possible entropy
# for 3 classes, so 0.35 means members disagree over roughly a third of the
# theoretical maximum — well beyond normal boundary noise.
MIN_CONFIDENCE = 0.55
MAX_DISAGREEMENT = 0.35

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_transform(img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class EnsembleDetector:
    def __init__(self, device: str | None = None):
        # Ensemble + TTA is 2 models x 4 views = 8 forward passes per image.
        # On CPU at 380px that is seconds; prefer the GPU when one exists.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.members = []
        self.temperature = load_temperature(default=1.0)
        self._load_members()

    def _load_members(self) -> None:
        for spec in MEMBER_SPECS:
            path = CHECKPOINT_DIR / spec["checkpoint"]
            if not path.exists():
                continue
            try:
                state = torch.load(path, map_location="cpu")
                model, img_size = build_model(spec["arch"], pretrained=False)
                model.load_state_dict(state["model"] if "model" in state else state)
                model.eval().to(self.device)
                self.members.append({
                    "name": spec["arch"],
                    "model": model,
                    "transform": _build_transform(state.get("img_size", img_size)),
                    "weight": spec["weight"],
                    "val_macro_f1": state.get("val_macro_f1"),
                })
                print(f"[FakeGuard] Ensemble member loaded: {spec['arch']} "
                      f"(val_macro_f1={state.get('val_macro_f1', '?')})")
            except Exception as exc:
                print(f"[FakeGuard] Failed to load {spec['arch']}: {exc}")

        total = sum(m["weight"] for m in self.members)
        for member in self.members:
            member["weight"] /= total if total else 1.0

    @property
    def is_available(self) -> bool:
        return len(self.members) > 0

    @property
    def member_names(self) -> list[str]:
        return [m["name"] for m in self.members]

    @torch.no_grad()
    def predict(self, img: np.ndarray, use_tta: bool = True) -> dict:
        """
        Returns fused probabilities plus an uncertainty decomposition.

        `img` is an HxWx3 uint8 RGB array.
        """
        pil_img = Image.fromarray(img)

        member_probs = []   # calibrated per-member probability vectors
        weights = []
        per_member = {}
        b4_tensor = None    # kept for Grad-CAM, which needs B4's own input

        for member in self.members:
            tensor = member["transform"](pil_img).unsqueeze(0).to(self.device)
            if member["name"] == "efficientnet_b4":
                b4_tensor = tensor

            logits = (
                tta_logits(member["model"], tensor)
                if use_tta
                else member["model"](tensor)
            )
            probs = F.softmax(logits.float() / self.temperature, dim=1).squeeze(0).cpu().numpy()

            member_probs.append(probs)
            weights.append(member["weight"])
            per_member[member["name"]] = {
                CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CLASSES))
            }

        member_probs = np.stack(member_probs)          # [M, C]
        weights = np.asarray(weights)[:, None]         # [M, 1]
        fused = (member_probs * weights).sum(axis=0)
        fused = fused / fused.sum()

        # Uncertainty decomposition (nats)
        eps = 1e-12
        predictive_entropy = float(-(fused * np.log(fused + eps)).sum())
        expected_entropy = float(
            -(member_probs * np.log(member_probs + eps)).sum(axis=1).mean()
        )
        # Epistemic term; clamped since it is only non-negative in exact arithmetic.
        mutual_information = max(predictive_entropy - expected_entropy, 0.0)

        pred_idx = int(fused.argmax())
        prediction = CLASSES[pred_idx]
        confidence = float(fused[pred_idx])

        members_agree = len({
            CLASSES[int(p.argmax())] for p in member_probs
        }) == 1

        abstain_reasons = []
        if confidence < MIN_CONFIDENCE:
            abstain_reasons.append("low_confidence")
        if mutual_information > MAX_DISAGREEMENT:
            abstain_reasons.append("member_disagreement")

        return {
            "prediction": prediction,
            "confidence": confidence,
            "class_probabilities": {
                CLASSES[i]: round(float(fused[i]), 4) for i in range(len(CLASSES))
            },
            "uncertainty": {
                "predictive_entropy": round(predictive_entropy, 4),
                "aleatoric": round(expected_entropy, 4),
                "epistemic": round(mutual_information, 4),
            },
            "members": per_member,
            "members_agree": members_agree,
            "abstain": len(abstain_reasons) > 0,
            "abstain_reasons": abstain_reasons,
            "tta_used": use_tta,
            "temperature": self.temperature,
            "_b4_tensor": b4_tensor,
        }
