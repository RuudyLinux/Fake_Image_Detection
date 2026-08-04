"""
Confidence calibration via temperature scaling.

A raw softmax score is not a probability you can act on. Training with label
smoothing (as this model was) systematically distorts it, so "82% confident"
may correspond to a very different real-world hit rate.

Temperature scaling (Guo et al., 2017) fits ONE scalar T on held-out data and
divides the logits by it before softmax. It cannot change which class wins —
accuracy is untouched — it only reshapes the confidence distribution so the
number means what it says.

T > 1 softens overconfident outputs; T < 1 sharpens underconfident ones.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CALIBRATION_PATH = Path(__file__).parent.parent / "checkpoints" / "calibration.json"


class TemperatureScaler(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(float(temperature)).log())

    @property
    def temperature(self) -> float:
        return float(self.log_temperature.exp())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # optimise in log-space so T stays strictly positive
        return logits / self.log_temperature.exp()

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> float:
        """Fit T by minimising NLL on a held-out split. Returns the fitted T."""
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=0.05, max_iter=max_iter)

        def _closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(_closure)
        return self.temperature


def save_temperature(temperature: float, metadata: dict | None = None) -> None:
    payload = {"temperature": float(temperature)}
    if metadata:
        payload.update(metadata)
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(payload, indent=2))


def load_temperature(default: float = 1.0) -> float:
    """Returns the fitted temperature, or `default` (no-op) if none was fitted."""
    if not CALIBRATION_PATH.exists():
        return default
    try:
        return float(json.loads(CALIBRATION_PATH.read_text())["temperature"])
    except Exception:
        return default


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature-scaled softmax probabilities."""
    return F.softmax(logits / temperature, dim=-1)
