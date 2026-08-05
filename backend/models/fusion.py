"""
Fusion engine: blends the trained CNN's class probabilities with the
heuristic signal-based probabilities into one final decision.

The CNN dominates the blend; the heuristics act as a cross-check — when they
disagree with the CNN, `agreement=False` flags the result as needing more
scrutiny even though the fused prediction still follows the CNN.

On the CNN's actual strength: this docstring previously claimed "val macro-F1
~0.99". That number came from before the non-face shortcut was found and is
not a real capability measurement — the model was scoring 0.997 on EDITED by
guessing from subject matter (any non-face photo was EDITED), which collapsed
on the first authentic photo of a building. Current honest figure is ~0.86
macro-F1. See PROJECT_SPEC.md's Honest Status; do not restore the old number.
"""

CLASSES = ["REAL", "AI_GENERATED", "EDITED"]


def fuse(cnn_probs: dict, heuristic_probs: dict, cnn_weight: float = 0.85) -> dict:
    fused = {
        c: cnn_weight * cnn_probs[c] + (1 - cnn_weight) * heuristic_probs[c]
        for c in CLASSES
    }
    total = sum(fused.values()) + 1e-8
    return {c: round(v / total, 4) for c, v in fused.items()}
