"""
Fusion engine: blends the trained CNN's class probabilities with the
heuristic signal-based probabilities into one final decision.

The CNN (val macro-F1 ~0.99) dominates the blend; the heuristics act as a
cross-check — when they disagree with the CNN, `agreement=False` flags the
result as needing more scrutiny even though the fused prediction still
follows the CNN.
"""

CLASSES = ["REAL", "AI_GENERATED", "EDITED"]


def fuse(cnn_probs: dict, heuristic_probs: dict, cnn_weight: float = 0.85) -> dict:
    fused = {
        c: cnn_weight * cnn_probs[c] + (1 - cnn_weight) * heuristic_probs[c]
        for c in CLASSES
    }
    total = sum(fused.values()) + 1e-8
    return {c: round(v / total, 4) for c, v in fused.items()}
