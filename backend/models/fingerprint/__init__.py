"""
Generation Fingerprint module — implements
RESEARCH_PROPOSAL_GENERATION_FINGERPRINT.md §4-6.

Status: architecture only, NOT trained, NOT wired into ensemble.py/fusion.py.
This package is additive — importing it has zero effect on the live
detection pipeline until a checkpoint exists and something explicitly loads
it, exactly like clip_detector.py's own bring-up path. The existing
EfficientNet-B4 + ConvNeXt-Tiny + CLIP ensemble stays the live system.

See training/train_fingerprint.py (once written) for the training loop; see
training/test_fingerprint_module.py for the shape/sanity smoke test this was
validated against before being left in the tree.
"""

from .representations import extract_representations, REPRESENTATION_CHANNELS
from .encoder import SmallEncoder
from .model import (
    FingerprintExtractor,
    GradientReversalLayer,
    PatchMIL,
    FingerprintOutput,
)
from .ood import OODScorer

__all__ = [
    "extract_representations",
    "REPRESENTATION_CHANNELS",
    "SmallEncoder",
    "FingerprintExtractor",
    "GradientReversalLayer",
    "PatchMIL",
    "FingerprintOutput",
    "OODScorer",
]
