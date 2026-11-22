"""
Generation Fingerprint extractor — proposal §4.2-4.4.

Core novelty under test (proposal §2): a domain-adversarial regularizer
trained against generator identity, forcing the fused embedding z to be
generator-invariant while still separating real from synthetic. Implemented
here as an architecture; whether it actually beats the baseline on the LOGO
protocol (proposal §8 #3) is an empirical question this module does not
answer by existing — see training/train_fingerprint.py for the run that
measures it.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from .encoder import SmallEncoder
from .representations import REPRESENTATION_NAMES, REPRESENTATION_CHANNELS

TASK_CLASSES = ["REAL", "AI_GENERATED", "AI_EDITED"]


# ─── Gradient Reversal Layer (Ganin & Lempitsky, 2015) ──────────────────────

class _GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Identity on the forward pass; negates and scales the gradient on the
    backward pass. This is the whole domain-adversarial mechanism — the
    encoder ends up trained to MAXIMIZE the domain head's loss, i.e. to make
    z as uninformative about generator identity as possible, while the task
    head (no GRL in its path) still pulls z toward separating real/fake.
    `lambda_` should be ramped 0 -> target during warmup (see training loop
    docstring) — an untrained encoder produces adversarial gradients that are
    close to pure noise early on; DANN's own paper recommends this."""

    def __init__(self, lambda_: float = 0.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradReverseFn.apply(x, self.lambda_)


# ─── Attention-MIL pooling (Ilse et al., 2018) — proposal §4.3 ──────────────

class PatchMIL(nn.Module):
    """
    Gated attention pooling over patch embeddings. Supervised ONLY by the
    image-level label — no pixel masks required to get a per-patch score out,
    which is what makes the resulting Generation Trace Map (attention_weight
    x patch_score, reshaped to a grid) usable even on datasets that don't
    carry per-pixel ground truth (most of the AI_GENERATED sources in
    proposal §5 don't; only the AI_EDITED sources like DEFACTO do).
    """

    def __init__(self, embed_dim: int, hidden: int = 64):
        super().__init__()
        self.attn_v = nn.Linear(embed_dim, hidden)
        self.attn_u = nn.Linear(embed_dim, hidden)
        self.attn_w = nn.Linear(hidden, 1)
        self.patch_score = nn.Linear(embed_dim, 1)

    def forward(self, patch_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        patch_embeds: [B, N, D] (N patches per image)
        Returns:
          bag_score:    [B]      — image-level fake probability (pre-sigmoid logit)
          patch_scores: [B, N]   — per-patch fake probability (pre-sigmoid logit),
                                   the raw material for the Generation Trace Map
        """
        # Gated attention (Ilse et al. eq. 9): sigmoid gate on a tanh branch,
        # empirically more stable than plain tanh attention.
        a = torch.tanh(self.attn_v(patch_embeds)) * torch.sigmoid(self.attn_u(patch_embeds))
        attn_logits = self.attn_w(a).squeeze(-1)          # [B, N]
        attn_weights = torch.softmax(attn_logits, dim=1)   # [B, N]

        patch_scores = self.patch_score(patch_embeds).squeeze(-1)  # [B, N]
        bag_score = (attn_weights * patch_scores).sum(dim=1)       # [B]
        return bag_score, patch_scores


@dataclass
class FingerprintOutput:
    z: torch.Tensor                    # [B, fusion_dim] fused embedding
    task_logits: torch.Tensor          # [B, 3] REAL/AI_GENERATED/AI_EDITED
    domain_logits: torch.Tensor | None  # [B, n_generators] — only if training with GRL
    attribution_logits: torch.Tensor   # [B, n_generators + 1] — last = UNKNOWN
    bag_score: torch.Tensor | None = None      # [B] — only if patches passed in
    patch_scores: torch.Tensor | None = None   # [B, N] — only if patches passed in


class FingerprintExtractor(nn.Module):
    """
    Ties together §4.1 (multi-representation encoders + fusion),
    §4.2 (domain-adversarial invariance), §4.3 (patch MIL), §4.4
    (generator attribution with explicit UNKNOWN). §4.5's OOD scorer is
    deliberately separate (ood.py) — it's non-parametric, fit on top of a
    trained z, not part of the forward pass.
    """

    def __init__(self, n_generators: int, embed_dim: int = 128, fusion_dim: int = 256,
                 grl_lambda: float = 0.0):
        super().__init__()
        self.representation_names = REPRESENTATION_NAMES
        self.encoders = nn.ModuleDict({
            name: SmallEncoder(REPRESENTATION_CHANNELS[name], embed_dim)
            for name in REPRESENTATION_NAMES
        })

        concat_dim = embed_dim * len(REPRESENTATION_NAMES)
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        self.task_head = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, len(TASK_CLASSES)),
        )

        self.grl = GradientReversalLayer(grl_lambda)
        self.domain_head = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.GELU(),
            nn.Linear(128, n_generators),
        )

        # +1 for the explicit UNKNOWN logit — proposal §4.4. Trained via
        # outlier exposure: whichever generator a given LOGO fold holds out
        # is used as an UNKNOWN-labeled negative for THIS head only, never
        # for task_head (which must still correctly say AI_GENERATED on it).
        self.attribution_head = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.GELU(),
            nn.Linear(128, n_generators + 1),
        )

        self.patch_mil = PatchMIL(fusion_dim)
        self.n_generators = n_generators
        self.embed_dim = embed_dim
        self.fusion_dim = fusion_dim

    def set_grl_lambda(self, lambda_: float):
        """Call once per step/epoch during the warmup ramp described in the
        GRL docstring — e.g. lambda_ = 2/(1+exp(-10*progress)) - 1, DANN's
        own schedule, progress in [0, 1] over training."""
        self.grl.lambda_ = lambda_

    def encode(self, reps: dict[str, torch.Tensor]) -> torch.Tensor:
        """reps: {name: [B, C, H, W]} for every name in self.representation_names.
        Returns fused z: [B, fusion_dim]."""
        embeds = [self.encoders[name](reps[name]) for name in self.representation_names]
        return self.fusion(torch.cat(embeds, dim=1))

    def forward(self, reps: dict[str, torch.Tensor],
                patch_reps: dict[str, torch.Tensor] | None = None,
                patch_shape: tuple[int, int] | None = None) -> FingerprintOutput:
        """
        reps: whole-image representations, {name: [B, C, H, W]}.
        patch_reps (optional): representations for a batch of patches
            flattened into the batch dim, {name: [B*N, C, H, W]} — caller
            reshapes back to [B, N, ...] after encoding. patch_shape=(B, N)
            required if patch_reps is given.
        """
        z = self.encode(reps)

        task_logits = self.task_head(z)
        domain_logits = self.domain_head(self.grl(z))
        attribution_logits = self.attribution_head(z)

        bag_score, patch_scores = None, None
        if patch_reps is not None:
            assert patch_shape is not None, "patch_shape=(B, N) required alongside patch_reps"
            B, N = patch_shape
            patch_z = self.encode(patch_reps)              # [B*N, fusion_dim]
            patch_z = patch_z.view(B, N, self.fusion_dim)   # [B, N, fusion_dim]
            bag_score, patch_scores = self.patch_mil(patch_z)

        return FingerprintOutput(
            z=z, task_logits=task_logits, domain_logits=domain_logits,
            attribution_logits=attribution_logits,
            bag_score=bag_score, patch_scores=patch_scores,
        )
