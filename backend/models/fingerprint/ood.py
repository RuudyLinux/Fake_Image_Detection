"""
Open-set / OOD scorer — proposal §4.5.

Non-parametric: fit AFTER the extractor is trained, on its frozen embeddings.
Not part of FingerprintExtractor's forward pass because it has no gradient —
it's a statistic over a fitted training-set distribution, computed once and
reused at inference time.

Two complementary scores, both computed independent of the task softmax
(that's the point — an input can have a "confident" softmax and still be
flagged here, which is exactly the failure mode proposal §2.5's Mahalanobis/
energy citation targets):

  Mahalanobis — distance to the nearest class-conditional Gaussian in z-space.
  Energy      — -logsumexp(task_logits), a temperature-free measure of how
                "peaked" the model's own decision surface is at this point.

High values on both => OOD_risk = HIGH in the evidence dossier (proposal
§4.6), independent of whatever the softmax confidence claims.
"""

import torch


class OODScorer:
    def __init__(self):
        self.class_means: torch.Tensor | None = None   # [n_classes, D]
        self.shared_precision: torch.Tensor | None = None  # [D, D], inverse of pooled covariance
        self._fitted = False

    def fit(self, z: torch.Tensor, labels: torch.Tensor, n_classes: int, eps: float = 1e-3):
        """
        z:      [N, D] embeddings from the trained extractor, held-out or
                train split (consistent with fit_calibration.py's existing
                split discipline — see proposal §4.6).
        labels: [N] task labels (0=REAL, 1=AI_GENERATED, 2=AI_EDITED).
        """
        D = z.shape[1]
        means = torch.zeros(n_classes, D)
        pooled_cov = torch.zeros(D, D)
        total_n = 0
        for c in range(n_classes):
            mask = labels == c
            n_c = int(mask.sum())
            if n_c < 2:
                continue
            z_c = z[mask]
            means[c] = z_c.mean(dim=0)
            centered = z_c - means[c]
            pooled_cov += centered.T @ centered
            total_n += n_c

        pooled_cov = pooled_cov / max(total_n - n_classes, 1)
        pooled_cov += eps * torch.eye(D)  # regularize — keeps inversion stable on small D
        self.class_means = means
        self.shared_precision = torch.linalg.inv(pooled_cov)
        self._fitted = True

    def mahalanobis(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, D] -> [B] min Mahalanobis distance to any class mean."""
        assert self._fitted, "call fit() first"
        dists = []
        for c in range(self.class_means.shape[0]):
            diff = z - self.class_means[c]                      # [B, D]
            d2 = torch.einsum("bi,ij,bj->b", diff, self.shared_precision, diff)
            dists.append(d2)
        return torch.stack(dists, dim=1).min(dim=1).values.clamp_min(0).sqrt()

    @staticmethod
    def energy(task_logits: torch.Tensor) -> torch.Tensor:
        """task_logits: [B, n_classes] -> [B]. Higher = more OOD-like (lower
        confidence the model's decision surface has anywhere for this input)."""
        return -torch.logsumexp(task_logits, dim=1)

    def risk_tier(self, z: torch.Tensor, task_logits: torch.Tensor,
                   mahalanobis_threshold: float, energy_threshold: float) -> list[str]:
        """Combines both scores into the dossier's OOD_risk field (proposal
        §4.6, §9's example output). Thresholds must be fit on a held-out
        calibration split alongside the conformal quantiles (proposal §4.6)
        — no defaults are hard-coded here on purpose, unlike the current
        ensemble.py's MIN_CONFIDENCE/MAX_DISAGREEMENT constants this module
        exists partly to replace (proposal §2, item 5)."""
        maha = self.mahalanobis(z)
        eng = self.energy(task_logits)
        tiers = []
        for m, e in zip(maha.tolist(), eng.tolist()):
            high_maha = m > mahalanobis_threshold
            high_energy = e > energy_threshold
            if high_maha and high_energy:
                tiers.append("HIGH")
            elif high_maha or high_energy:
                tiers.append("MEDIUM")
            else:
                tiers.append("LOW")
        return tiers
