"""
Smoke test for models/fingerprint/ — shape/correctness sanity check, NOT a
training run. Run this after any change to the module before trusting it in
a real training loop.

Usage: python training/test_fingerprint_module.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.fingerprint import (
    extract_representations, REPRESENTATION_CHANNELS,
    FingerprintExtractor, OODScorer,
)
from models.fingerprint.model import TASK_CLASSES


def make_fake_image(h=256, w=256):
    return (np.random.rand(h, w, 3) * 255).astype(np.uint8)


def reps_to_batch(reps_list: list[dict], size=128) -> dict:
    """Stack a list of per-image representation dicts into batched tensors,
    resizing every representation to a common size (encoders are conv+GAP so
    they tolerate any size, but a real batch needs matching shapes)."""
    import cv2
    names = reps_list[0].keys()
    batch = {}
    for name in names:
        arrs = []
        for reps in reps_list:
            arr = reps[name]  # [C, H, W]
            c = arr.shape[0]
            resized = np.stack([cv2.resize(arr[ch], (size, size)) for ch in range(c)])
            arrs.append(resized)
        batch[name] = torch.from_numpy(np.stack(arrs)).float()
    return batch


def main():
    print("1. extract_representations on a random image...")
    img = make_fake_image()
    reps = extract_representations(img)
    for name, arr in reps.items():
        expected_c = REPRESENTATION_CHANNELS[name]
        assert arr.shape[0] == expected_c, f"{name}: expected {expected_c} channels, got {arr.shape}"
        assert arr.dtype == np.float32
        assert 0.0 <= arr.min() and arr.max() <= 1.0 + 1e-4, f"{name} out of [0,1]: [{arr.min()}, {arr.max()}]"
    print(f"   OK — {list(reps.keys())}, shapes: {[reps[n].shape for n in reps]}")

    print("2. batch 4 images into tensors...")
    batch = reps_to_batch([extract_representations(make_fake_image()) for _ in range(4)])
    for name, t in batch.items():
        print(f"   {name}: {tuple(t.shape)}")

    print("3. FingerprintExtractor forward pass (whole-image only)...")
    n_generators = 5  # e.g. stylegan, sd14, sdxl, dalle3, mj6
    model = FingerprintExtractor(n_generators=n_generators)
    model.set_grl_lambda(0.5)
    out = model(batch)
    assert out.z.shape == (4, model.fusion_dim)
    assert out.task_logits.shape == (4, len(TASK_CLASSES))
    assert out.domain_logits.shape == (4, n_generators)
    assert out.attribution_logits.shape == (4, n_generators + 1)
    assert out.bag_score is None and out.patch_scores is None
    print(f"   OK — z{tuple(out.z.shape)}, task{tuple(out.task_logits.shape)}, "
          f"domain{tuple(out.domain_logits.shape)}, attribution{tuple(out.attribution_logits.shape)}")

    print("4. FingerprintExtractor forward pass WITH patches (MIL path)...")
    B, N = 4, 6  # 4 images, 6 patches each
    patch_batch = reps_to_batch(
        [extract_representations(make_fake_image(64, 64)) for _ in range(B * N)], size=64
    )
    out_p = model(batch, patch_reps=patch_batch, patch_shape=(B, N))
    assert out_p.bag_score.shape == (B,)
    assert out_p.patch_scores.shape == (B, N)
    print(f"   OK — bag_score{tuple(out_p.bag_score.shape)}, patch_scores{tuple(out_p.patch_scores.shape)}")

    print("5. backward pass (gradients flow, including through GRL)...")
    loss = out_p.task_logits.sum() + out_p.domain_logits.sum() + out_p.bag_score.sum()
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0 and all(g == g for g in grad_norms), "NaN or missing gradients"
    print(f"   OK — {len(grad_norms)} parameter tensors received gradients")

    print("6. OODScorer fit + score...")
    torch.manual_seed(0)
    z_fake = torch.randn(60, model.fusion_dim)
    labels_fake = torch.randint(0, len(TASK_CLASSES), (60,))
    scorer = OODScorer()
    scorer.fit(z_fake, labels_fake, n_classes=len(TASK_CLASSES))
    maha = scorer.mahalanobis(z_fake[:4])
    energy = scorer.energy(out.task_logits)
    assert maha.shape == (4,)
    assert energy.shape == (4,)
    tiers = scorer.risk_tier(z_fake[:4], out.task_logits, mahalanobis_threshold=5.0, energy_threshold=-2.0)
    assert len(tiers) == 4 and all(t in ("LOW", "MEDIUM", "HIGH") for t in tiers)
    print(f"   OK — mahalanobis={maha.tolist()}, energy={energy.tolist()}, tiers={tiers}")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
