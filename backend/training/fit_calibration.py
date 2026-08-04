"""
Fit the confidence temperature for the deployed ensemble.

Two choices here matter for the result being meaningful:

  1. The temperature is fitted on a HELD-OUT half of the validation set and
     evaluated on the other half. Fitting and reporting on the same images
     would produce a flattering ECE that says nothing about new inputs.

  2. The objective is the NLL of the *deployed* fusion function — weighted
     average of each member's temperature-scaled softmax — not of a single
     model's logits. Calibrating a component in isolation does not calibrate
     the pipeline that is actually shipped.

Usage:
    python training/fit_calibration.py --data ../data
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.cnn_model import build_model, CLASSES, NUM_CLASSES
from models.calibration import save_temperature
from models.ensemble import MEMBER_SPECS, CHECKPOINT_DIR
from training.dataset import FakeImageDataset


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    accuracies = (probs.argmax(axis=1) == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if not mask.any():
            continue
        ece += (mask.sum() / len(labels)) * abs(accuracies[mask].mean() - confidences[mask].mean())
    return float(ece)


@torch.no_grad()
def collect_logits(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_all, labels_all = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(imgs)
        logits_all.append(logits.float().cpu())
        labels_all.append(labels)
    return torch.cat(logits_all), torch.cat(labels_all)


def fused_probs(member_logits: list[torch.Tensor], weights: list[float], temperature: torch.Tensor):
    """Exactly the deployed fusion: weighted mean of temperature-scaled softmaxes."""
    stacked = torch.stack([
        F.softmax(logits / temperature, dim=1) * w
        for logits, w in zip(member_logits, weights)
    ])
    fused = stacked.sum(dim=0)
    return fused / fused.sum(dim=1, keepdim=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    available = [s for s in MEMBER_SPECS if (CHECKPOINT_DIR / s["checkpoint"]).exists()]
    if not available:
        print("No checkpoints found — nothing to calibrate.")
        return
    total_weight = sum(s["weight"] for s in available)
    weights = [s["weight"] / total_weight for s in available]
    print(f"Calibrating ensemble: {[s['arch'] for s in available]}  weights={[round(w,3) for w in weights]}")

    # Deterministic 50/50 split of the val set: fit on one half, report on the other.
    probe_ds = FakeImageDataset(args.data, split="val", img_size=224)
    n = len(probe_ds)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    calib_idx, test_idx = perm[: n // 2], perm[n // 2:]
    print(f"Val set {n} images -> calibration {len(calib_idx)} / held-out test {len(test_idx)}\n")

    calib_member_logits, test_member_logits = [], []
    calib_labels = test_labels = None

    for spec in available:
        state = torch.load(CHECKPOINT_DIR / spec["checkpoint"], map_location="cpu")
        model, default_size = build_model(spec["arch"], pretrained=False)
        model.load_state_dict(state["model"] if "model" in state else state)
        model.to(device)
        img_size = state.get("img_size", default_size)

        ds = FakeImageDataset(args.data, split="val", img_size=img_size)
        for name, idx, bucket in (
            ("calib", calib_idx, calib_member_logits),
            ("test", test_idx, test_member_logits),
        ):
            loader = DataLoader(
                Subset(ds, idx.tolist()), batch_size=args.batch_size,
                shuffle=False, num_workers=args.workers,
                pin_memory=device.type == "cuda",
            )
            logits, labels = collect_logits(model, loader, device)
            bucket.append(logits)
            if name == "calib":
                calib_labels = labels
            else:
                test_labels = labels
        print(f"  collected logits: {spec['arch']} ({img_size}px)")
        del model
        torch.cuda.empty_cache()

    # ─── fit T against the deployed fusion ──────────────────────────────────
    log_t = torch.zeros(1, requires_grad=True)  # T = exp(0) = 1 at init
    optimizer = torch.optim.LBFGS([log_t], lr=0.05, max_iter=300)

    def _closure():
        optimizer.zero_grad()
        probs = fused_probs(calib_member_logits, weights, log_t.exp())
        loss = F.nll_loss(torch.log(probs + 1e-12), calib_labels)
        loss.backward()
        return loss

    optimizer.step(_closure)
    temperature = float(log_t.exp())

    # ─── report on the held-out half ────────────────────────────────────────
    before = fused_probs(test_member_logits, weights, torch.tensor(1.0)).numpy()
    after = fused_probs(test_member_logits, weights, torch.tensor(temperature)).numpy()
    labels_np = test_labels.numpy()

    ece_before = expected_calibration_error(before, labels_np)
    ece_after = expected_calibration_error(after, labels_np)
    acc_before = float((before.argmax(axis=1) == labels_np).mean())
    acc_after = float((after.argmax(axis=1) == labels_np).mean())

    print("\n" + "=" * 68)
    print("CALIBRATION RESULT (held-out half of val)")
    print("=" * 68)
    print(f"Fitted temperature : {temperature:.4f}"
          f"   ({'sharpening — model was under-confident' if temperature < 1 else 'softening — model was over-confident'})")
    print(f"ECE  before -> after : {ece_before:.4f} -> {ece_after:.4f}   ({ece_before - ece_after:+.4f})")
    print(f"Acc  before -> after : {acc_before:.4%} -> {acc_after:.4%}   (temperature cannot change argmax)")
    print(f"Mean confidence      : {before.max(axis=1).mean():.2%} -> {after.max(axis=1).mean():.2%}")
    print("=" * 68)

    if ece_after < ece_before:
        save_temperature(temperature, {
            "ece_before": ece_before,
            "ece_after": ece_after,
            "members": [s["arch"] for s in available],
            "fitted_on": "held-out 50% split of val",
        })
        print(f"\nSaved temperature to checkpoints/calibration.json")
    else:
        print("\nCalibration did not improve ECE on held-out data — not saving "
              "(pipeline keeps T=1.0).")


if __name__ == "__main__":
    main()
