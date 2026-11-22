"""
Selective-prediction evaluation — PLAN_HIGH_CONFIDENCE_DETECTION.md §5.

The metric change this whole plan turns on:

    stop reporting   accuracy at 100% coverage
    start reporting  COVERAGE AT >=99% PRECISION

i.e. not "how often are we right?" but "how much can we answer while
staying right?". A detector that answers 60% of images at 99.3% precision is
far more useful for forensics than one that answers everything at 87% — the
first one's answers can be acted on, the second one's cannot.

This script sweeps the confidence threshold over a checkpoint's predictions
and reports the full accuracy-vs-coverage curve, plus the specific operating
points that matter. It trains nothing and changes nothing; it measures.

Reports separately:
  * overall selective accuracy at each coverage level
  * FALSE-REAL rate — calling a synthetic/edited image REAL. The dangerous
    direction: a false alarm gets reviewed and dismissed, a false REAL gets
    believed. Tracked on its own because a good aggregate number can hide a
    bad one here.
  * per-domain coverage, so a generator that is only answerable at low
    precision is visible instead of averaged away.

Usage:
    python training/evaluate_selective.py \
        --checkpoint ../checkpoints/efficientnet_b4_fake.pth --data ../data
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.cnn_model import build_model, CLASSES
from training.dataset import FakeImageDataset
from training.evaluate_by_category import category_of

REAL_IDX = CLASSES.index("REAL")


@torch.no_grad()
def collect_predictions(checkpoint_path: str, data_dir: str, batch_size: int,
                         workers: int, device) -> tuple[np.ndarray, np.ndarray, list]:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = state.get("arch", "efficientnet_b4")
    img_size = state.get("img_size", 380)

    model, _ = build_model(arch, pretrained=False)
    model.load_state_dict(state["model"])
    model = model.to(device).eval()

    ds = FakeImageDataset(data_dir, split="val", img_size=img_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    total = len(loader)

    probs_all, labels_all = [], []
    for step, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device)
        with torch.autocast(device_type=device.type):
            logits = model(imgs)
        probs_all.append(F.softmax(logits.float(), dim=1).cpu().numpy())
        labels_all.append(labels.numpy())
        if (step + 1) % 20 == 0 or (step + 1) == total:
            print(f"  inference {step + 1}/{total} ({100 * (step + 1) / total:.1f}%)", flush=True)

    categories = [category_of(p) for p, _ in ds.paths]
    return np.concatenate(probs_all), np.concatenate(labels_all), categories


def selective_curve(probs: np.ndarray, labels: np.ndarray, n_points: int = 50) -> list:
    """Sweep the confidence threshold; at each point report what fraction of
    images are answered (coverage) and how accurate those answers are."""
    confidence = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = preds == labels

    curve = []
    for thr in np.linspace(0.0, 0.999, n_points):
        answered = confidence >= thr
        n_answered = int(answered.sum())
        if n_answered == 0:
            continue
        # False-REAL among ANSWERED images: predicted REAL but actually not.
        false_real = int(((preds == REAL_IDX) & (labels != REAL_IDX) & answered).sum())
        curve.append({
            "threshold": float(thr),
            "coverage": n_answered / len(labels),
            "selective_accuracy": float(correct[answered].mean()),
            "selective_error": float(1 - correct[answered].mean()),
            "n_answered": n_answered,
            "false_real_count": false_real,
            "false_real_rate_of_answered": false_real / n_answered,
        })
    return curve


def coverage_at_precision(curve: list, target_precision: float) -> dict | None:
    """The headline metric: the MOST coverage achievable while keeping
    selective accuracy at or above the target. Returns None when the target
    is unreachable at any threshold — which is itself an important, honest
    result and must be reported as such rather than silently omitted."""
    viable = [p for p in curve if p["selective_accuracy"] >= target_precision]
    if not viable:
        return None
    return max(viable, key=lambda p: p["coverage"])


def per_domain_coverage(probs, labels, categories, threshold: float) -> dict:
    confidence = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    out = {}
    for cat, conf, pred, label in zip(categories, confidence, preds, labels):
        d = out.setdefault(cat, {"n": 0, "answered": 0, "correct_answered": 0})
        d["n"] += 1
        if conf >= threshold:
            d["answered"] += 1
            d["correct_answered"] += int(pred == label)
    for d in out.values():
        d["coverage"] = d["answered"] / d["n"] if d["n"] else 0.0
        d["selective_accuracy"] = (d["correct_answered"] / d["answered"]) if d["answered"] else None
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="../data")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"],
                         help="Force a device — 'cpu' to run alongside a GPU training job. "
                              "Do NOT use CUDA_VISIBLE_DEVICES='' instead; that breaks "
                              "torch.load on a GPU-saved checkpoint (see evaluate_by_category.py).")
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    probs, labels, categories = collect_predictions(
        args.checkpoint, args.data, args.batch_size, args.workers, device)
    curve = selective_curve(probs, labels)

    full = curve[0]  # threshold 0 == answer everything == the conventional metric
    print("\n" + "=" * 72)
    print("SELECTIVE PREDICTION — accuracy vs. coverage")
    print("=" * 72)
    print(f"Conventional metric (answer everything): accuracy={full['selective_accuracy']:.4f} "
          f"at coverage={full['coverage']:.1%}")
    print(f"  false-REAL rate at full coverage: {full['false_real_rate_of_answered']:.4f} "
          f"({full['false_real_count']} images called REAL that were not)")

    targets = {}
    print("\nCoverage achievable at each precision target:")
    for target in (0.95, 0.99, 0.999):
        point = coverage_at_precision(curve, target)
        targets[f"{target}"] = point
        if point is None:
            print(f"  >= {target:.1%} precision : UNREACHABLE at any threshold "
                  f"(max selective accuracy is {max(p['selective_accuracy'] for p in curve):.4f})")
        else:
            print(f"  >= {target:.1%} precision : coverage={point['coverage']:.1%} "
                  f"({point['n_answered']} of {len(labels)} images), "
                  f"actual accuracy={point['selective_accuracy']:.4f}, "
                  f"threshold={point['threshold']:.3f}, "
                  f"false-REAL={point['false_real_rate_of_answered']:.4f}")

    print("\nCurve (every 5th point):")
    print(f"  {'thresh':>7}  {'coverage':>9}  {'sel.acc':>8}  {'falseREAL':>10}")
    for p in curve[::5]:
        print(f"  {p['threshold']:>7.3f}  {p['coverage']:>8.1%}  "
              f"{p['selective_accuracy']:>8.4f}  {p['false_real_rate_of_answered']:>10.4f}")

    # Per-domain breakdown at the 99%-precision operating point when one
    # exists — an aggregate coverage number can hide a domain that is
    # entirely unanswerable, which is exactly what matters for unseen
    # generators.
    ref = targets.get("0.99") or full
    domains = per_domain_coverage(probs, labels, categories, ref["threshold"])
    print(f"\nPer-domain coverage at threshold={ref['threshold']:.3f}:")
    print(f"  {'domain':>26}  {'n':>6}  {'coverage':>9}  {'sel.acc':>8}")
    for cat, d in sorted(domains.items()):
        acc = f"{d['selective_accuracy']:.4f}" if d["selective_accuracy"] is not None else "  n/a"
        print(f"  {cat:>26}  {d['n']:>6}  {d['coverage']:>8.1%}  {acc:>8}")

    if args.out_json:
        Path(args.out_json).write_text(json.dumps({
            "checkpoint": args.checkpoint,
            "full_coverage_point": full,
            "coverage_at_precision": targets,
            "curve": curve,
            "per_domain_at_reference_threshold": domains,
            "reference_threshold": ref["threshold"],
        }, indent=2))
        print(f"\nSaved -> {args.out_json}")


if __name__ == "__main__":
    main()
