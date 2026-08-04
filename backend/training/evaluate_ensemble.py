"""
Evaluate the deployed ensemble pipeline end to end.

Runs the real EnsembleDetector.predict() path — same preprocessing, TTA,
calibration and fusion the API uses — rather than a reimplementation, so the
numbers describe what actually ships.

Reports three things a plain accuracy figure cannot:

  * SINGLE vs ENSEMBLE vs ENSEMBLE+TTA, so each addition has to justify itself.

  * RISK-COVERAGE. For a system allowed to answer "uncertain", accuracy alone
    is meaningless — it can be driven to 100% by refusing everything. The
    honest pair is (coverage, accuracy-on-answered): of the questions it chose
    to answer, how many did it get right?

  * ROBUSTNESS under the same perturbation sweep as the single-model baseline,
    so the comparison is like-for-like.

Usage:
    python training/evaluate_ensemble.py --limit-per-class 150
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.cnn_model import CLASSES
from models.ensemble import EnsembleDetector
from training.dataset import CLASS_NAMES, LABEL_MAP, EXTENSIONS
from training.robustness_eval import PERTURBATIONS


def load_samples(data_root: str, split: str, limit_per_class: int):
    base = Path(data_root) / split
    samples = []
    for class_name, label in LABEL_MAP.items():
        class_dir = base / class_name
        if not class_dir.exists():
            continue
        imgs = sorted(p for p in class_dir.rglob("*") if p.suffix.lower() in EXTENSIONS)
        samples.extend((p, label) for p in imgs[:limit_per_class])
    return samples


def run_pipeline(detector, samples, perturb, use_tta: bool):
    """Returns (predicted_indices, abstained_flags, confidences, true_labels)."""
    preds, abstains, confs, labels = [], [], [], []
    for path, label in samples:
        img = Image.open(path).convert("RGB")
        img = perturb(img)
        out = detector.predict(np.array(img), use_tta=use_tta)
        preds.append(CLASSES.index(out["prediction"]))
        abstains.append(out["abstain"])
        confs.append(out["confidence"])
        labels.append(label)
    return (
        np.array(preds), np.array(abstains, dtype=bool),
        np.array(confs), np.array(labels),
    )


def risk_coverage(preds, abstains, labels):
    answered = ~abstains
    coverage = float(answered.mean())
    if answered.sum() == 0:
        return coverage, float("nan"), 0.0
    acc_answered = float((preds[answered] == labels[answered]).mean())
    acc_forced = float((preds == labels).mean())  # if abstention were disabled
    return coverage, acc_answered, acc_forced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data")
    parser.add_argument("--limit-per-class", type=int, default=150)
    parser.add_argument("--skip-robustness", action="store_true")
    args = parser.parse_args()

    detector = EnsembleDetector()
    if not detector.is_available:
        print("No checkpoints available.")
        return
    print(f"Members: {detector.member_names}  temperature={detector.temperature:.4f}")
    print(f"Device: {detector.device}\n")

    samples = load_samples(args.data, "val", args.limit_per_class)
    print(f"Evaluating on {len(samples)} images "
          f"({args.limit_per_class}/class)\n")

    identity = PERTURBATIONS["none (pristine)"]

    # ─── ablation: does each component earn its cost? ───────────────────────
    print("=" * 78)
    print("COMPONENT ABLATION (pristine images)")
    print("=" * 78)
    header = f"{'Configuration':<32} {'Coverage':>10} {'Acc(answered)':>15} {'Acc(forced)':>13}"
    print(header)
    print("-" * len(header))

    ablations = {}
    for label, use_tta in (("ensemble, no TTA", False), ("ensemble + TTA", True)):
        preds, abstains, confs, labels = run_pipeline(detector, samples, identity, use_tta)
        cov, acc_ans, acc_forced = risk_coverage(preds, abstains, labels)
        ablations[label] = (cov, acc_ans, acc_forced)
        print(f"{label:<32} {cov:>9.1%} {acc_ans:>14.2%} {acc_forced:>12.2%}")

    print("\nAcc(answered) = accuracy on images it chose to answer.")
    print("Acc(forced)   = accuracy if abstention were disabled (comparable to the")
    print("                single-model baseline, which always answers).")

    # ─── robustness sweep ───────────────────────────────────────────────────
    if not args.skip_robustness:
        print("\n" + "=" * 78)
        print("ROBUSTNESS — full pipeline (ensemble + TTA + abstention)")
        print("=" * 78)
        header = f"{'Perturbation':<24} {'Coverage':>10} {'Acc(answered)':>15} {'Acc(forced)':>13}"
        print(header)
        print("-" * len(header))

        forced_accs = {}
        for name, perturb in PERTURBATIONS.items():
            preds, abstains, confs, labels = run_pipeline(detector, samples, perturb, True)
            cov, acc_ans, acc_forced = risk_coverage(preds, abstains, labels)
            forced_accs[name] = acc_forced
            print(f"{name:<24} {cov:>9.1%} {acc_ans:>14.2%} {acc_forced:>12.2%}")

        best = max(forced_accs.values())
        worst_name = min(forced_accs, key=forced_accs.get)
        print("\n" + "=" * 78)
        print(f"Best-case accuracy      : {best:.2%}")
        print(f"Worst-case perturbation : {worst_name} -> {forced_accs[worst_name]:.2%}")
        print(f"Robustness gap          : {best - forced_accs[worst_name]:+.2%}")
        print("=" * 78)


if __name__ == "__main__":
    main()
