"""
Stage 3 — conformal calibration + OOD threshold fitting for the Generation
Fingerprint module. Proposal §4.6, §6 (C1), §7 stage 3.

Same split discipline as the existing fit_calibration.py: fit on one held-out
half of val, report/validate on the other — fitting and reporting on the
same images produces a flattering number that says nothing about new inputs.
This is Mondrian (class-conditional) split conformal prediction
(Angelopoulos & Bates 2021), NOT the same mechanism as fit_calibration.py's
temperature scaling — conformal prediction sets replace MIN_CONFIDENCE/
MAX_DISAGREEMENT with a stated coverage guarantee (proposal §2 item 5, §9),
temperature scaling only reshapes the softmax. Both matter; they answer
different questions.

Usage:
    python training/calibrate_fingerprint.py \
        --checkpoint ../checkpoints/fingerprint/fingerprint_extractor.pth --data ../data
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.fingerprint import FingerprintExtractor, OODScorer
from models.fingerprint.model import TASK_CLASSES
from training.fingerprint_dataset import FingerprintDataset, collate_fingerprint

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints" / "fingerprint"


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    zs, task_logits, labels = [], [], []
    for reps, task_labels, _domain_ids, _domain_keys in loader:
        reps = {k: v.to(device) for k, v in reps.items()}
        out = model(reps)
        zs.append(out.z.cpu())
        task_logits.append(out.task_logits.cpu())
        labels.append(task_labels)
    return torch.cat(zs), torch.cat(task_logits), torch.cat(labels)


def fit_conformal_quantiles(task_logits: torch.Tensor, labels: torch.Tensor,
                             n_classes: int, alpha: float = 0.05) -> dict:
    """
    Mondrian split conformal: one quantile per TRUE class, fit on the
    nonconformity score 1 - softmax_prob(true_class). Finite-sample corrected
    (ceil((n_c+1)(1-alpha))/n_c-th order statistic), per Angelopoulos & Bates.

    At inference: class c is in the prediction set iff softmax_prob(c) >=
    1 - q_hat[c]. Singleton set -> answer; multi-element -> UNCERTAIN
    (conflicting evidence); empty set -> UNCERTAIN (unknown distribution).
    """
    probs = torch.softmax(task_logits, dim=1).numpy()
    labels_np = labels.numpy()
    q_hat = {}
    for c in range(n_classes):
        mask = labels_np == c
        n_c = int(mask.sum())
        if n_c == 0:
            q_hat[c] = 1.0  # no calibration data for this class — maximally conservative
            continue
        nonconformity = 1.0 - probs[mask, c]
        level = min(np.ceil((n_c + 1) * (1 - alpha)) / n_c, 1.0)
        q_hat[c] = float(np.quantile(nonconformity, level, method="higher"))
    return q_hat


def evaluate_conformal(task_logits: torch.Tensor, labels: torch.Tensor,
                        q_hat: dict, n_classes: int) -> dict:
    """Reports EMPIRICAL coverage and mean set size on held-out data — the
    actual validation that the guarantee holds, not just that fitting ran."""
    probs = torch.softmax(task_logits, dim=1).numpy()
    labels_np = labels.numpy()
    set_sizes, covered, empty = [], 0, 0
    for i in range(len(labels_np)):
        pred_set = [c for c in range(n_classes) if probs[i, c] >= 1 - q_hat[c]]
        set_sizes.append(len(pred_set))
        if len(pred_set) == 0:
            empty += 1
        if labels_np[i] in pred_set:
            covered += 1
    return {
        "empirical_coverage": covered / len(labels_np),
        "mean_set_size": float(np.mean(set_sizes)),
        "empty_set_rate": empty / len(labels_np),
        "singleton_rate": float(np.mean([s == 1 for s in set_sizes])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="../data")
    parser.add_argument("--alpha", type=float, default=0.05, help="Target miscoverage rate (1-alpha = coverage).")
    parser.add_argument("--ood-quantile", type=float, default=0.95,
                         help="Percentile of IN-DISTRIBUTION calibration scores used as the "
                              "OOD risk threshold — NOT hand-picked, unlike ensemble.py's current "
                              "MIN_CONFIDENCE/MAX_DISAGREEMENT constants (proposal §2 item 5).")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device)
    model = FingerprintExtractor(n_generators=state["n_generators"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    val_ds = FingerprintDataset(args.data, split="val", exclude_domains=())
    n = len(val_ds)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    calib_idx, test_idx = perm[: n // 2], perm[n // 2:]
    print(f"Val set {n} images -> calibration {len(calib_idx)} / held-out test {len(test_idx)}")

    calib_loader = DataLoader(Subset(val_ds, calib_idx.tolist()), batch_size=args.batch_size,
                               shuffle=False, num_workers=args.workers, collate_fn=collate_fingerprint)
    test_loader = DataLoader(Subset(val_ds, test_idx.tolist()), batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, collate_fn=collate_fingerprint)

    print("Collecting calibration-split embeddings/logits...")
    calib_z, calib_logits, calib_labels = collect(model, calib_loader, device)
    print("Collecting held-out test-split embeddings/logits...")
    test_z, test_logits, test_labels = collect(model, test_loader, device)

    n_classes = len(TASK_CLASSES)

    # ─── conformal quantiles ─────────────────────────────────────────────────
    q_hat = fit_conformal_quantiles(calib_logits, calib_labels, n_classes, alpha=args.alpha)
    report = evaluate_conformal(test_logits, test_labels, q_hat, n_classes)

    print(f"\nConformal (alpha={args.alpha}, target coverage={1 - args.alpha:.0%}):")
    print(f"  q_hat per class: " + ", ".join(f"{TASK_CLASSES[c]}={q:.4f}" for c, q in q_hat.items()))
    print(f"  empirical coverage on held-out test: {report['empirical_coverage']:.4f} "
          f"(target {1 - args.alpha:.4f} — should be >= target if fitting worked correctly)")
    print(f"  mean prediction-set size: {report['mean_set_size']:.3f}  "
          f"(1.0 = always a confident single answer)")
    print(f"  singleton rate: {report['singleton_rate']:.4f}  empty-set rate: {report['empty_set_rate']:.4f}")

    # ─── OOD thresholds (fit on calibration split, same split as conformal) ──
    scorer = OODScorer()
    scorer.fit(calib_z, calib_labels, n_classes=n_classes)
    calib_maha = scorer.mahalanobis(calib_z)
    calib_energy = scorer.energy(calib_logits)
    maha_threshold = float(torch.quantile(calib_maha, args.ood_quantile))
    energy_threshold = float(torch.quantile(calib_energy, args.ood_quantile))
    print(f"\nOOD thresholds (from {args.ood_quantile:.0%} of in-distribution calibration scores):")
    print(f"  mahalanobis_threshold={maha_threshold:.4f}  energy_threshold={energy_threshold:.4f}")

    out = {
        "checkpoint": args.checkpoint,
        "alpha": args.alpha,
        "conformal_q_hat": {TASK_CLASSES[c]: q for c, q in q_hat.items()},
        "conformal_validation": report,
        "ood_mahalanobis_threshold": maha_threshold,
        "ood_energy_threshold": energy_threshold,
        "ood_quantile": args.ood_quantile,
        "class_means": scorer.class_means.tolist(),
        "shared_precision": scorer.shared_precision.tolist(),
        "fitted_on": "held-out 50% split of val, same split for conformal + OOD",
    }
    out_path = CHECKPOINT_DIR / "fingerprint_calibration.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
