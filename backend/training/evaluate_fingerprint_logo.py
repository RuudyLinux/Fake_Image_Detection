"""
LOGO evaluation for the Generation Fingerprint module — proposal §8 #3, the
primary claim metric (§2, §9): does a fingerprint checkpoint trained with one
or more domains excluded actually generalize to them, zero-shot?

Mirrors evaluate_by_category.py's role for the baseline CNN ensemble — same
idea (score a checkpoint against the FULL, unfiltered val set, broken down
by domain), applied to the new module's own checkpoint format and domain
vocabulary instead of the baseline's filename-category convention.

Usage:
    python training/evaluate_fingerprint_logo.py \
        --checkpoint ../checkpoints/fingerprint/fingerprint_extractor.pth --data ../data

Reads which domains were excluded straight from the checkpoint (train_fingerprint.py
saves `excluded_domains`), so this script doesn't need to be told twice what
the LOGO fold actually held out.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.fingerprint import FingerprintExtractor
from training.fingerprint_dataset import FingerprintDataset, collate_fingerprint


@torch.no_grad()
def evaluate(checkpoint_path: str, data_dir: str, batch_size: int, workers: int, device):
    state = torch.load(checkpoint_path, map_location=device)
    n_generators = state["n_generators"]
    excluded_domains = state.get("excluded_domains", ())

    model = FingerprintExtractor(n_generators=n_generators).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    # Deliberately load the FULL val set — no exclude_domains here, unlike
    # training. The whole point is to see accuracy ON the domains that were
    # excluded from training, not to re-confirm they're still excluded.
    val_ds = FingerprintDataset(data_dir, split="val", exclude_domains=())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=workers, collate_fn=collate_fingerprint)

    per_domain = {}  # domain_key -> {"n": int, "correct": int}

    for reps, task_labels, _domain_ids, domain_keys in val_loader:
        reps = {k: v.to(device) for k, v in reps.items()}
        out = model(reps)
        preds = out.task_logits.argmax(1).cpu()
        # domain_keys are the TRUE string keys from the dataset, not derived
        # by reversing DOMAIN_IDS — several keys share one id by design
        # (proposal §5 cross-dataset generator merges), so that reversal is
        # lossy and silently mislabeled a held-out domain this session,
        # producing a fabricated 0.0 LOGO-accuracy result. Fixed at the
        # source in fingerprint_dataset.py rather than patched here.
        for pred, label, key in zip(preds, task_labels, domain_keys):
            d = per_domain.setdefault(key, {"n": 0, "correct": 0})
            d["n"] += 1
            d["correct"] += int(pred == label)

    for key, d in per_domain.items():
        d["accuracy"] = d["correct"] / d["n"] if d["n"] else 0.0
        d["was_excluded_from_training"] = key in excluded_domains

    return {
        "checkpoint": checkpoint_path,
        "excluded_domains": list(excluded_domains),
        "per_domain": per_domain,
        # The headline LOGO number — mean accuracy over ONLY the excluded
        # domains. This is what proposal §9 compares against the baseline's
        # equivalent (evaluate_by_category.py's ai_generated/sd14 row on the
        # pre-SD checkpoint, or its successor once that's re-measured).
        "logo_accuracy": (
            sum(d["correct"] for k, d in per_domain.items() if k in excluded_domains)
            / max(sum(d["n"] for k, d in per_domain.items() if k in excluded_domains), 1)
        ) if excluded_domains else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="../data")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-json", default=None,
                         help="Optional path to save the full per-domain breakdown as JSON — "
                              "same tracked-metrics convention as *_by_category.json (see "
                              "PROJECT_SPEC.md's Evaluation section on why these stay in git).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = evaluate(args.checkpoint, args.data, args.batch_size, args.workers, device)

    print(json.dumps(result, indent=2))
    if result["logo_accuracy"] is not None:
        print(f"\nLOGO accuracy (excluded domains only, zero-shot): {result['logo_accuracy']:.4f}")
    else:
        print("\nNo excluded_domains recorded in this checkpoint — nothing was held out, "
              "so there's no LOGO number to report (this run measured in-distribution accuracy only).")

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(result, indent=2))
        print(f"Saved -> {args.out_json}")


if __name__ == "__main__":
    main()
