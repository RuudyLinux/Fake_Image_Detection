"""
Stage 1 training loop for the Generation Fingerprint module — proposal §7.

Trains FingerprintExtractor's encoders + fusion + task_head + domain_head
jointly, with the GRL lambda ramped over training (DANN's own schedule).
Stage 2 (patch MIL + attribution heads, frozen extractor) and Stage 3
(conformal calibration) are separate scripts, written once Stage 1 has a
checkpoint worth building on top of — training them against random-init
Stage-1 weights would produce numbers that don't mean anything.

Usage:
    python training/train_fingerprint.py --data ../data --epochs 20 \
        --exclude-domain openfake_flux11dev openfake_flux2dev   # LOGO fold

This is NOT wired into the live ensemble (ensemble.py / fusion.py are
untouched). It produces a standalone checkpoint for evaluate_fingerprint_logo.py
to score, exactly like efficientnet_b4_logo_nosd14.pth is being produced and
scored for the baseline right now.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.fingerprint import FingerprintExtractor
from models.fingerprint.model import TASK_CLASSES
from training.fingerprint_dataset import FingerprintDataset, collate_fingerprint, N_DOMAINS

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints" / "fingerprint"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def grl_lambda_schedule(epoch: int, warmup_epochs: int, max_lambda: float = 1.0) -> float:
    """DANN's own schedule (Ganin & Lempitsky 2015, eq. 15): ramps 0 -> max_lambda
    smoothly so early, near-random domain-head gradients don't destabilize the
    encoder before it has learned anything useful to be invariant ABOUT.

    Deliberately keyed on a FIXED warmup_epochs, not args.epochs / total budget.
    The first real run of this module (see git history) tied the ramp to
    progress=(epoch-1)/(args.epochs-1) — early-stopping cut that run to 7 of
    a nominal 20 epochs, which front-loaded the ramp to lambda=0.92 by epoch 7
    instead of the intended slow warmup, destabilized training before any
    domain-invariance benefit could show, and the run never recovered past
    its epoch-1 (pre-ramp) accuracy before patience triggered. Decoupling the
    ramp from the epoch budget — and raising --patience alongside it — is the
    actual fix, not a smaller max_lambda band-aid over the same root cause."""
    progress = min(epoch / warmup_epochs, 1.0)
    return max_lambda * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


def run_epoch(model, loader, optimizer, task_criterion, domain_criterion, device,
              train: bool, grl_lambda: float = None):
    model.train() if train else model.eval()
    total_task_loss, total_domain_loss = 0.0, 0.0
    correct_task, n = 0, 0

    for reps, task_labels, domain_ids, _domain_keys in loader:
        reps = {k: v.to(device) for k, v in reps.items()}
        task_labels, domain_ids = task_labels.to(device), domain_ids.to(device)

        if train:
            model.set_grl_lambda(grl_lambda)
            optimizer.zero_grad()
            out = model(reps)
            task_loss = task_criterion(out.task_logits, task_labels)
            domain_loss = domain_criterion(out.domain_logits, domain_ids)
            # GRL already negates the encoder's gradient contribution from
            # domain_loss — adding both losses normally here is correct, NOT
            # a double-negation, because the sign flip happens inside the
            # layer, not in this loss expression. See model.py's GRL docstring.
            loss = task_loss + domain_loss
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                out = model(reps)
                task_loss = task_criterion(out.task_logits, task_labels)
                domain_loss = domain_criterion(out.domain_logits, domain_ids)

        total_task_loss += task_loss.item() * len(task_labels)
        total_domain_loss += domain_loss.item() * len(task_labels)
        correct_task += (out.task_logits.argmax(1) == task_labels).sum().item()
        n += len(task_labels)

    return {
        "task_loss": total_task_loss / n,
        "domain_loss": total_domain_loss / n,
        "task_acc": correct_task / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="fingerprint_extractor.pth")
    parser.add_argument("--exclude-domain", nargs="*", default=(),
                         help="Domain KEYS (see fingerprint_dataset.DOMAIN_IDS) to exclude from "
                              "BOTH train and val — the LOGO protocol (proposal §7-8).")
    parser.add_argument("--patience", type=int, default=10,
                         help="Raised from the first run's 6 — the GRL ramp needs room to "
                              "destabilize-then-recover before patience judges it a failure. "
                              "See grl_lambda_schedule's docstring for what went wrong at 6.")
    parser.add_argument("--grl-warmup-epochs", type=int, default=12,
                         help="Epochs to ramp grl_lambda 0->max over, decoupled from --epochs "
                              "so early stopping can't front-load the ramp — see "
                              "grl_lambda_schedule's docstring.")
    parser.add_argument("--grl-max-lambda", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    exclude = tuple(args.exclude_domain)
    train_ds = FingerprintDataset(args.data, split="train", exclude_domains=exclude)
    val_ds = FingerprintDataset(args.data, split="val", exclude_domains=exclude)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, collate_fn=collate_fingerprint,
                               pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, collate_fn=collate_fingerprint,
                             pin_memory=device.type == "cuda")

    model = FingerprintExtractor(n_generators=N_DOMAINS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    task_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    domain_criterion = nn.CrossEntropyLoss()

    ckpt_path = CHECKPOINT_DIR / args.out
    best_task_acc = 0.0
    epochs_since_best = 0

    for epoch in range(1, args.epochs + 1):
        lam = grl_lambda_schedule(epoch, args.grl_warmup_epochs, args.grl_max_lambda)
        t0 = time.time()
        train_m = run_epoch(model, train_loader, optimizer, task_criterion, domain_criterion,
                             device, train=True, grl_lambda=lam)
        val_m = run_epoch(model, val_loader, optimizer, task_criterion, domain_criterion,
                           device, train=False)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs} [{elapsed:.0f}s] grl_lambda={lam:.3f}")
        print(f"  train: task_loss={train_m['task_loss']:.4f} domain_loss={train_m['domain_loss']:.4f} "
              f"task_acc={train_m['task_acc']:.4f}")
        print(f"  val  : task_loss={val_m['task_loss']:.4f} domain_loss={val_m['domain_loss']:.4f} "
              f"task_acc={val_m['task_acc']:.4f}")

        if val_m["task_acc"] > best_task_acc:
            best_task_acc = val_m["task_acc"]
            epochs_since_best = 0
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "n_generators": N_DOMAINS,
                "val_task_acc": best_task_acc,
                "excluded_domains": exclude,
                "classes": TASK_CLASSES,
            }, ckpt_path)
            print(f"  -> best checkpoint saved (task_acc={best_task_acc:.4f}) to {ckpt_path}")
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f"No improvement for {args.patience} epochs. Early stopping.")
                break

    print(f"\nTraining complete. Best val task_acc: {best_task_acc:.4f}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
