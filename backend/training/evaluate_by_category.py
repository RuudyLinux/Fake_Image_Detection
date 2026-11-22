"""
Per-category evaluation — breaks val accuracy down by the same sub-domain
groups the balanced sampler uses (see FakeImageDataset.make_sampler), instead
of only the class-level aggregate evaluate.py reports.

Why this exists: overall val macro_f1 is blind to a minority sub-population.
txt/ind non-face REAL photos are 20/47 out of 3001 REAL val images — a real
improvement there barely moves the aggregate, while EDITED re-adapting to its
own tamper-type rebalance can drag the aggregate down first. A checkpoint that
is genuinely better where the sampler fix targets can lose on macro_f1 alone.
This script measures the thing the fix was actually for.

Usage:
    python training/evaluate_by_category.py --checkpoint ../checkpoints/efficientnet_b4_fake.pth --data ../data
    python training/evaluate_by_category.py --checkpoint-dir ../checkpoints/epoch_ckpts --data ../data
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.cnn_model import build_model, CLASSES
from training.dataset import FakeImageDataset


_CATEGORY_ALIASES = {"indoor67_": "ind", "textocr_": "txt"}


def category_of(path: Path) -> str:
    """Same grouping as FakeImageDataset.make_sampler's group_key, plus a
    readable 'face' label for the majority face-crop group."""
    name = path.name
    if name.startswith("casia_au_"):
        parts = name.split("_")
        return f"real/{parts[3]}" if len(parts) >= 4 else "real/casia_au_other"
    if name.startswith(("Tp_S_", "Tp_D_", "Sp_S_", "Sp_D_")):
        return f"edited/{name[:4]}"
    if name.startswith("sd14_"):
        return "ai_generated/sd14"
    # Generation Fingerprint research data (RESEARCH_PROPOSAL_GENERATION_FINGERPRINT.md
    # §5) — without this, defactify_sdxl_00012.jpg etc. silently fall through to the
    # generic real/face or ai_generated/face bucket below, understating both the
    # face-photo baseline's own accuracy AND hiding per-generator accuracy on data
    # that DOES have a known generator label. Found by inspection when a baseline
    # retrain's val_macro_f1 shifted unexpectedly after a concurrent research-data
    # download landed partial files mid-run — see git history for that incident.
    if name.startswith("defactify_coco_"):
        return "real/defactify_coco"
    for prefix in ("defactify_", "openfake_"):
        if name.startswith(prefix):
            parts = name.split("_")
            gen = parts[1] if len(parts) >= 2 else "unknown"
            return f"ai_generated/{prefix.rstrip('_')}_{gen}"
    for alias_prefix, cat in _CATEGORY_ALIASES.items():
        if name.startswith(alias_prefix):
            return f"real/{cat}"  # merged into the same bucket as casia_au_<cat>
    label_dir = path.parent.name
    if label_dir == "real":
        return "real/face"
    if label_dir == "ai_generated":
        return "ai_generated/face"
    return f"{label_dir}/other"


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path: str, data_dir: str, batch_size: int, workers: int, device):
    # weights_only=False: these are this project's own locally-produced
    # checkpoints, not untrusted downloads, so the stricter default is
    # unnecessary friction here. (Separately: forcing CPU for this eval must
    # go through --device, NOT CUDA_VISIBLE_DEVICES="" — hiding the GPU
    # entirely breaks torch.load's device validation for a GPU-saved
    # checkpoint even with map_location=cpu. See --device's help text.)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = state.get("arch", "efficientnet_b4")
    img_size = state.get("img_size", 380)

    model, _ = build_model(arch, pretrained=False)
    model.load_state_dict(state["model"])
    model = model.to(device).eval()

    val_ds = FakeImageDataset(data_dir, split="val", img_size=img_size)
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    total_batches = len(loader)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for step, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device)
            with torch.autocast(device_type=device.type):
                logits = model(imgs)
            preds = F.softmax(logits, dim=1).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)
            # flush=True: without it this loop is silent until process exit
            # when stdout is redirected to a file (not a tty) — no way to
            # tell "still working" from "stuck" on a slow CPU eval otherwise.
            if (step + 1) % 10 == 0 or (step + 1) == total_batches:
                pct = 100 * (step + 1) / total_batches
                print(f"  eval progress: {step + 1}/{total_batches} batches ({pct:.1f}%)", flush=True)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    cats = [category_of(p) for p, _ in val_ds.paths]

    per_cat = {}
    for cat, pred, label in zip(cats, all_preds, all_labels):
        d = per_cat.setdefault(cat, {"n": 0, "correct": 0, "pred_counts": {c: 0 for c in CLASSES}})
        d["n"] += 1
        d["correct"] += int(pred == label)
        d["pred_counts"][CLASSES[pred]] += 1

    for cat, d in per_cat.items():
        d["accuracy"] = d["correct"] / d["n"]

    return {
        "checkpoint": str(checkpoint_path),
        "arch": arch,
        "epoch": state.get("epoch"),
        "val_macro_f1": state.get("val_macro_f1"),
        "per_category": per_cat,
    }


def print_report(result: dict):
    print(f"\n{'='*78}")
    print(f"{result['checkpoint']}  (epoch={result['epoch']}, val_macro_f1={result['val_macro_f1']})")
    print(f"{'='*78}")
    header = f"{'category':>24}  {'n':>6}  {'acc':>7}  predicted-as"
    print(header)
    print("-" * len(header))
    for cat, d in sorted(result["per_category"].items()):
        preds_str = "  ".join(f"{c}={v}" for c, v in d["pred_counts"].items() if v)
        print(f"{cat:>24}  {d['n']:>6}  {d['accuracy']:>7.3f}  {preds_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",     default=None, help="Single checkpoint to evaluate.")
    parser.add_argument("--checkpoint-dir", default=None, help="Evaluate every .pth in this dir.")
    parser.add_argument("--data",           default="../data")
    parser.add_argument("--batch-size",     type=int, default=16)
    parser.add_argument("--workers",        type=int, default=2)
    parser.add_argument("--out-json",       default=None, help="Write full results here as JSON.")
    parser.add_argument("--device",         default=None, choices=["cuda", "cpu"],
                         help="Force a device — e.g. 'cpu' to run this eval alongside a GPU-bound "
                              "training job without VRAM contention. Deliberately NOT done via "
                              "CUDA_VISIBLE_DEVICES=\"\": that hides the GPU from torch entirely, "
                              "which breaks torch.load's internal device validation for a "
                              "GPU-saved checkpoint even when map_location=cpu is given (raises "
                              "'torch.cuda.device_count() is 0' regardless of map_location — a "
                              "real torch quirk, hit and diagnosed this session). Leaving CUDA "
                              "visible and forcing compute to cpu here instead sidesteps it.")
    args = parser.parse_args()

    if not args.checkpoint and not args.checkpoint_dir:
        parser.error("Pass --checkpoint or --checkpoint-dir")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpts = [args.checkpoint] if args.checkpoint else sorted(
        str(p) for p in Path(args.checkpoint_dir).glob("*.pth")
    )

    all_results = []
    for ckpt in ckpts:
        result = evaluate_checkpoint(ckpt, args.data, args.batch_size, args.workers, device)
        print_report(result)
        all_results.append(result)

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(all_results, indent=2))
        print(f"\nWrote {args.out_json}")
