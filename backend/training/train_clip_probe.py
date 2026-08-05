"""
Train the CLIP linear probe (see models/clip_detector.py for why it's frozen).

Two stages, deliberately split:

  1. EXTRACT — run every image through the frozen CLIP vision tower once and
     cache the feature vectors to disk. This is the expensive part (~30k images)
     and it only has to happen once, because a frozen backbone produces
     deterministic features.

  2. FIT — train the probe on cached features. Seconds, not hours, so the
     sampler weighting, class balance, and head capacity can all be re-tuned
     cheaply without touching the backbone again.

Usage:
    python training/train_clip_probe.py --data ../../data --extract
    python training/train_clip_probe.py --data ../../data --fit
    python training/train_clip_probe.py --data ../../data --extract --fit
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.clip_detector import CLIPFeatureExtractor, CLIPProbe, CLIP_MODEL_ID, CLASSES
from training.dataset import FakeImageDataset, CLASS_NAMES

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
CACHE_DIR = CHECKPOINT_DIR / "clip_features"


class RawImageDataset(Dataset):
    """Yields PIL images + labels + paths. CLIP does its own preprocessing, so
    no ImageNet normalization here — applying the CNN pipeline's transform
    would feed CLIP inputs it was never trained on."""

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path, label = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        return img, label, str(path)


def _collate(batch):
    imgs, labels, paths = zip(*batch)
    return list(imgs), torch.tensor(labels, dtype=torch.long), list(paths)


def extract(data_dir: str, split: str, model_id: str, batch_size: int):
    ds = FakeImageDataset(data_dir, split=split)
    raw = RawImageDataset(ds.paths)
    loader = DataLoader(raw, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=_collate)

    extractor = CLIPFeatureExtractor(model_id)
    dim = extractor.output_dim
    print(f"[{split}] {len(raw)} images -> CLIP {model_id} (dim={dim})")

    feats, labels, paths = [], [], []
    for i, (imgs, lbls, pths) in enumerate(loader):
        feats.append(extractor.encode_pil(imgs).cpu().numpy().astype(np.float32))
        labels.append(lbls.numpy())
        paths.extend(pths)
        if (i + 1) % 20 == 0:
            done = min((i + 1) * batch_size, len(raw))
            print(f"  {done}/{len(raw)}", flush=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"{split}.npz"
    np.savez_compressed(
        out,
        features=np.concatenate(feats),
        labels=np.concatenate(labels),
        paths=np.array(paths),
        model_id=model_id,
    )
    print(f"  saved -> {out}")


def _subdomain_key(path_str: str) -> str:
    """Mirrors FakeImageDataset.make_sampler's grouping so the probe gets the
    same sub-domain balancing the CNN members get. Without it the probe would
    re-learn the face-majority bias the sampler exists to prevent."""
    name = Path(path_str).name
    if name.startswith("casia_au_"):
        parts = name.split("_")
        return f"casia_au_{parts[3]}" if len(parts) >= 4 else "casia_au_"
    if name.startswith(("Tp_S_", "Tp_D_", "Sp_S_", "Sp_D_")):
        return name[:4]
    if name.startswith("indoor67_"):
        return "casia_au_ind"
    if name.startswith("textocr_"):
        return "casia_au_txt"
    if name.startswith("sd14_"):
        return "sd14_"
    return "base"


def _balanced_weights(labels: np.ndarray, paths: np.ndarray) -> np.ndarray:
    groups = np.array([_subdomain_key(p) for p in paths])
    weights = np.zeros(len(labels), dtype=np.float64)
    n_classes = len(CLASS_NAMES)
    for lbl in np.unique(labels):
        cls_mask = labels == lbl
        cls_groups = np.unique(groups[cls_mask])
        for g in cls_groups:
            m = cls_mask & (groups == g)
            weights[m] = 1.0 / (n_classes * len(cls_groups) * m.sum())
    return weights


def macro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    f1s = []
    for c in range(len(CLASSES)):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1s.append(2 * prec * rec / (prec + rec + 1e-9))
    return float(np.mean(f1s))


def fit(epochs: int, lr: float, hidden: int | None, batch_size: int, out_name: str):
    tr = np.load(CACHE_DIR / "train.npz", allow_pickle=True)
    va = np.load(CACHE_DIR / "val.npz", allow_pickle=True)

    Xtr = torch.from_numpy(tr["features"]); ytr = torch.from_numpy(tr["labels"]).long()
    Xva = torch.from_numpy(va["features"]); yva = torch.from_numpy(va["labels"]).long()
    print(f"train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = CLIPProbe(feature_dim=Xtr.shape[1], hidden=hidden).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    weights = _balanced_weights(tr["labels"], tr["paths"])
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                    num_samples=len(weights), replacement=True)
    idx_loader = DataLoader(torch.arange(len(ytr)), batch_size=batch_size, sampler=sampler)

    Xva_d, yva_d = Xva.to(device), yva.to(device)
    best_f1, best_state = 0.0, None

    for ep in range(1, epochs + 1):
        probe.train()
        total = 0.0
        for idx in idx_loader:
            xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
            opt.zero_grad()
            loss = crit(probe(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()

        probe.eval()
        with torch.no_grad():
            preds = probe(Xva_d).argmax(1).cpu().numpy()
        acc = float((preds == yva.numpy()).mean())
        f1 = macro_f1(preds, yva.numpy())
        flag = ""
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.clone() for k, v in probe.state_dict().items()}
            flag = "  <- best"
        print(f"epoch {ep:>3}/{epochs}  loss={total/max(len(idx_loader),1):.4f}  "
              f"val_acc={acc:.4f}  val_macro_f1={f1:.4f}{flag}")

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    out = CHECKPOINT_DIR / out_name
    torch.save({
        "probe": best_state,
        "feature_dim": Xtr.shape[1],
        "hidden": hidden,
        "classes": CLASSES,
        "clip_model_id": str(tr["model_id"]),
        "val_macro_f1": best_f1,
    }, out)
    print(f"\nBest val macro-F1: {best_f1:.4f}\nSaved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../data")
    ap.add_argument("--model-id", default=CLIP_MODEL_ID)
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--extract-batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=None,
                    help="Omit for a pure linear probe (UnivFD's setting; recommended).")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", default="clip_probe.pth")
    args = ap.parse_args()

    if not args.extract and not args.fit:
        ap.error("pass --extract and/or --fit")
    if args.extract:
        for split in ("train", "val"):
            extract(args.data, split, args.model_id, args.extract_batch)
    if args.fit:
        fit(args.epochs, args.lr, args.hidden, args.batch_size, args.out)
