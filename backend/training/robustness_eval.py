"""
Honest reliability evaluation for the trained detector.

Two things a clean val-set accuracy number hides:

  1. CALIBRATION — softmax outputs are usually overconfident. A model that says
     "95% confident" should be right 95% of the time; if it is right 80% of the
     time, every downstream decision built on that number is wrong. Measured
     here as Expected Calibration Error (ECE).

  2. ROBUSTNESS — real uploads are recompressed by messaging apps, resized by
     CDNs, and screenshotted. Val images are pristine. Accuracy under those
     perturbations is the number that predicts real-world behaviour.

Usage:
    python training/robustness_eval.py --checkpoint checkpoints/efficientnet_b4_fake.pth
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.cnn_model import FakeImageDetector, CLASSES, NUM_CLASSES
from training.dataset import CLASS_NAMES, LABEL_MAP, EXTENSIONS, IMG_SIZE

NORMALIZE = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


# ─── perturbations that mimic real-world image handling ─────────────────────

# Defined as classes, not closures: Windows uses spawn for DataLoader workers,
# which pickles the dataset (and therefore the perturbation) — closures fail.

class Identity:
    def __call__(self, img: Image.Image) -> Image.Image:
        return img


class JpegRecompress:
    def __init__(self, quality: int):
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class Downscale:
    """Downscale then upscale back — mimics CDN/messenger resizing."""
    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        small = img.resize(
            (max(int(w * self.factor), 8), max(int(h * self.factor), 8)), Image.LANCZOS
        )
        return small.resize((w, h), Image.LANCZOS)


class Blur:
    def __init__(self, radius: float):
        self.radius = radius

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))


class GaussianNoise:
    def __init__(self, std: float):
        self.std = std

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.clip(arr + np.random.randn(*arr.shape) * self.std, 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))


PERTURBATIONS = {
    "none (pristine)":        Identity(),
    "JPEG q=90":              JpegRecompress(90),
    "JPEG q=75":              JpegRecompress(75),
    "JPEG q=50":              JpegRecompress(50),
    "JPEG q=30":              JpegRecompress(30),
    "downscale 0.5x":         Downscale(0.5),
    "downscale 0.25x":        Downscale(0.25),
    "blur r=1.0":             Blur(1.0),
    "blur r=2.0":             Blur(2.0),
    "gaussian noise s=0.02":  GaussianNoise(0.02),
    "gaussian noise s=0.05":  GaussianNoise(0.05),
}


# ─── dataset that applies a perturbation before the eval transform ──────────

class PerturbedDataset(Dataset):
    def __init__(self, root: str, split: str, perturb_fn, limit_per_class: int | None = None):
        base = Path(root) / split
        self.paths: list[tuple[Path, int]] = []
        for class_name, label in LABEL_MAP.items():
            class_dir = base / class_name
            if not class_dir.exists():
                continue
            imgs = sorted(p for p in class_dir.rglob("*") if p.suffix.lower() in EXTENSIONS)
            if limit_per_class:
                imgs = imgs[:limit_per_class]
            self.paths.extend((p, label) for p in imgs)
        self.perturb_fn = perturb_fn
        self.resize = T.Resize((IMG_SIZE, IMG_SIZE))
        self.to_tensor = T.ToTensor()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        path, label = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.perturb_fn(img)
        tensor = NORMALIZE(self.to_tensor(self.resize(img)))
        return tensor, torch.tensor(label, dtype=torch.long)


# ─── metrics ─────────────────────────────────────────────────────────────────

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """
    ECE: average gap between confidence and actual accuracy, bucketed by
    confidence. 0 = perfectly calibrated. Also returns the per-bin table so the
    direction of the error (over- vs under-confident) is visible.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        bin_conf = float(confidences[mask].mean())
        bin_acc = float(accuracies[mask].mean())
        ece += (count / len(labels)) * abs(bin_acc - bin_conf)
        rows.append((lo, hi, count, bin_conf, bin_acc))
    return float(ece), rows


def macro_f1(probs: np.ndarray, labels: np.ndarray) -> float:
    preds = probs.argmax(axis=1)
    f1s = []
    for c in range(NUM_CLASSES):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1s.append(2 * prec * rec / (prec + rec + 1e-8))
    return float(np.mean(f1s))


@torch.no_grad()
def collect_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(imgs)
        probs = F.softmax(logits.float(), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/efficientnet_b4_fake.pth")
    parser.add_argument("--data", default="../data")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-per-class", type=int, default=300,
                        help="Subsample per class per perturbation to keep the sweep tractable.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    state = torch.load(args.checkpoint, map_location=device)
    model = FakeImageDetector(pretrained=False).to(device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    print(f"Loaded {args.checkpoint} (val_macro_f1={state.get('val_macro_f1', '?')})")
    print(f"Sampling {args.limit_per_class} images/class per perturbation\n")

    print("=" * 78)
    print("ROBUSTNESS UNDER REAL-WORLD IMAGE HANDLING")
    print("=" * 78)
    header = f"{'Perturbation':<24} {'Accuracy':>10} {'Macro F1':>10} {'ECE':>8} {'MeanConf':>10}"
    print(header)
    print("-" * len(header))

    pristine_rows = None
    results = {}
    for name, fn in PERTURBATIONS.items():
        ds = PerturbedDataset(args.data, "val", fn, args.limit_per_class)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
        probs, labels = collect_probs(model, loader, device)
        acc = float((probs.argmax(axis=1) == labels).mean())
        f1 = macro_f1(probs, labels)
        ece, rows = expected_calibration_error(probs, labels)
        mean_conf = float(probs.max(axis=1).mean())
        results[name] = (acc, f1, ece, mean_conf)
        if name.startswith("none"):
            pristine_rows = rows
        print(f"{name:<24} {acc:>9.2%} {f1:>10.4f} {ece:>8.4f} {mean_conf:>9.2%}")

    print("\n" + "=" * 78)
    print("CALIBRATION DETAIL — pristine images")
    print("=" * 78)
    print(f"{'Confidence bin':<20} {'Count':>8} {'AvgConf':>10} {'Accuracy':>10} {'Gap':>10}")
    print("-" * 60)
    for lo, hi, count, bin_conf, bin_acc in pristine_rows or []:
        gap = bin_acc - bin_conf
        flag = "  <-- overconfident" if gap < -0.05 else ""
        print(f"({lo:.2f}, {hi:.2f}]{'':<8} {count:>8} {bin_conf:>9.2%} {bin_acc:>9.2%} {gap:>+9.2%}{flag}")

    pristine_acc = results["none (pristine)"][0]
    worst_name = min(results, key=lambda k: results[k][0])
    worst_acc = results[worst_name][0]
    print("\n" + "=" * 78)
    print(f"Pristine accuracy        : {pristine_acc:.2%}")
    print(f"Worst-case perturbation  : {worst_name} -> {worst_acc:.2%}")
    print(f"Robustness gap           : {pristine_acc - worst_acc:+.2%}")
    print("=" * 78)


if __name__ == "__main__":
    main()
