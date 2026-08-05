"""
Dataset loader for fake image detection (3-class).

Expected directory layout:
    data/
      train/
        real/          *.jpg / *.png / *.jpeg / *.webp
        ai_generated/  *.jpg / *.png / *.jpeg / *.webp
        edited/        *.jpg / *.png / *.jpeg / *.webp
      val/
        real/
        ai_generated/
        edited/

Label mapping: real=0, ai_generated=1, edited=2

Compatible sources:
  real:          FFHQ, Flickr-Faces (140k dataset real split)
  ai_generated:  StyleGAN faces (140k fake split), DALL-E, Stable Diffusion generated
  edited:        CASIA v2, Coverage, CoMoFoD (copy-move + splicing + retouching)
"""

import io
import random
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms as T
import numpy as np

IMG_SIZE = 380  # EfficientNet-B4 native resolution
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

CLASS_NAMES = ["real", "ai_generated", "edited"]
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}


class JPEGAugment:
    """Simulate JPEG compression — hardens model against compression artifacts."""
    def __init__(self, quality_range=(60, 95)):
        self.quality_range = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        q = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class GaussianNoise:
    def __init__(self, std_range=(0.0, 0.02)):
        self.std_range = std_range

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        std = random.uniform(*self.std_range)
        return (tensor + torch.randn_like(tensor) * std).clamp(0, 1)


class RandomDownscale:
    """
    Downscale then restore size — simulates CDN/messenger resizing and
    thumbnailing. Robustness testing showed the model losing ~15 points of
    accuracy on 0.25x-rescaled inputs, because nothing in the original recipe
    taught it to survive resampling.
    """
    def __init__(self, scale_range=(0.35, 1.0), p: float = 0.4):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        factor = random.uniform(*self.scale_range)
        w, h = img.size
        small = img.resize((max(int(w * factor), 8), max(int(h * factor), 8)), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


class RandomBlur:
    """
    Mild Gaussian blur. Blur was the single worst robustness failure (99.7% ->
    78.8% at radius 2.0) precisely because it was absent from training.
    """
    def __init__(self, radius_range=(0.0, 1.8), p: float = 0.3):
        self.radius_range = radius_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        radius = random.uniform(*self.radius_range)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


def build_transforms(split: str, img_size: int = IMG_SIZE) -> T.Compose:
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    if split == "train":
        return T.Compose([
            T.Resize((img_size + 40, img_size + 40)),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=12),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.RandomGrayscale(p=0.05),
            RandomDownscale(scale_range=(0.35, 1.0), p=0.4),
            RandomBlur(radius_range=(0.0, 1.8), p=0.3),
            JPEGAugment(quality_range=(40, 95)),
            T.ToTensor(),
            GaussianNoise(std_range=(0.0, 0.03)),
            normalize,
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            normalize,
        ])


class FakeImageDataset(Dataset):
    def __init__(self, root: str, split: str = "train", img_size: int = IMG_SIZE):
        self.split = split
        self.img_size = img_size
        base = Path(root) / split

        self.paths: list[Tuple[Path, int]] = []
        counts = {}

        for class_name, label in LABEL_MAP.items():
            class_dir = base / class_name
            if not class_dir.exists():
                print(f"  WARNING: {class_dir} not found, skipping")
                counts[class_name] = 0
                continue
            imgs = [p for p in class_dir.rglob("*") if p.suffix.lower() in EXTENSIONS]
            self.paths.extend([(p, label) for p in imgs])
            counts[class_name] = len(imgs)

        if not self.paths:
            raise FileNotFoundError(
                f"No images found under {base}. "
                "Expected subdirs: real/, ai_generated/, edited/"
            )

        self.transform = build_transforms(split, img_size)

        total = len(self.paths)
        print(f"[{split}] " + "  ".join(f"{k}={v:,}" for k, v in counts.items()) + f"  total={total:,}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.img_size, self.img_size), color=(128, 128, 128))
        return self.transform(img), torch.tensor(label, dtype=torch.long)

    # Filename prefixes marking a distinct injected sub-domain within a class,
    # so it can be given equal sampling weight to the majority data instead of
    # being drowned out by it. Measured directly: CASIA "Au" (authentic,
    # non-face) photos injected into `real` alongside 8000 face photos were
    # outnumbered 2:1, and two full fine-tune rounds barely moved held-out
    # accuracy on non-face photos (0 -> 17/40 -> 18/40) — the model kept
    # seeing faces twice as often and the non-face concept never caught up.
    SUBDOMAIN_PREFIXES = ("casia_au_", "sd14_")

    def make_sampler(self) -> WeightedRandomSampler:
        """
        Balanced sampler: equal expected frequency per class, AND within each
        class, equal expected frequency per data-source sub-domain (see
        SUBDOMAIN_PREFIXES). Class-only balancing leaves a minority sub-domain
        proportionally starved even though its class is balanced.
        """
        num_classes = len(CLASS_NAMES)

        def group_key(path: Path) -> str:
            name = path.name
            for prefix in self.SUBDOMAIN_PREFIXES:
                if name.startswith(prefix):
                    return prefix
            return "base"

        groups = [group_key(path) for path, _ in self.paths]
        group_counts: dict[tuple, int] = {}
        for (_, lbl), g in zip(self.paths, groups):
            group_counts[(lbl, g)] = group_counts.get((lbl, g), 0) + 1

        groups_per_class: dict[int, int] = {}
        for (lbl, _g) in group_counts:
            groups_per_class[lbl] = groups_per_class.get(lbl, 0) + 1

        sample_weights = []
        for (_, lbl), g in zip(self.paths, groups):
            n_groups = groups_per_class[lbl]
            group_size = group_counts[(lbl, g)]
            sample_weights.append(1.0 / (num_classes * n_groups * group_size))

        return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
