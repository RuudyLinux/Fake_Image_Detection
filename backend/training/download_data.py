"""
Dataset acquisition helper for 3-class fake image detection.
  Classes: real | ai_generated | edited

─── Quick Start ─────────────────────────────────────────────────────────────
  pip install kaggle
  # Put kaggle.json in ~/.kaggle/
  python training/download_data.py --source all --output ../data

─── Source Overview ─────────────────────────────────────────────────────────
  REAL        → 140k dataset (Flickr faces) or FFHQ
  AI_GENERATED→ 140k dataset (StyleGAN faces) + optional Stable Diffusion
  EDITED      → CASIA v2 (splicing + copy-move) from Kaggle
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import subprocess
import shutil
import random
import zipfile
from pathlib import Path


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _copy_images(src: Path, dst: Path, limit: int | None = None):
    dst.mkdir(parents=True, exist_ok=True)
    files = [f for f in src.rglob("*") if f.suffix.lower() in EXTENSIONS]
    if limit:
        files = random.sample(files, min(limit, len(files)))
    for f in files:
        shutil.copy2(f, dst / f.name)
    print(f"  → {dst.relative_to(dst.parent.parent.parent)}: {len(files)} images")
    return len(files)


# ─── 140k Real and Fake Faces ──────────────────────────────────────────────────

def download_140k(output_dir: Path, max_per_class: int = 10000):
    """
    Kaggle: xhlulu/140k-real-and-fake-faces
    70k real (Flickr) + 70k fake (StyleGAN). ~2.5 GB.
    Populates: real/ and ai_generated/
    """
    print("\n[1/3] Downloading 140k Real and Fake Faces (Kaggle)...")
    dl_dir = output_dir / "_tmp_140k"
    dl_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", "xhlulu/140k-real-and-fake-faces",
         "-p", str(dl_dir), "--unzip"],
        check=False,
    )

    if result.returncode != 0:
        print("\n  Kaggle download failed. Manual steps:")
        print("  1. Visit https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces")
        print(f"  2. Download + extract to: {dl_dir}")
        print("  3. Re-run with --skip-140k-download")
        return

    for split in ["train", "val"]:
        src_split = "train" if split == "train" else "test"
        for src_cls, dst_cls in [("real", "real"), ("fake", "ai_generated")]:
            base_140k = dl_dir / "real_vs_fake"
            inner_140k = base_140k / "real-vs-fake" if (base_140k / "real-vs-fake").exists() else base_140k / "real-or-fake"
            src = inner_140k / src_split / src_cls
            dst = output_dir / split / dst_cls
            if src.exists():
                _copy_images(src, dst, limit=max_per_class)
            else:
                print(f"  WARNING: {src} not found")

    shutil.rmtree(dl_dir, ignore_errors=True)
    print("  140k dataset organized.")


# ─── CASIA v2 — Edited/Manipulated images ─────────────────────────────────────

def download_casia(output_dir: Path, max_per_split: int = 5000):
    """
    Kaggle: sophatvathana/casia-dataset
    CASIA v2: copy-move + splicing manipulations. ~1 GB.
    Populates: edited/

    Alternative: https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset
    """
    print("\n[2/3] Downloading CASIA v2 (edited/manipulated images)...")
    dl_dir = output_dir / "_tmp_casia"
    dl_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", "sophatvathana/casia-dataset",
         "-p", str(dl_dir), "--unzip"],
        check=False,
    )

    if result.returncode != 0:
        print("\n  Kaggle download failed. Manual steps:")
        print("  1. Visit https://www.kaggle.com/datasets/sophatvathana/casia-dataset")
        print(f"  2. Download + extract to: {dl_dir}")
        print("  3. Re-run with --skip-casia-download")
        _print_casia_alternatives()
        return

    _organize_casia(dl_dir, output_dir, max_per_split)
    shutil.rmtree(dl_dir, ignore_errors=True)


def _organize_casia(dl_dir: Path, output_dir: Path, max_per_split: int):
    """CASIA v2 contains Tp (tampered) and Au (authentic) folders."""
    tampered_dirs = list(dl_dir.rglob("Tp")) + list(dl_dir.rglob("tampered"))
    if not tampered_dirs:
        print("  WARNING: Could not find tampered images folder. Check extraction.")
        return

    all_tampered = []
    for d in tampered_dirs:
        all_tampered.extend([f for f in d.rglob("*") if f.suffix.lower() in EXTENSIONS])

    random.shuffle(all_tampered)
    split_idx = int(len(all_tampered) * 0.8)
    splits = {
        "train": all_tampered[:split_idx],
        "val": all_tampered[split_idx:],
    }

    for split, files in splits.items():
        dst = output_dir / split / "edited"
        dst.mkdir(parents=True, exist_ok=True)
        chosen = files[:max_per_split]
        for f in chosen:
            shutil.copy2(f, dst / f.name)
        print(f"  → {split}/edited: {len(chosen)} images")

    print("  CASIA organized.")


def _print_casia_alternatives():
    print("\n  Alternative edited image datasets:")
    print("  • CoMoFoD: https://www.vcl.fer.hr/comofod/")
    print("  • Coverage: https://github.com/wenbihan/coverage")
    print("  • Columbia Uncompressed: https://www.ee.columbia.edu/ln/dvmm/downloads/AuthSplicedDataSet/")
    print("  • NIST16: https://www.nist.gov/system/files/documents/2019/01/17/nist16_0.zip")
    print(f"\n  Place edited images in: data/train/edited/ and data/val/edited/")


# ─── Diffusion-model generated (fixes a real, measured blind spot) ─────────────
#
# Without this, ai_generated is 100% StyleGAN (from the 140k dataset). A model
# trained only on that is blind to diffusion-model artifacts — measured
# directly: 0/30 correct (25/30 confidently wrong as REAL) on genuine Stable
# Diffusion faces, with near-zero epistemic uncertainty on the wrong ones
# because BOTH ensemble members share the identical blind spot. Injecting even
# a modest fraction of real diffusion output into training closed this: 0/30
# correct -> 27/30 correct, 0 confidently wrong, on a held-out test set never
# seen in training. See training/generate_diffusion_data.py for the full story.

def generate_sd_images(output_dir: Path, n_train: int = 800, n_val: int = 200, steps: int = 20):
    """Thin wrapper — see training/generate_diffusion_data.py for the real implementation."""
    from training.generate_diffusion_data import generate
    generate(output_dir, n_train=n_train, n_val=n_val, steps=steps)


# ─── Verify dataset structure ──────────────────────────────────────────────────

def verify(output_dir: Path):
    print("\n─── Dataset Summary ──────────────────────────────────────────")
    total = 0
    for split in ["train", "val"]:
        for cls in ["real", "ai_generated", "edited"]:
            d = output_dir / split / cls
            n = len([f for f in d.rglob("*") if f.suffix.lower() in EXTENSIONS]) if d.exists() else 0
            status = "✓" if n >= 500 else ("⚠" if n > 0 else "✗")
            print(f"  {status}  {split:5}/{cls:14}: {n:6,} images")
            total += n
    print(f"\n  Total: {total:,} images")
    print("  Recommended minimum: 1000 per class per split")
    print("─────────────────────────────────────────────────────────────")


# ─── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",  choices=["140k", "casia", "sd", "all", "verify"], default="all",
                        help="Data source to download, or 'verify' to just show dataset stats")
    parser.add_argument("--output",  default="../data")
    parser.add_argument("--max",     type=int, default=10000, help="Max images per class per split")
    parser.add_argument("--sd-n",    type=int, default=3000,  help="SD images to generate")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if args.source in ("140k", "all"):
        download_140k(output, max_per_class=args.max)
    if args.source in ("casia", "all"):
        download_casia(output, max_per_split=args.max)
    if args.source == "sd":
        generate_sd_images(output, n=args.sd_n)

    verify(output)


if __name__ == "__main__":
    main()
