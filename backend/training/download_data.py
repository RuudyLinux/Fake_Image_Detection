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
import sys
import zipfile
from pathlib import Path


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Bare "kaggle" relies on the console-script shim being on PATH, which isn't
# guaranteed (e.g. a venv not activated in the calling shell) — it fails with
# FileNotFoundError, not a Kaggle API error, so the existing `except`-free
# `check=False` + returncode check below never even sees it. Invoking the
# module through the same interpreter that's running this script sidesteps
# PATH entirely.
KAGGLE_CMD = [sys.executable, "-m", "kaggle"]


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
        KAGGLE_CMD + ["datasets", "download", "-d", "xhlulu/140k-real-and-fake-faces",
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
        KAGGLE_CMD + ["datasets", "download", "-d", "sophatvathana/casia-dataset",
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
    """
    CASIA v2 contains Tp (tampered) and Au (authentic) folders. Both matter:

    Tp -> edited class, as always.

    Au -> real class, prefixed `casia_au_` so training/dataset.py's sampler
    can recognize it as a distinct sub-domain and give it equal sampling
    weight to the face-photo majority in `real`. Skipping Au here was a real,
    measured bug: `real` and `ai_generated` ended up 100% face closeups (the
    140k-faces dataset), while `edited` (from Tp) spans general photography --
    architecture, nature, animals, scenes. The model learned "not a face ->
    edited" as a shortcut having never seen a genuine unmanipulated non-face
    photo. Confirmed directly: unrelated demo images (a tower, a cheetah, a
    bus -- zero editing) were all classified EDITED at ~88% confidence.
    """
    tampered_dirs = list(dl_dir.rglob("Tp")) + list(dl_dir.rglob("tampered"))
    authentic_dirs = list(dl_dir.rglob("Au"))

    if not tampered_dirs:
        print("  WARNING: Could not find tampered images folder. Check extraction.")
    if not authentic_dirs:
        print("  WARNING: Could not find Au (authentic) folder -- 'real' will stay "
              "face-only and this download will not fix the non-face false-positive bug.")

    def _split_and_copy(files: list[Path], dst_class: str, prefix: str = ""):
        random.shuffle(files)
        split_idx = int(len(files) * 0.8)
        splits = {"train": files[:split_idx], "val": files[split_idx:]}
        for split, split_files in splits.items():
            dst = output_dir / split / dst_class
            dst.mkdir(parents=True, exist_ok=True)
            chosen = split_files[:max_per_split]
            for f in chosen:
                shutil.copy2(f, dst / f"{prefix}{f.name}")
            print(f"  -> {split}/{dst_class}: {len(chosen)} images")

    if tampered_dirs:
        all_tampered = []
        for d in tampered_dirs:
            all_tampered.extend([f for f in d.rglob("*") if f.suffix.lower() in EXTENSIONS])
        _split_and_copy(all_tampered, "edited")

    if authentic_dirs:
        all_authentic = []
        for d in authentic_dirs:
            all_authentic.extend([f for f in d.rglob("*") if f.suffix.lower() in EXTENSIONS])
        _split_and_copy(all_authentic, "real", prefix="casia_au_")

    print("  CASIA organized.")


# ─── MIT Indoor Scenes — grows the thin `ind` sub-category ───────────────────

def download_indoor67(output_dir: Path, n_train: int = 800, n_val: int = 200):
    """
    Kaggle: itsahmad/indoor-scenes-cvpr-2019 (MIT Indoor Scenes / IndoorCVPR09)
    ~2.5 GB, 67 indoor categories, ~15k images total.

    Populates: real/, prefixed `indoor67_`. dataset.py's group_key merges
    this into the SAME sampler bucket as `casia_au_*_ind_*` (not a new
    parallel group) — CASIA's own Au 'ind' split is only 186 images total,
    the sampler fix (1a677e5) could only oversample what existed, and 186
    unique images is too thin to fix real/ind's false-EDITED rate without
    real memorization risk. This is more source images for that same
    category, not a new category. See PROJECT_SPEC.md Honest Status #4.
    """
    print("\nDownloading MIT Indoor Scenes (Kaggle, ~2.5GB)...")
    dl_dir = output_dir / "_tmp_indoor67"
    dl_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        KAGGLE_CMD + ["datasets", "download", "-d", "itsahmad/indoor-scenes-cvpr-2019",
                      "-p", str(dl_dir), "--unzip"],
        check=False,
    )
    if result.returncode != 0:
        print("\n  Kaggle download failed. Manual steps:")
        print("  1. Visit https://www.kaggle.com/datasets/itsahmad/indoor-scenes-cvpr-2019")
        print(f"  2. Download + extract to: {dl_dir}")
        print("  3. Re-run with --skip-indoor-download")
        return

    _sample_and_copy(dl_dir, output_dir, "real", "indoor67_", n_train, n_val)
    shutil.rmtree(dl_dir, ignore_errors=True)
    print("  Indoor Scenes organized.")


# ─── TextOCR — grows the thin `txt` sub-category ──────────────────────────────

def download_textocr(output_dir: Path, n_train: int = 800, n_val: int = 200):
    """
    Kaggle: robikscube/textocr-text-extraction-from-images-dataset
    ~7.2 GB full dataset (whole zip is downloaded; only n_train+n_val images
    are kept — the rest is deleted after sampling). Real photos containing
    text in the wild (signs, storefronts, documents), sourced from OpenImages.

    Populates: real/, prefixed `textocr_`. Merges into the SAME sampler
    bucket as `casia_au_*_txt_*` — CASIA's own Au 'txt' split is only 79
    images total, the thinnest category and the worst-performing one
    (0.150 accuracy even after the sampler fix). See PROJECT_SPEC.md
    Honest Status #4.
    """
    print("\nDownloading TextOCR (Kaggle, ~7.2GB — only a subsample is kept)...")
    dl_dir = output_dir / "_tmp_textocr"
    dl_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        KAGGLE_CMD + ["datasets", "download", "-d",
                      "robikscube/textocr-text-extraction-from-images-dataset",
                      "-p", str(dl_dir), "--unzip"],
        check=False,
    )
    if result.returncode != 0:
        print("\n  Kaggle download failed. Manual steps:")
        print("  1. Visit https://www.kaggle.com/datasets/robikscube/textocr-text-extraction-from-images-dataset")
        print(f"  2. Download + extract to: {dl_dir}")
        print("  3. Re-run with --skip-textocr-download")
        return

    _sample_and_copy(dl_dir, output_dir, "real", "textocr_", n_train, n_val)
    shutil.rmtree(dl_dir, ignore_errors=True)
    print("  TextOCR organized.")


def _sample_and_copy(dl_dir: Path, output_dir: Path, dst_class: str, prefix: str,
                      n_train: int, n_val: int):
    all_imgs = [f for f in dl_dir.rglob("*") if f.suffix.lower() in EXTENSIONS]
    random.shuffle(all_imgs)
    splits = {"train": all_imgs[:n_train], "val": all_imgs[n_train:n_train + n_val]}
    for split, files in splits.items():
        dst = output_dir / split / dst_class
        dst.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dst / f"{prefix}{f.name}")
        print(f"  -> {split}/{dst_class}: {len(files)} {prefix.rstrip('_')} images")


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
    parser.add_argument("--source",  choices=["140k", "casia", "sd", "indoor", "textocr", "all", "verify"],
                        default="all",
                        help="Data source to download, or 'verify' to just show dataset stats. "
                             "'indoor'/'textocr' are not included in 'all' — they're targeted fixes "
                             "for the thin real/ind and real/txt sampler buckets, not part of the "
                             "base pipeline.")
    parser.add_argument("--output",  default="../data")
    parser.add_argument("--max",     type=int, default=10000, help="Max images per class per split")
    parser.add_argument("--sd-n",    type=int, default=3000,  help="SD images to generate")
    parser.add_argument("--indoor-n",  type=int, default=800, help="Indoor67 train images (val = n/4)")
    parser.add_argument("--textocr-n", type=int, default=800, help="TextOCR train images (val = n/4)")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if args.source in ("140k", "all"):
        download_140k(output, max_per_class=args.max)
    if args.source in ("casia", "all"):
        download_casia(output, max_per_split=args.max)
    if args.source == "sd":
        generate_sd_images(output, n=args.sd_n)
    if args.source == "indoor":
        download_indoor67(output, n_train=args.indoor_n, n_val=args.indoor_n // 4)
    if args.source == "textocr":
        download_textocr(output, n_train=args.textocr_n, n_val=args.textocr_n // 4)

    verify(output)


if __name__ == "__main__":
    main()
