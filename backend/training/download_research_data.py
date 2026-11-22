"""
Dataset acquisition for the Generation Fingerprint research module — see
RESEARCH_PROPOSAL_GENERATION_FINGERPRINT.md §5 for the sourcing rationale
(each source below was independently verified, not taken on reputation).

This is additive to download_data.py, not a replacement — it populates the
SAME data/train, data/val class dirs with new, generator-tagged filename
prefixes, so training/dataset.py's existing sampler and the new
--exclude-prefix LOGO flag (train.py) both pick them up for free.

─── Quick Start ─────────────────────────────────────────────────────────────
  pip install datasets huggingface_hub   # already added to requirements.txt
  python training/download_research_data.py --source defactify --output ../data
  python training/download_research_data.py --source openfake --output ../data
─────────────────────────────────────────────────────────────────────────────

Sources:
  defactify  → Rajarshi-Roy-research/Defactify_Image_Dataset (HF, CC BY 4.0)
               5 generators with a real per-image generator-ID label
               (Label_B: 1=SD2.1, 2=SDXL, 3=SD3, 4=DALL-E3, 5=MJv6).
               Confirmed by direct inspection — see proposal §5.
  openfake   → ComplexDataLab/OpenFake (HF, CC-BY-SA-4.0 / non-commercial for
               proprietary-generator subsets). 3.44TB full — this script only
               ever streams + filters a bounded sample per target generator,
               never bulk-downloads. `model` column confirmed to carry exact
               per-image generator identity by direct inspection.

Both intentionally exclude DEFACTO (Kaggle, gated request-form access) — that
one needs a human to submit the access request first; no API key alone gets
it. See proposal §5's "Next Steps" note.
"""

import argparse
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class StreamingBucket:
    """
    Saves each accepted image to disk IMMEDIATELY instead of accumulating a
    Python list of PIL Image objects in memory. The earlier version of this
    file buffered every bucket in full until the end of the scan — with
    ~2000 images/generator x 5+ generators that's thousands of decoded PIL
    images resident at once, which measured at 7.8GB RSS in practice and
    starved a concurrent training run into an OOM-equivalent DataLoader
    crash (torch shared-memory allocation failing with Windows error 1455,
    "the paging file is too small"). Real, observed failure, not a
    hypothetical — see RESEARCH_PROPOSAL_GENERATION_FINGERPRINT.md's Next
    Steps section / this session's git history for the incident. Fixed by
    writing each image out as soon as it's accepted and keeping only a
    running count, so peak memory is O(1) per bucket instead of O(target).

    Train/val split is decided per-image up front (val_frac chance) rather
    than shuffle-then-slice, since nothing is buffered to shuffle.
    """

    def __init__(self, output_dir: Path, dst_class: str, prefix: str,
                 n_train: int, n_val: int, val_frac: float = 0.2):
        self.train_dir = output_dir / "train" / dst_class
        self.val_dir = output_dir / "val" / dst_class
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.n_train_target = n_train
        self.n_val_target = n_val
        self.val_frac = val_frac
        self.n_train = 0
        self.n_val = 0

    def __len__(self):
        return self.n_train + self.n_val

    def is_full(self) -> bool:
        return self.n_train >= self.n_train_target and self.n_val >= self.n_val_target

    def add(self, img) -> bool:
        """Returns True if the image was accepted (bucket had room)."""
        want_val = self.n_val < self.n_val_target and (
            self.n_train >= self.n_train_target or random.random() < self.val_frac
        )
        if want_val:
            dst, idx = self.val_dir, self.n_val
        elif self.n_train < self.n_train_target:
            dst, idx = self.train_dir, self.n_train
        else:
            return False

        try:
            img.convert("RGB").save(dst / f"{self.prefix}{idx:05d}.jpg", format="JPEG", quality=95)
        except Exception as e:
            print(f"    skipped one image ({e})")
            return False
        finally:
            img.close()  # release the decoded buffer now, not at GC's convenience

        if want_val:
            self.n_val += 1
        else:
            self.n_train += 1
        return True


# ─── Defactify_Image_Dataset ────────────────────────────────────────────────

_DEFACTIFY_GENERATORS = {
    1: "sd21",
    2: "sdxl",
    3: "sd3",
    4: "dalle3",
    5: "mj6",
}


def download_defactify(output_dir: Path, n_per_generator_train: int = 1500,
                        n_per_generator_val: int = 375, n_real_train: int = 1500,
                        n_real_val: int = 375, max_scan: int = 60000):
    """
    Rajarshi-Roy-research/Defactify_Image_Dataset (HF, CC BY 4.0).
    Populates ai_generated/ (5 generator-tagged sub-pools, prefix
    `defactify_<gen>_`) and real/ (prefix `defactify_coco_`).

    Streams rather than loading the full 96k/7.5GB at once — stops scanning
    once max_scan examples have been seen or every generator bucket has
    enough images, whichever comes first.
    """
    from datasets import load_dataset

    print("\nStreaming Defactify_Image_Dataset (HuggingFace, CC BY 4.0)...")
    ds = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset", split="train", streaming=True)

    buckets = {
        gen: StreamingBucket(output_dir, "ai_generated", f"defactify_{gen}_",
                              n_per_generator_train, n_per_generator_val)
        for gen in _DEFACTIFY_GENERATORS.values()
    }
    real_bucket = StreamingBucket(output_dir, "real", "defactify_coco_", n_real_train, n_real_val)

    for i, ex in enumerate(ds):
        label_a, label_b = ex["Label_A"], ex["Label_B"]
        if label_a == 0 and not real_bucket.is_full():
            real_bucket.add(ex["Image"])
        elif label_a == 1:
            gen = _DEFACTIFY_GENERATORS.get(label_b)
            if gen and not buckets[gen].is_full():
                buckets[gen].add(ex["Image"])

        if i % 2000 == 0:
            print(f"  scanned {i} — real={len(real_bucket)}, "
                  + ", ".join(f"{g}={len(b)}" for g, b in buckets.items()), flush=True)

        if real_bucket.is_full() and all(b.is_full() for b in buckets.values()):
            break
        if i >= max_scan:
            print(f"  hit max_scan={max_scan} before filling every bucket — "
                  f"taking what was found (some generators may be under-represented "
                  f"in this dataset's collection, not a bug in this script).")
            break

    print(f"  real (defactify_coco_): {real_bucket.n_train} train, {real_bucket.n_val} val")
    for gen, b in buckets.items():
        if len(b) == 0:
            print(f"  WARNING: 0 images found for generator '{gen}' — check Label_B mapping "
                  f"still matches the dataset (verified 2026-08-10, may drift on updates).")
        else:
            print(f"  {gen} (defactify_{gen}_): {b.n_train} train, {b.n_val} val")

    print("  Defactify organized.")


# ─── OpenFake (filtered, streamed — never bulk-downloaded) ─────────────────

# Exact `model` column values to pull — confirmed present via direct schema
# inspection (see proposal §5). Verify this list still matches on re-run;
# OpenFake is actively growing and model-string spellings are not guaranteed
# stable across dataset versions.
OPENFAKE_TARGET_MODELS = {
    "flux-1.1-dev": "flux11dev",
    "flux-1.1-schnell": "flux11schnell",
    "flux-2-dev": "flux2dev",
    "sdxl": "sdxl",
    "dalle-3": "dalle3",
    "midjourney-6": "mj6",
    "midjourney-7": "mj7",
}


def download_openfake(output_dir: Path, n_per_generator_train: int = 1200,
                       n_per_generator_val: int = 300, split: str = "test",
                       max_scan: int = 200000, models: dict | None = None):
    """
    ComplexDataLab/OpenFake (HF). core/{split} — streamed and filtered on the
    `model` column, capped per generator. NEVER loads the dataset's full
    3.44TB — only the shards needed to fill the requested sample are touched,
    and only matching rows are decoded to an image.

    Default split="test" deliberately — OpenFake's own test split already
    pairs unseen generators with unseen real-image sources (DOCCI/ImageNet),
    so pulling from it gives images consistent with THIS project's own LOGO
    use case (holding a generator out of training) rather than accidentally
    sampling from images that were themselves used to train something.

    NOTE: if OPENFAKE_TARGET_MODELS' exact strings have drifted (dataset is
    actively growing), run with --list-models first to see what's actually
    present before assuming 0-image buckets are this script's fault.
    """
    from datasets import load_dataset
    from huggingface_hub import HfApi

    models = models or OPENFAKE_TARGET_MODELS

    api = HfApi()
    info = api.dataset_info("ComplexDataLab/OpenFake")
    shard_files = sorted(s.rfilename for s in info.siblings
                          if s.rfilename.startswith(f"core/{split}-"))
    if not shard_files:
        print(f"  ERROR: no shards found for split='{split}' — check the dataset's file "
              f"layout hasn't changed (was core/{{train,test}}-*.parquet as of 2026-08-10).")
        return

    print(f"\nStreaming OpenFake core/{split} ({len(shard_files)} shards, filtering to "
          f"{len(models)} target generators)...")
    ds = load_dataset("ComplexDataLab/OpenFake", data_files={split: shard_files},
                       split=split, streaming=True)

    buckets = {
        prefix: StreamingBucket(output_dir, "ai_generated", f"openfake_{prefix}_",
                                 n_per_generator_train, n_per_generator_val)
        for prefix in models.values()
    }

    for i, ex in enumerate(ds):
        model = ex.get("model")
        if model in models and ex.get("label") == "fake":
            b = buckets[models[model]]
            if not b.is_full():
                b.add(ex["image"])

        if i % 2000 == 0:
            print(f"  scanned {i} — " + ", ".join(f"{p}={len(b)}" for p, b in buckets.items()), flush=True)

        if all(b.is_full() for b in buckets.values()):
            break
        if i >= max_scan:
            print(f"  hit max_scan={max_scan} — some generators may be under-filled if "
                  f"they're rare in this split, or if their model-string has drifted "
                  f"(run --list-models to check).")
            break

    for prefix, b in buckets.items():
        if len(b) == 0:
            print(f"  WARNING: 0 images found for '{prefix}' — check OPENFAKE_TARGET_MODELS "
                  f"still matches actual `model` values (run --list-models).")
        else:
            print(f"  {prefix}: {b.n_train} train, {b.n_val} val")

    print("  OpenFake filtered sample organized.")


def list_openfake_models(split: str = "test", scan: int = 20000):
    """Diagnostic: print the actual `model` column values seen, with counts —
    run this if download_openfake reports 0-image buckets, to see what
    strings are actually present instead of guessing."""
    from datasets import load_dataset
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.dataset_info("ComplexDataLab/OpenFake")
    shard_files = sorted(s.rfilename for s in info.siblings
                          if s.rfilename.startswith(f"core/{split}-"))[:3]
    ds = load_dataset("ComplexDataLab/OpenFake", data_files={split: shard_files},
                       split=split, streaming=True)

    counts: dict[str, int] = {}
    for i, ex in enumerate(ds):
        counts[ex["model"]] = counts.get(ex["model"], 0) + 1
        if i >= scan:
            break

    for model, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {count:6d}  {model}")


def verify(output_dir: Path):
    print("\n─── Research Data Summary (defactify_ / openfake_ prefixes) ─────")
    for split in ["train", "val"]:
        for cls in ["real", "ai_generated"]:
            d = output_dir / split / cls
            if not d.exists():
                continue
            new_files = [f for f in d.iterdir()
                         if f.name.startswith(("defactify_", "openfake_")) and f.suffix.lower() in EXTENSIONS]
            if new_files:
                by_prefix: dict[str, int] = {}
                for f in new_files:
                    tag = "_".join(f.name.split("_")[:2])
                    by_prefix[tag] = by_prefix.get(tag, 0) + 1
                print(f"  {split}/{cls}: " + ", ".join(f"{k}={v}" for k, v in sorted(by_prefix.items())))
    print("───────────────────────────────────────────────────────────────")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["defactify", "openfake", "list-openfake-models", "verify"],
                         default="defactify")
    parser.add_argument("--output", default="../data")
    parser.add_argument("--n-train", type=int, default=1500,
                         help="Images per generator for train (defactify) / per model (openfake).")
    parser.add_argument("--n-val", type=int, default=375)
    parser.add_argument("--openfake-split", default="test",
                         help="OpenFake split to pull from — 'test' by default (see docstring).")
    parser.add_argument("--max-scan", type=int, default=200000,
                         help="Safety cap on rows scanned while streaming, so a rare generator "
                              "doesn't turn into an unbounded download.")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if args.source == "defactify":
        download_defactify(output, n_per_generator_train=args.n_train, n_per_generator_val=args.n_val,
                            n_real_train=args.n_train, n_real_val=args.n_val)
    elif args.source == "openfake":
        download_openfake(output, n_per_generator_train=args.n_train, n_per_generator_val=args.n_val,
                           split=args.openfake_split, max_scan=args.max_scan)
    elif args.source == "list-openfake-models":
        list_openfake_models(split=args.openfake_split)
        return
    elif args.source == "verify":
        verify(output)
        return

    verify(output)


if __name__ == "__main__":
    main()
