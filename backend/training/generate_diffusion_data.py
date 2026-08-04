"""
Generate diffusion-model face images to inject into the ai_generated class.

Why this exists
----------------
The 140k-faces dataset supplies every "real" and every StyleGAN "fake" image
in this project. Diffusion models (Stable Diffusion, Midjourney, DALL-E, ...)
were entirely absent from training, and this was measured to matter:

    Before (StyleGAN-only training), on 30 genuine Stable Diffusion faces
    never seen in training:
        0/30 correct, 25/30 confidently classified REAL, 5/30 abstained.
        Epistemic uncertainty (member disagreement) was near-zero on the
        wrong predictions — both ensemble members share the identical blind
        spot, so architecture diversity alone cannot catch it. Disagreement-
        based abstention only helps when at least one member has seen enough
        to disagree.

    After fine-tuning both members on a dataset with ~9% of the ai_generated
    class replaced by real Stable Diffusion output (800 train / 200 val
    added on top of the existing 8000 StyleGAN train images), on the SAME
    held-out 30 images:
        27/30 correct, 0/30 confidently wrong, 3/30 abstained.

Main-task metrics on the (now harder, diffusion-inclusive) val set barely
moved: 99.19% -> 98.69% accuracy, 0.9930 -> 0.9883 macro-F1. Robustness to
JPEG/blur/downscale perturbations, measured separately, did not regress
(the gap actually improved, 10.9% -> 6.4%, likely incidental to the extra
training exposure rather than a targeted effect).

Usage
-----
    pip install diffusers accelerate safetensors
    python training/generate_diffusion_data.py --data ../data --n-train 800 --n-val 200

Then fine-tune from the existing checkpoints (see train.py's --resume flag).
IMPORTANT: --resume now re-measures the baseline on the CURRENT val set before
deciding whether a fine-tuned epoch is "better" — comparing against the
checkpoint's stored (stale, pre-augmentation) val_macro_f1 silently discarded
a genuinely improved epoch once already. Always back up checkpoints before a
fine-tune run regardless; --resume writing back to its own source file is a
real footgun if a script bug (or a different one you introduce) papers over
protections like this one.
"""

import argparse
import os
import random
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

# runwayml/stable-diffusion-v1-5 has been unreliable/delisted at times; fall
# back through alternatives rather than hard-failing on one repo.
MODEL_CANDIDATES = [
    "stabilityai/stable-diffusion-2-1-base",
    "CompVis/stable-diffusion-v1-4",
    "runwayml/stable-diffusion-v1-5",
]

PROMPTS = [
    "professional headshot portrait of a person, studio lighting, photorealistic",
    "candid photo of a smiling person outdoors, natural lighting, photorealistic",
    "passport style photo of a person, neutral expression, plain background",
    "linkedin profile photo of a business professional, photorealistic",
    "close-up portrait photo of a person, soft lighting, high detail",
    "photo of a person's face, natural skin texture, realistic",
    "portrait of an elderly person, wrinkles, realistic skin, natural light",
    "portrait of a young adult, casual clothing, outdoor background, photo",
    "studio portrait of a person wearing glasses, photorealistic",
    "portrait photo of a person with curly hair, natural lighting",
    "portrait of a bearded man, photorealistic, daylight",
    "portrait of a woman with short hair, photorealistic, indoor lighting",
    "professional corporate headshot, neutral background, sharp focus",
    "portrait photo taken with a DSLR camera, shallow depth of field, person",
    "realistic photo of a person laughing, natural expression",
    "portrait of a person in winter clothing, outdoor, photorealistic",
    "id photo of a person, front facing, plain white background",
    "portrait of a person with freckles, natural skin, photorealistic",
    "portrait photo, warm lighting, person looking at camera, realistic",
]


def _load_pipeline():
    last_err = None
    for model_id in MODEL_CANDIDATES:
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16, safety_checker=None,
            )
            print(f"Loaded {model_id}")
            return pipe
        except Exception as e:
            last_err = e
            print(f"  {model_id} failed: {str(e)[:150]}")
    raise RuntimeError(f"No Stable Diffusion checkpoint could be loaded: {last_err}")


def generate(
    output_dir: Path,
    n_train: int = 800,
    n_val: int = 200,
    steps: int = 20,
    seed_offset: int = 1000,
    filename_prefix: str = "sd14",
):
    """
    Writes directly into <output_dir>/train/ai_generated and .../val/ai_generated
    with a distinct filename prefix, so it mixes into the existing class
    without touching dataset.py or requiring a new label.
    """
    output_dir = Path(output_dir)
    train_dir = output_dir / "train" / "ai_generated"
    val_dir = output_dir / "val" / "ai_generated"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    pipe = _load_pipeline().to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)

    rng = random.Random(42)
    total = n_train + n_val
    for i in range(total):
        split_dir = train_dir if i < n_train else val_dir
        idx_in_split = i if i < n_train else i - n_train
        prompt = rng.choice(PROMPTS)
        seed = seed_offset + i
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        image = pipe(
            prompt, num_inference_steps=steps, height=512, width=512, generator=generator,
        ).images[0]
        image = image.resize((256, 256))
        fname = f"{filename_prefix}_{idx_in_split:05d}.jpg"
        image.convert("RGB").save(split_dir / fname, quality=90)
        if (i + 1) % 25 == 0:
            print(f"{i+1}/{total} generated", flush=True)

    print(f"Done. {n_train} -> {train_dir}, {n_val} -> {val_dir}")


def generate_holdout_test_set(
    output_dir: Path, n: int = 30, seed_offset: int = 90000, steps: int = 20,
):
    """
    A genuinely held-out evaluation set: separate directory, seed range far
    from both the train (seed_offset..) and val ranges used by generate(),
    so it can honestly measure generalization rather than memorization.
    Not part of data/ — write it wherever you're keeping eval artifacts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = _load_pipeline().to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)

    rng = random.Random(7)
    for i in range(n):
        prompt = rng.choice(PROMPTS)
        generator = torch.Generator(device=pipe.device).manual_seed(seed_offset + i)
        image = pipe(prompt, num_inference_steps=steps, height=512, width=512, generator=generator).images[0]
        image.resize((256, 256)).convert("RGB").save(output_dir / f"holdout_{i:03d}.jpg", quality=90)
        print(f"{i+1}/{n} saved", flush=True)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data")
    parser.add_argument("--n-train", type=int, default=800)
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed-offset", type=int, default=1000)
    args = parser.parse_args()
    generate(args.data, n_train=args.n_train, n_val=args.n_val, steps=args.steps, seed_offset=args.seed_offset)
