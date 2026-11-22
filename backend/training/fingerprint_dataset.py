"""
Dataset for the Generation Fingerprint module — proposal §4.2, §7.

Unlike training/dataset.py's FakeImageDataset (raw RGB tensor + class label),
this yields the 6 multi-representation maps (models/fingerprint/representations.py)
PLUS a generator/domain id, since the domain-adversarial objective needs to
know which "domain" (generator family, or real sub-source) each image came
from — that's the signal the GRL regularizer is trained to make z invariant
to (proposal §4.2).

Extends the SAME filename-prefix convention training/dataset.py already uses
(casia_au_, sd14_, indoor67_, textocr_) rather than inventing a parallel one,
plus the new prefixes download_research_data.py writes
(defactify_<gen>_, openfake_<gen>_).
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.fingerprint.representations import extract_representations, REPRESENTATION_NAMES
from training.dataset import LABEL_MAP, EXTENSIONS

FP_IMG_SIZE = 128  # matches the fingerprint encoders' scale (encoder.py) —
                    # deliberately smaller than the 380px CNN ensemble members;
                    # these are lightweight per-representation encoders, not
                    # one large backbone, and 6 of them run per image.

# Canonical generator-family id. Real images get their own ids too (proposal
# §4.2: "REAL images get their own 'domain' too, so the regularizer doesn't
# trivially collapse onto only the fake side"). Extend this dict as new
# sources are added — do not invent a second parallel mapping elsewhere.
DOMAIN_IDS: dict[str, int] = {
    # AI_GENERATED domains
    "stylegan": 0,       # untagged ai_generated/ files — 140k-faces StyleGAN split
    "sd14": 1,            # sd14_ prefix (generate_diffusion_data.py)
    "defactify_sd21": 2,
    "defactify_sdxl": 3,
    "defactify_sd3": 4,
    "defactify_dalle3": 5,
    "defactify_mj6": 6,
    "openfake_flux11dev": 7,
    "openfake_flux11schnell": 8,
    "openfake_flux2dev": 9,
    "openfake_sdxl": 3,   # same family as defactify_sdxl — merged, not a new id
    "openfake_dalle3": 5,
    "openfake_mj6": 6,
    "openfake_mj7": 10,
    # REAL sub-domains (kept coarser — the sampler already balances these at
    # a finer grain via dataset.py's category machinery; the domain head here
    # only needs "which acquisition pipeline", not the full category split)
    "real_face": 11,
    "real_casia_au": 12,
    "real_indoor67": 13,
    "real_textocr": 14,
    "defactify_coco": 15,
}
N_DOMAINS = len(set(DOMAIN_IDS.values()))


def domain_of(path: Path, class_name: str) -> str:
    """Same spirit as evaluate_by_category.py's category_of — parses the
    filename-prefix convention into a domain key. Returns the domain KEY
    (string), look up DOMAIN_IDS[key] for the integer id."""
    name = path.name
    if class_name == "ai_generated":
        if name.startswith("sd14_"):
            return "sd14"
        for prefix in ("defactify_", "openfake_"):
            if name.startswith(prefix):
                # e.g. "defactify_sdxl_00012.jpg" -> "defactify_sdxl"
                parts = name.split("_")
                return f"{parts[0]}_{parts[1]}"
        return "stylegan"
    if class_name == "real":
        if name.startswith("casia_au_"):
            return "real_casia_au"
        if name.startswith("indoor67_"):
            return "real_indoor67"
        if name.startswith("textocr_"):
            return "real_textocr"
        if name.startswith("defactify_coco_"):
            return "defactify_coco"
        return "real_face"
    # edited / ai_edited — not yet domain-tagged (proposal §5's AI_EDITED
    # sourcing, DEFACTO + local inpainting, isn't wired in yet). Falls back
    # to a single shared domain so training doesn't crash before that lands.
    return "edited_untagged"


DOMAIN_IDS.setdefault("edited_untagged", N_DOMAINS)
N_DOMAINS = len(set(DOMAIN_IDS.values()))


class FingerprintDataset(Dataset):
    def __init__(self, root: str, split: str = "train",
                 exclude_domains: tuple[str, ...] = (), img_size: int = FP_IMG_SIZE):
        """
        exclude_domains: domain KEYS (not ids — see DOMAIN_IDS) to drop
        entirely, for the LOGO protocol — e.g. exclude_domains=("defactify_flux",)
        Mirrors FakeImageDataset's exclude_prefixes but keyed on the parsed
        domain rather than a raw filename prefix, since one domain can map to
        multiple prefixes (openfake_sdxl_ and defactify_sdxl_ are the same
        domain by design — see DOMAIN_IDS).
        """
        self.img_size = img_size
        base = Path(root) / split
        self.items: list[tuple[Path, int, int, str]] = []  # (path, task_label, domain_id, domain_key)
        key_counts: dict[str, int] = {}

        for class_name, task_label in LABEL_MAP.items():
            class_dir = base / class_name
            if not class_dir.exists():
                continue
            for p in class_dir.rglob("*"):
                if p.suffix.lower() not in EXTENSIONS:
                    continue
                domain_key = domain_of(p, class_name)
                if domain_key in exclude_domains:
                    continue
                domain_id = DOMAIN_IDS.get(domain_key, N_DOMAINS - 1)
                self.items.append((p, task_label, domain_id, domain_key))
                key_counts[domain_key] = key_counts.get(domain_key, 0) + 1

        if not self.items:
            raise FileNotFoundError(f"No images found under {base}")

        # Reported by domain KEY (what's actually on disk), not by re-deriving a
        # name from the integer id — several keys deliberately share one id
        # (proposal §5's "same family, different source dataset" merges, e.g.
        # defactify_sdxl and openfake_sdxl), so id -> name is not 1:1 and picking
        # one back would silently mislabel whichever source wasn't actually present.
        print(f"[fingerprint/{split}] {len(self.items)} images across {len(key_counts)} domains: "
              + ", ".join(f"{k}={n}" for k, n in sorted(key_counts.items())))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, task_label, domain_id, domain_key = self.items[idx]
        try:
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
        except Exception:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        reps = extract_representations(img)
        reps_t = {name: torch.from_numpy(arr) for name, arr in reps.items()}
        return reps_t, torch.tensor(task_label, dtype=torch.long), torch.tensor(domain_id, dtype=torch.long), domain_key


def collate_fingerprint(batch):
    """Custom collate — batch is a list of (reps_dict, task_label, domain_id,
    domain_key); default_collate can't stack a list of dicts on its own, and
    domain_key (a plain string) is passed through as a list rather than
    stacked into a tensor.

    domain_key is carried alongside the integer domain_id specifically so
    downstream code (evaluate_fingerprint_logo.py) can report results by the
    TRUE string key. Several keys deliberately share one domain_id (proposal
    §5's cross-dataset generator merges, e.g. defactify_sdxl and openfake_sdxl
    both train the same domain-invariance signal) — reversing id -> a single
    name is lossy and silently mislabeled a held-out LOGO domain as a
    different one that was never even downloaded, producing a fabricated
    0.0 LOGO-accuracy result. Found and fixed this session; keeping the key
    as a first-class field here instead of ever reconstructing it is the
    actual fix, not just a patched symptom."""
    reps_batched = {name: torch.stack([b[0][name] for b in batch]) for name in REPRESENTATION_NAMES}
    task_labels = torch.stack([b[1] for b in batch])
    domain_ids = torch.stack([b[2] for b in batch])
    domain_keys = [b[3] for b in batch]
    return reps_batched, task_labels, domain_ids, domain_keys
