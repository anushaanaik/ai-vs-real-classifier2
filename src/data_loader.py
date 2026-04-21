"""
data_loader.py — Dataset loading, balancing, and train/val/test splitting.

Supports:
  - Loading from a zip file (read-only, never modified)
  - Auto-detecting 'ai' and 'real' class folders
  - Balanced splitting with configurable fractions
"""
import io
import os
import random
import zipfile

from PIL import Image, ImageFile
from tqdm import tqdm

from src.utils import get_logger

ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate slightly corrupt images

logger = get_logger()

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


# ─── Helpers ────────────────────────────────────────────────────────────────

def is_image(path: str) -> bool:
    return path.lower().endswith(IMG_EXTS)


def _detect_class(zip_path: str) -> str | None:
    """Infer 'ai' or 'real' from a zip-internal path."""
    parts = zip_path.replace("\\", "/").split("/")
    for part in parts[:-1]:
        lp = part.lower()
        if lp == "real":
            return "real"
        if lp in ("fake", "ai", "artificial", "generated", "synthetic"):
            return "ai"
    # fall back to filename keywords
    basename = parts[-1].lower()
    if "real" in basename:
        return "real"
    if any(kw in basename for kw in ("fake", "ai_gen", "artificial")):
        return "ai"
    return None


# ─── Public API ─────────────────────────────────────────────────────────────

def scan_zip(zip_path: str) -> dict[str, list[str]]:
    """
    Scan the zip (no extraction) and return
    {'ai': [...], 'real': [...]} of internal paths.
    """
    logger.info("Scanning zip: %s", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_entries = zf.namelist()

    image_entries = [
        e for e in all_entries
        if is_image(e) and not e.startswith("__MACOSX")
    ]
    logger.info("Total image entries in zip: %d", len(image_entries))

    categorized: dict[str, list[str]] = {"ai": [], "real": []}
    for fpath in image_entries:
        cls = _detect_class(fpath)
        if cls:
            categorized[cls].append(fpath)

    for cls, files in categorized.items():
        logger.info("  %s: %d images", cls, len(files))

    if not categorized["ai"] and not categorized["real"]:
        raise RuntimeError(
            "Could not auto-detect 'ai' or 'real' class folders in the zip.\n"
            "Update _detect_class() to match your zip structure."
        )
    return categorized


def make_splits(
    categorized: dict[str, list[str]],
    train_frac: float = 0.75,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, dict[str, list[str]]]:
    """Balance and split into train / val / test."""
    random.seed(seed)
    min_count = min(len(v) for v in categorized.values())
    logger.info("Balancing to %d images per class", min_count)

    balanced = {}
    for cls in categorized:
        shuffled = list(categorized[cls])
        random.shuffle(shuffled)
        balanced[cls] = shuffled[:min_count]

    n_train = int(min_count * train_frac)
    n_val = int(min_count * val_frac)

    splits: dict[str, dict[str, list[str]]] = {
        "train": {cls: balanced[cls][:n_train] for cls in balanced},
        "val":   {cls: balanced[cls][n_train:n_train + n_val] for cls in balanced},
        "test":  {cls: balanced[cls][n_train + n_val:] for cls in balanced},
    }

    for split, classes in splits.items():
        total = sum(len(v) for v in classes.values())
        logger.info("  %s: %d total", split, total)

    return splits


def extract_splits(
    zip_path: str,
    splits: dict[str, dict[str, list[str]]],
    dataset_root: str = "data/dataset",
) -> None:
    """Extract only the required files from the zip into dataset_root."""
    # Check if already extracted
    already_done = all(
        os.path.exists(f"{dataset_root}/{split}/{cls}")
        and len(os.listdir(f"{dataset_root}/{split}/{cls}")) > 10
        for split in splits
        for cls in ["ai", "real"]
    )
    if already_done:
        logger.info("Dataset already extracted — skipping.")
        return

    # Create directories
    for split in splits:
        for cls in ["ai", "real"]:
            os.makedirs(f"{dataset_root}/{split}/{cls}", exist_ok=True)

    logger.info("Extracting from zip → %s", dataset_root)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for split, classes in splits.items():
            for cls, file_list in classes.items():
                dst_dir = f"{dataset_root}/{split}/{cls}"
                for i, zpath in enumerate(
                    tqdm(file_list, desc=f"{split}/{cls}", unit="img")
                ):
                    dst = os.path.join(dst_dir, f"{i:06d}_{os.path.basename(zpath)}")
                    if os.path.exists(dst):
                        continue
                    try:
                        with zf.open(zpath) as src:
                            data = src.read()
                        Image.open(io.BytesIO(data)).verify()
                        with open(dst, "wb") as f:
                            f.write(data)
                    except Exception:
                        pass  # skip corrupt files silently

    logger.info("Extraction complete.")


def load_dataset(
    zip_path: str,
    dataset_root: str = "data/dataset",
    train_frac: float = 0.75,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, dict[str, list[str]]]:
    """
    Full pipeline: scan → split → extract.
    Returns the split dict (for generator creation in train.py).
    """
    categorized = scan_zip(zip_path)
    splits = make_splits(categorized, train_frac=train_frac, val_frac=val_frac, seed=seed)
    extract_splits(zip_path, splits, dataset_root=dataset_root)
    return splits
