"""
utils.py — Logging, seeding, config loading, and miscellaneous helpers.
"""
import os
import json
import random
import logging
import yaml
import numpy as np


# ─── Logging ────────────────────────────────────────────────────────────────

def get_logger(name: str = "ai_vs_real", log_file: str = "logs/app.log") -> logging.Logger:
    """Return a logger that writes to both console and file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ─── Config ─────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    """Load YAML config and return as a plain dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─── Reproducibility ────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─── Class config helpers ────────────────────────────────────────────────────

def save_class_config(
    path: str,
    val_accuracy: float = 0.0,
    val_auc: float = 0.0,
    test_accuracy: float | None = None,
    test_auc: float | None = None,
    total_images: int = 0,
    train_frac: float = 0.75,
    val_frac: float = 0.15,
) -> None:
    """Write class_names.json with metadata used by predict.py."""
    config = {
        "class_names": ["ai", "real"],
        "class_indices": {"ai": 0, "real": 1},
        "label_map": {"0": "AI", "1": "REAL"},
        "confidence_threshold": 0.60,
        "model_architecture": "EfficientNetB3",
        "input_size": 224,
        "tta_steps_default": 8,
        "final_metrics": {
            "val_accuracy": round(val_accuracy, 4),
            "val_auc": round(val_auc, 4),
            "test_accuracy": round(test_accuracy, 4) if test_accuracy is not None else None,
            "test_auc": round(test_auc, 4) if test_auc is not None else None,
        },
        "dataset": {
            "source": "Kaggle — tristanzhang32/ai-generated-images-vs-real-images",
            "total_images": total_images,
            "train_frac": train_frac,
            "val_frac": val_frac,
            "split_strategy": "stratified split from zipped dataset",
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def load_class_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ─── Misc ────────────────────────────────────────────────────────────────────

def count_images(directory: str, exts: tuple = (".jpg", ".jpeg", ".png", ".webp")) -> int:
    total = 0
    for root, _, files in os.walk(directory):
        total += sum(1 for f in files if f.lower().endswith(exts))
    return total


def human_size(path: str) -> str:
    size = os.path.getsize(path)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
