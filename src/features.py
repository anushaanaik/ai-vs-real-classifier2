"""
features.py — Feature extraction utilities.

Provides:
  - EfficientNetB3 feature extractor (without classification head)
  - Batch feature extraction for downstream use (e.g. t-SNE, kNN)
  - Basic image-level statistics (mean, std, entropy) as lightweight features
"""
from __future__ import annotations

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, Model


# ─── Deep feature extractor ──────────────────────────────────────────────────

def build_feature_extractor(img_size: int = 224) -> Model:
    """
    Return an EfficientNetB3 model that outputs a 1536-d feature vector
    (GlobalAveragePooling over the last conv output).
    Weights are frozen ImageNet weights — no fine-tuning here.
    """
    base = EfficientNetB3(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    return Model(inputs, x, name="EffNetB3_features")


def extract_features(
    image_paths: list[str],
    extractor: Model | None = None,
    img_size: int = 224,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Extract deep features from a list of image file paths.

    Returns
    -------
    np.ndarray of shape (N, 1536)
    """
    if extractor is None:
        extractor = build_feature_extractor(img_size)

    from src.preprocess import preprocess_image

    all_features = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch = np.stack(
            [preprocess_image(p, img_size=img_size) for p in batch_paths], axis=0
        )
        feats = extractor.predict(batch, verbose=0)
        all_features.append(feats)

    return np.concatenate(all_features, axis=0)


# ─── Lightweight statistical features ───────────────────────────────────────

def _image_entropy(arr: np.ndarray) -> float:
    """Shannon entropy of pixel value histogram (grayscale)."""
    gray = arr.mean(axis=-1).astype(np.uint8).flatten()
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def extract_stat_features(image_path: str, img_size: int = 224) -> np.ndarray:
    """
    Fast, model-free feature vector for a single image.

    Features (9-dim):
      - Per-channel mean (R, G, B)
      - Per-channel std  (R, G, B)
      - Overall entropy
      - Aspect ratio
      - Relative saturation (mean S in HSV)
    """
    img = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32) / 255.0

    means = arr.mean(axis=(0, 1))          # (3,)
    stds  = arr.std(axis=(0, 1))           # (3,)
    entropy = _image_entropy((arr * 255).astype(np.uint8))

    # Saturation from HSV
    hsv = np.array(img.convert("HSV"), dtype=np.float32)
    saturation = hsv[:, :, 1].mean() / 255.0

    orig = Image.open(image_path)
    w, h = orig.size
    aspect = w / h if h != 0 else 1.0

    return np.concatenate([means, stds, [entropy, saturation, aspect]])
