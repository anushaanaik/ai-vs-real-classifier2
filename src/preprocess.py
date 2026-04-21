"""
preprocess.py — Image preprocessing and augmentation pipelines.

Provides:
  - Real-world augmentation functions (JPEG compression, noise, blur, resize artifacts)
  - Phase 1 and Phase 2 Keras ImageDataGenerators
  - Single-image preprocessing for inference
"""
import io
import random

import numpy as np
from PIL import Image, ImageFile, ImageFilter

from tensorflow.keras.preprocessing.image import ImageDataGenerator

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42


# ─── Real-world augmentation functions ──────────────────────────────────────

def apply_jpeg_compression(img: np.ndarray, prob: float = 0.6) -> np.ndarray:
    """Simulate social-media JPEG recompression (quality 50–92)."""
    if random.random() < prob:
        quality = random.randint(50, 92)
        pil = Image.fromarray(img.astype(np.uint8))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return np.array(Image.open(buf)).astype(np.float32)
    return img


def apply_gaussian_noise(img: np.ndarray, prob: float = 0.4) -> np.ndarray:
    """Camera sensor noise."""
    if random.random() < prob:
        sigma = random.uniform(0.5, 8.0)
        noise = np.random.normal(0, sigma, img.shape)
        return np.clip(img + noise, 0, 255).astype(np.float32)
    return img


def apply_resize_artifact(
    img: np.ndarray, prob: float = 0.3, img_size: int = IMG_SIZE
) -> np.ndarray:
    """Downscale-then-upscale (common in social-media pipelines)."""
    if random.random() < prob:
        pil = Image.fromarray(img.astype(np.uint8))
        small = random.randint(96, img_size - 1)
        pil = pil.resize((small, small), Image.BILINEAR).resize(
            (img_size, img_size), Image.BILINEAR
        )
        return np.array(pil).astype(np.float32)
    return img


def apply_blur(img: np.ndarray, prob: float = 0.2) -> np.ndarray:
    """Slight motion/focus blur."""
    if random.random() < prob:
        pil = Image.fromarray(img.astype(np.uint8))
        pil = pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.5)))
        return np.array(pil).astype(np.float32)
    return img


def real_world_augmentation(img: np.ndarray) -> np.ndarray:
    """Full real-world augmentation pipeline (used as preprocessing_function)."""
    img = apply_jpeg_compression(img)
    img = apply_gaussian_noise(img)
    img = apply_resize_artifact(img)
    img = apply_blur(img)
    return img


# ─── Keras generators ────────────────────────────────────────────────────────

def get_phase1_generators(
    dataset_root: str,
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
):
    """Lightweight generators for Phase 1 (head-only warmup)."""
    datagen = ImageDataGenerator(rescale=1.0 / 255, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = datagen.flow_from_directory(
        f"{dataset_root}/train",
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        classes=["ai", "real"],
        shuffle=True,
        seed=seed,
    )
    val_gen = val_datagen.flow_from_directory(
        f"{dataset_root}/val",
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        classes=["ai", "real"],
        shuffle=False,
    )
    return train_gen, val_gen


def get_phase2_generators(
    dataset_root: str,
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
):
    """Heavy real-world augmentation generators for Phase 2 fine-tuning."""
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.65, 1.35],
        fill_mode="reflect",
        preprocessing_function=real_world_augmentation,
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = datagen.flow_from_directory(
        f"{dataset_root}/train",
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        classes=["ai", "real"],
        shuffle=True,
        seed=seed,
    )
    val_gen = val_datagen.flow_from_directory(
        f"{dataset_root}/val",
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        classes=["ai", "real"],
        shuffle=False,
    )
    return train_gen, val_gen


def get_test_generator(
    dataset_root: str,
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
):
    """Clean generator for test-set evaluation (no augmentation)."""
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    return datagen.flow_from_directory(
        f"{dataset_root}/test",
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        classes=["ai", "real"],
        shuffle=False,
    )


# ─── Single-image preprocessing (for inference) ──────────────────────────────

def preprocess_image(
    image_source,
    img_size: int = IMG_SIZE,
    apply_aug: bool = False,
) -> np.ndarray:
    """
    Load and preprocess a single image for inference.

    Parameters
    ----------
    image_source : str | PIL.Image
        File path or PIL Image object.
    img_size : int
        Target size (default 224).
    apply_aug : bool
        Apply lightweight TTA augmentation (random flip + brightness).

    Returns
    -------
    np.ndarray of shape (img_size, img_size, 3), dtype float32, values in [0, 1].
    """
    if isinstance(image_source, str):
        img = Image.open(image_source).convert("RGB")
    else:
        img = image_source.copy().convert("RGB")

    img = img.resize((img_size, img_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)

    if apply_aug:
        # Random horizontal flip
        if random.random() > 0.5:
            arr = arr[:, ::-1, :]
        # Slight brightness jitter
        if random.random() > 0.7:
            arr = np.clip(arr * random.uniform(0.9, 1.1), 0, 255)

    return arr / 255.0
