"""
train.py — Two-phase EfficientNetB3 training pipeline.

Usage:
  python src/train.py --zip data/ai-generated-images-vs-real-images.zip

Options:
  --zip          Path to the dataset zip (read-only, never modified)
  --dataset_root Directory to extract images into   [default: data/dataset]
  --model_out    Output model path                  [default: models/model_v1.keras]
  --config       Path to config.yaml                [default: config.yaml]
  --epochs_p1    Phase 1 epochs override
  --epochs_p2    Phase 2 epochs override
  --batch_size   Batch size override
  --gpu          Enable GPU memory growth (auto-detected)
"""
import argparse
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam

from src.data_loader import load_dataset
from src.preprocess import get_phase1_generators, get_phase2_generators
from src.utils import get_logger, load_config, save_class_config, set_seed

logger = get_logger()


# ─── Model builder ──────────────────────────────────────────────────────────

def build_model(img_size: int = 224) -> tuple[Model, object]:
    """
    Build EfficientNetB3 classifier with a strong custom head.
    Returns (full_model, base_model) so Phase 2 can unfreeze the base.
    """
    base = EfficientNetB3(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="EffNetB3_AIvsReal")
    return model, base


def compile_model(model: Model, lr: float, label_smoothing: float = 0.05) -> None:
    model.compile(
        optimizer=Adam(learning_rate=lr, clipnorm=1.0),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


# ─── Training phases ─────────────────────────────────────────────────────────

def train_phase1(
    model: Model,
    train_gen,
    val_gen,
    epochs: int = 12,
    ckpt_path: str = "models/ckpt_p1.keras",
) -> object:
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            min_delta=0.001,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
        ),
        ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]

    logger.info("Phase 1: training head only (base frozen) — %d epochs max", epochs)
    history = model.fit(
        train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks
    )
    best_acc = max(history.history["val_accuracy"])
    best_auc = max(history.history["val_auc"])
    logger.info("Phase 1 done | val_acc=%.4f | AUC=%.4f", best_acc, best_auc)
    return history


def train_phase2(
    model: Model,
    base,
    train_gen,
    val_gen,
    epochs: int = 20,
    unfreeze_top: int = 50,
    ckpt_p1: str = "models/ckpt_p1.keras",
    ckpt_path: str = "models/ckpt_best.keras",
) -> object:
    # Load best Phase 1 weights
    if os.path.exists(ckpt_p1):
        model.load_weights(ckpt_p1)
        logger.info("Loaded best Phase 1 weights from %s", ckpt_p1)

    # Unfreeze top layers
    base.trainable = True
    for layer in base.layers[:-unfreeze_top]:
        layer.trainable = False

    n_trainable = sum(1 for l in base.layers if l.trainable)
    n_frozen = sum(1 for l in base.layers if not l.trainable)
    logger.info("Phase 2 unfreeze: trainable=%d, frozen=%d", n_trainable, n_frozen)

    compile_model(model, lr=2e-5, label_smoothing=0.03)

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    callbacks = [
        EarlyStopping(
            monitor="val_auc", patience=5, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1
        ),
        ModelCheckpoint(
            ckpt_path, monitor="val_auc", save_best_only=True, verbose=1
        ),
    ]

    logger.info(
        "Phase 2: fine-tuning top %d layers (LR=2e-5) — %d epochs max",
        unfreeze_top,
        epochs,
    )
    history = model.fit(
        train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks
    )
    best_acc = max(history.history["val_accuracy"])
    best_auc = max(history.history["val_auc"])
    logger.info("Phase 2 done | val_acc=%.4f | AUC=%.4f", best_acc, best_auc)
    return history


# ─── Entry point ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train AI vs Real classifier")
    p.add_argument("--zip", required=True, help="Path to dataset zip")
    p.add_argument("--dataset_root", default="data/dataset")
    p.add_argument("--model_out", default="models/model_v1.keras")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--epochs_p1", type=int, default=None)
    p.add_argument("--epochs_p2", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Apply overrides
    seed = cfg["training"]["seed"]
    img_size = cfg["model"]["img_size"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    epochs_p1 = args.epochs_p1 or cfg["training"]["phase1"]["epochs"]
    epochs_p2 = args.epochs_p2 or cfg["training"]["phase2"]["epochs"]
    lr_p1 = cfg["training"]["phase1"]["learning_rate"]
    unfreeze_top = cfg["training"]["phase2"]["unfreeze_top_layers"]
    train_frac = cfg["data"]["train_frac"]
    val_frac = cfg["data"]["val_frac"]

    set_seed(seed)

    # GPU config
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logger.info(
        "TensorFlow %s | %s", tf.__version__, "GPU" if gpus else "CPU"
    )

    # 1. Data
    load_dataset(
        args.zip,
        dataset_root=args.dataset_root,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
    )

    # 2. Generators
    train_gen_p1, val_gen = get_phase1_generators(
        args.dataset_root, img_size=img_size, batch_size=batch_size, seed=seed
    )
    train_gen_p2, _ = get_phase2_generators(
        args.dataset_root, img_size=img_size, batch_size=batch_size, seed=seed
    )

    # 3. Build
    model, base = build_model(img_size=img_size)
    compile_model(model, lr=lr_p1)
    model.summary(print_fn=logger.info)

    # 4. Phase 1
    history_p1 = train_phase1(
        model, train_gen_p1, val_gen, epochs=epochs_p1,
        ckpt_path="models/ckpt_p1.keras"
    )

    # 5. Phase 2
    history_p2 = train_phase2(
        model, base, train_gen_p2, val_gen,
        epochs=epochs_p2,
        unfreeze_top=unfreeze_top,
        ckpt_p1="models/ckpt_p1.keras",
        ckpt_path="models/ckpt_best.keras",
    )

    # 6. Save final model
    if os.path.exists("models/ckpt_best.keras"):
        model.load_weights("models/ckpt_best.keras")

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    model.save(args.model_out)
    logger.info("Model saved → %s", args.model_out)

    # 7. Save class config
    best_p1_acc = max(history_p1.history["val_accuracy"])
    best_p2_acc = max(history_p2.history["val_accuracy"])
    best_p2_auc = max(history_p2.history["val_auc"])

    save_class_config(
        "models/class_names.json",
        val_accuracy=best_p2_acc,
        val_auc=best_p2_auc,
        total_images=train_gen_p1.samples + val_gen.samples,
        train_frac=train_frac,
        val_frac=val_frac,
    )
    logger.info("class_names.json saved.")
    logger.info("Training complete. Best val_acc=%.4f | best AUC=%.4f", best_p2_acc, best_p2_auc)


if __name__ == "__main__":
    main()
