"""
pipeline.py — End-to-end orchestration of the AI vs Real classifier.

Runs: data loading → preprocessing → training → evaluation → model export.

Usage:
  python pipeline/pipeline.py --zip data/ai-generated-images-vs-real-images.zip
"""
import argparse
import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import load_dataset
from src.preprocess import (
    get_phase1_generators,
    get_phase2_generators,
    get_test_generator,
)
from src.train import build_model, compile_model, train_phase1, train_phase2
from src.evaluate import (
    evaluate_generator,
    plot_confusion_matrix,
    plot_roc_curve,
    save_report,
)
from src.utils import get_logger, load_config, save_class_config, set_seed

logger = get_logger()


def run_pipeline(
    zip_path: str,
    dataset_root: str = "data/dataset",
    model_out: str = "models/model_v1.keras",
    config_path: str = "config.yaml",
    skip_training: bool = False,
) -> None:
    """
    Full end-to-end pipeline.

    Parameters
    ----------
    zip_path : str
        Path to dataset zip.
    dataset_root : str
        Where to extract images.
    model_out : str
        Where to save the final model.
    config_path : str
        YAML config path.
    skip_training : bool
        If True, skip to evaluation (model_out must already exist).
    """
    cfg = load_config(config_path)
    seed = cfg["training"]["seed"]
    img_size = cfg["model"]["img_size"]
    batch_size = cfg["training"]["batch_size"]
    epochs_p1 = cfg["training"]["phase1"]["epochs"]
    epochs_p2 = cfg["training"]["phase2"]["epochs"]
    lr_p1 = cfg["training"]["phase1"]["learning_rate"]
    unfreeze_top = cfg["training"]["phase2"]["unfreeze_top_layers"]
    train_frac = cfg["data"]["train_frac"]
    val_frac = cfg["data"]["val_frac"]

    set_seed(seed)

    # ── Step 1: Data ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1 — Data loading & extraction")
    logger.info("=" * 60)
    load_dataset(
        zip_path,
        dataset_root=dataset_root,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
    )

    # ── Step 2: Generators ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2 — Building data generators")
    logger.info("=" * 60)
    train_gen_p1, val_gen = get_phase1_generators(
        dataset_root, img_size=img_size, batch_size=batch_size, seed=seed
    )
    train_gen_p2, _ = get_phase2_generators(
        dataset_root, img_size=img_size, batch_size=batch_size, seed=seed
    )
    test_gen = get_test_generator(dataset_root, img_size=img_size, batch_size=batch_size)

    if not skip_training:
        # ── Step 3: Build & compile ───────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 3 — Building model")
        logger.info("=" * 60)
        model, base = build_model(img_size=img_size)
        compile_model(model, lr=lr_p1)

        # ── Step 4: Phase 1 ───────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 4 — Phase 1: Head training")
        logger.info("=" * 60)
        history_p1 = train_phase1(
            model, train_gen_p1, val_gen,
            epochs=epochs_p1, ckpt_path="models/ckpt_p1.keras"
        )

        # ── Step 5: Phase 2 ───────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 5 — Phase 2: Fine-tuning")
        logger.info("=" * 60)
        history_p2 = train_phase2(
            model, base, train_gen_p2, val_gen,
            epochs=epochs_p2,
            unfreeze_top=unfreeze_top,
            ckpt_p1="models/ckpt_p1.keras",
            ckpt_path="models/ckpt_best.keras",
        )

        # ── Step 6: Save model ────────────────────────────────────────────────
        if os.path.exists("models/ckpt_best.keras"):
            model.load_weights("models/ckpt_best.keras")
        os.makedirs(os.path.dirname(model_out), exist_ok=True)
        model.save(model_out)
        logger.info("Model saved → %s", model_out)

    else:
        import tensorflow as tf
        logger.info("Skipping training — loading existing model: %s", model_out)
        model = tf.keras.models.load_model(model_out)
        history_p1 = history_p2 = None

    # ── Step 7: Evaluation ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6 — Evaluation")
    logger.info("=" * 60)
    val_metrics = evaluate_generator(model, val_gen, split_name="validation")
    test_metrics = evaluate_generator(model, test_gen, split_name="test")

    plot_confusion_matrix(val_metrics["y_true"], val_metrics["y_pred"],
                          save_path="logs/confusion_matrix_val.png")
    plot_confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"],
                          save_path="logs/confusion_matrix_test.png")
    plot_roc_curve(val_metrics["fpr"], val_metrics["tpr"], val_metrics["auc"],
                   save_path="logs/roc_curve_val.png")
    plot_roc_curve(test_metrics["fpr"], test_metrics["tpr"], test_metrics["auc"],
                   save_path="logs/roc_curve_test.png")
    save_report([val_metrics, test_metrics], save_path="logs/evaluation_report.txt")

    # ── Step 8: Class config ──────────────────────────────────────────────────
    save_class_config(
        "models/class_names.json",
        val_accuracy=val_metrics["accuracy"],
        val_auc=val_metrics["auc"],
        test_accuracy=test_metrics["accuracy"],
        test_auc=test_metrics["auc"],
        total_images=train_gen_p1.samples + val_gen.samples + test_gen.samples,
        train_frac=train_frac,
        val_frac=val_frac,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("  Val  Accuracy : %.4f", val_metrics["accuracy"])
    logger.info("  Val  AUC      : %.4f", val_metrics["auc"])
    logger.info("  Test Accuracy : %.4f", test_metrics["accuracy"])
    logger.info("  Test AUC      : %.4f", test_metrics["auc"])
    logger.info("  Model         : %s", model_out)
    logger.info("=" * 60)


def parse_args():
    p = argparse.ArgumentParser(description="AI vs Real — full pipeline")
    p.add_argument("--zip", required=True, help="Path to dataset zip")
    p.add_argument("--dataset_root", default="data/dataset")
    p.add_argument("--model_out", default="models/model_v1.keras")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--skip_training", action="store_true",
                   help="Skip training and only run evaluation")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        zip_path=args.zip,
        dataset_root=args.dataset_root,
        model_out=args.model_out,
        config_path=args.config,
        skip_training=args.skip_training,
    )
