"""
evaluate.py — Model evaluation: accuracy, AUC, confusion matrix, ROC curve.

Usage:
  python src/evaluate.py --model models/model_v1.keras --data data/dataset

Outputs:
  logs/training_curves.png  (if history pickle supplied)
  logs/confusion_matrix.png
  logs/roc_curve.png
  logs/evaluation_report.txt
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)

import tensorflow as tf

from src.preprocess import get_test_generator
from src.utils import get_logger, load_config

logger = get_logger()


# ─── Core evaluation ─────────────────────────────────────────────────────────

def evaluate_generator(model, generator, split_name: str = "validation") -> dict:
    """Run model on a generator and return metrics dict."""
    generator.reset()
    y_prob = model.predict(generator, verbose=1).flatten()
    y_pred = (y_prob > 0.5).astype(int)
    y_true = generator.classes

    acc = accuracy_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    logger.info("%s — Accuracy: %.4f | AUC: %.4f", split_name, acc, roc_auc)
    return {
        "split": split_name,
        "accuracy": acc,
        "auc": roc_auc,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "fpr": fpr,
        "tpr": tpr,
    }


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true, y_pred, save_path: str = "logs/confusion_matrix.png"
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["AI", "REAL"], yticklabels=["AI", "REAL"],
        ax=ax, linewidths=0.5,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix → %s", save_path)


def plot_roc_curve(fpr, tpr, roc_auc: float, save_path: str = "logs/roc_curve.png") -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, "b-", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("ROC curve → %s", save_path)


def plot_training_curves(history_dict: dict, save_path: str = "logs/training_curves.png") -> None:
    """
    history_dict: {'accuracy': [...], 'val_accuracy': [...], 'loss': [...], 'val_loss': [...]}
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    acc = history_dict.get("accuracy", [])
    val_acc = history_dict.get("val_accuracy", [])
    loss = history_dict.get("loss", [])
    val_loss = history_dict.get("val_loss", [])
    epochs_r = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("EfficientNetB3 — Training Curves", fontsize=13, fontweight="bold")

    ax1.plot(epochs_r, acc, "b-", label="Train Acc")
    ax1.plot(epochs_r, val_acc, "r--", label="Val Acc")
    ax1.set_title("Accuracy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs_r, loss, "b-", label="Train Loss")
    ax2.plot(epochs_r, val_loss, "r--", label="Val Loss")
    ax2.set_title("Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Training curves → %s", save_path)


# ─── Full evaluation report ───────────────────────────────────────────────────

def save_report(metrics_list: list[dict], save_path: str = "logs/evaluation_report.txt") -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    lines = ["=" * 60, "AI vs Real Classifier — Evaluation Report", "=" * 60, ""]

    for m in metrics_list:
        lines.append(f"  Split      : {m['split']}")
        lines.append(f"  Accuracy   : {m['accuracy']:.4f} ({m['accuracy']*100:.2f}%)")
        lines.append(f"  AUC        : {m['auc']:.4f}")
        lines.append("")
        lines.append(
            classification_report(
                m["y_true"], m["y_pred"], target_names=["AI", "REAL"]
            )
        )
        lines.append("-" * 60)

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Evaluation report → %s", save_path)


# ─── Entry point ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/model_v1.keras")
    p.add_argument("--data", default="data/dataset")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--output_dir", default="logs")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    img_size = cfg["model"]["img_size"]
    batch_size = cfg["training"]["batch_size"]

    logger.info("Loading model: %s", args.model)
    model = tf.keras.models.load_model(args.model)

    from src.preprocess import get_phase1_generators
    _, val_gen = get_phase1_generators(
        args.data, img_size=img_size, batch_size=batch_size
    )
    test_gen = get_test_generator(args.data, img_size=img_size, batch_size=batch_size)

    val_metrics = evaluate_generator(model, val_gen, split_name="validation")
    test_metrics = evaluate_generator(model, test_gen, split_name="test")

    plot_confusion_matrix(
        val_metrics["y_true"], val_metrics["y_pred"],
        save_path=os.path.join(args.output_dir, "confusion_matrix_val.png"),
    )
    plot_confusion_matrix(
        test_metrics["y_true"], test_metrics["y_pred"],
        save_path=os.path.join(args.output_dir, "confusion_matrix_test.png"),
    )
    plot_roc_curve(
        val_metrics["fpr"], val_metrics["tpr"], val_metrics["auc"],
        save_path=os.path.join(args.output_dir, "roc_curve_val.png"),
    )
    plot_roc_curve(
        test_metrics["fpr"], test_metrics["tpr"], test_metrics["auc"],
        save_path=os.path.join(args.output_dir, "roc_curve_test.png"),
    )
    save_report(
        [val_metrics, test_metrics],
        save_path=os.path.join(args.output_dir, "evaluation_report.txt"),
    )

    gap = abs(val_metrics["accuracy"] - test_metrics["accuracy"])
    logger.info(
        "Val↔Test accuracy gap: %.2f%%  %s",
        gap * 100,
        "✅ Generalises well" if gap < 0.05 else "⚠️ >5% gap — possible overfitting",
    )


if __name__ == "__main__":
    main()
