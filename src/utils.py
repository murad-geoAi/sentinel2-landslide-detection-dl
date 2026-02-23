"""
utils.py
========
Utility functions for the landslide classification project:

  • set_seed()              – Reproducible training (Python / NumPy / PyTorch)
  • plot_roc_curve()        – Saves ROC curve PNG
  • plot_confusion_matrix() – Saves confusion matrix heatmap PNG
  • plot_loss_curves()      – Saves train vs val loss curve PNG
  • save_metrics_csv()      – Exports metrics dict to CSV
  • threshold_optimization() – Grid search for best F1 threshold
  • extract_tb_scalars()    – Reads TensorBoard event logs into DataFrames
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────── #
#  Reproducibility                                                     #
# ─────────────────────────────────────────────────────────────────── #
def set_seed(seed: int = 42) -> None:
    """
    Sets random seeds for Python, NumPy, and PyTorch (CPU + CUDA).
    Enables deterministic cuDNN mode.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic cuDNN ops (may reduce speed slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Global seed set to %d.", seed)


# ─────────────────────────────────────────────────────────────────── #
#  Visualization                                                       #
# ─────────────────────────────────────────────────────────────────── #
def plot_roc_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    out_path: str | Path,
    title: str = "ROC Curve",
) -> float:
    """
    Plots and saves the ROC curve, returns the computed AUC.

    Args:
        y_true   : Ground-truth binary labels (0/1).
        y_probs  : Model probability for the positive class (landslide).
        out_path : File path to save the PNG.
        title    : Plot title.

    Returns:
        roc_auc (float)
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs, pos_label=1)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        fpr,
        tpr,
        color="#E63946",
        lw=2.5,
        label=f"AUC = {roc_auc:.4f}",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#6C757D", lw=1.5, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#E63946")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("ROC curve saved → %s  (AUC=%.4f)", out_path, roc_auc)
    return roc_auc


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    out_path: str | Path,
    title: str = "Confusion Matrix",
) -> np.ndarray:
    """
    Plots and saves a seaborn confusion matrix heatmap.

    Args:
        y_true      : Ground-truth labels.
        y_pred      : Predicted labels.
        class_names : List of class name strings.
        out_path    : File path to save the PNG.
        title       : Plot title.

    Returns:
        The raw confusion matrix as ndarray.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 14},
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", out_path)
    return cm


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    out_path: str | Path,
    title: str = "Training vs Validation Loss",
) -> None:
    """
    Plots and saves training vs validation loss curves.

    Args:
        train_losses : Per-epoch training losses.
        val_losses   : Per-epoch validation losses.
        out_path     : File path to save the PNG.
        title        : Plot title.
    """
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, "o-", color="#1D3557", lw=2, label="Train Loss")
    ax.plot(epochs, val_losses, "s-", color="#E63946", lw=2, label="Val Loss")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Loss curves saved → %s", out_path)


def plot_metrics_bar(
    metrics_df: pd.DataFrame,
    out_path: str | Path,
    title: str = "Model Comparison",
) -> None:
    """
    Plots a grouped bar chart comparing metrics across models.

    Args:
        metrics_df: DataFrame with columns ['Model', 'Accuracy', 'F1', 'Recall',
                    'Precision', 'AUC'] or similar.
        out_path  : File path to save the PNG.
        title     : Plot title.
    """
    metrics_df = metrics_df.set_index("Model")
    ax = metrics_df.plot(
        kind="bar",
        figsize=(10, 6),
        width=0.7,
        edgecolor="white",
        colormap="tab10",
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(fontsize=10, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Metrics bar chart saved → %s", out_path)


# ─────────────────────────────────────────────────────────────────── #
#  Metrics Export                                                      #
# ─────────────────────────────────────────────────────────────────── #
def save_metrics_csv(
    metrics: Dict[str, float],
    out_path: str | Path,
    model_name: str = "model",
) -> None:
    """
    Saves a metrics dictionary to CSV.

    Args:
        metrics    : Dict of metric_name → float value.
        out_path   : Path to write the CSV file.
        model_name : Model identifier prepended as a column.
    """
    row = {"model": model_name, **metrics}
    df = pd.DataFrame([row])

    out_path = Path(out_path)
    if out_path.exists():
        df_existing = pd.read_csv(out_path)
        df = pd.concat([df_existing, df], ignore_index=True)

    df.to_csv(out_path, index=False, float_format="%.4f")
    logger.info("Metrics saved → %s", out_path)


# ─────────────────────────────────────────────────────────────────── #
#  Threshold Optimization                                              #
# ─────────────────────────────────────────────────────────────────── #
def threshold_optimization(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    metric: str = "f1",
    n_thresholds: int = 200,
) -> Tuple[float, float]:
    """
    Grid searches for the optimal classification threshold that maximizes
    the specified metric.

    Args:
        y_true       : Ground-truth binary labels.
        y_probs      : Predicted probabilities for the positive class.
        metric       : Metric to maximize ('f1' or 'recall' or 'precision').
        n_thresholds : Number of threshold candidates to evaluate.

    Returns:
        Tuple of (best_threshold, best_score).
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    scores = []
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        if metric == "f1":
            s = f1_score(y_true, preds, zero_division=0)
        elif metric == "recall":
            from sklearn.metrics import recall_score

            s = recall_score(y_true, preds, zero_division=0)
        elif metric == "precision":
            from sklearn.metrics import precision_score

            s = precision_score(y_true, preds, zero_division=0)
        else:
            raise ValueError(f"Unknown metric '{metric}'.")
        scores.append(s)

    best_idx = int(np.argmax(scores))
    best_threshold = float(thresholds[best_idx])
    best_score = float(scores[best_idx])
    logger.info(
        "Optimal threshold (max %s): %.4f  → %s = %.4f",
        metric,
        best_threshold,
        metric,
        best_score,
    )
    return best_threshold, best_score


# ─────────────────────────────────────────────────────────────────── #
#  TensorBoard Log Reader                                              #
# ─────────────────────────────────────────────────────────────────── #
def extract_tb_scalars(log_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """
    Reads TensorBoard event files from log_dir and returns a dict of
    { tag: DataFrame(step, value) } for all scalar summaries.

    Args:
        log_dir: Path to TensorBoard log directory.

    Returns:
        Dictionary mapping tag names to DataFrames.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        logger.warning(
            "tensorboard package not found. Install it to extract TB scalars."
        )
        return {}

    ea = EventAccumulator(str(log_dir))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])

    result: Dict[str, pd.DataFrame] = {}
    for tag in tags:
        events = ea.Scalars(tag)
        result[tag] = pd.DataFrame(
            {"step": [e.step for e in events], "value": [e.value for e in events]}
        )
    return result
