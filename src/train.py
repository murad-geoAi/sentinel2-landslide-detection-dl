"""
train.py
========
CLI entry point for training the landslide detection model.

Usage examples:
    # Train ResNet18 with Focal Loss (default config)
    python src/train.py --data_root "g:/DataSet for CNN/LandslideData" --model resnet18 --loss focal

    # Train Baseline CNN with Weighted CE (custom settings)
    python src/train.py \\
        --data_root "g:/DataSet for CNN/LandslideData" \\
        --model baseline_cnn \\
        --loss weighted_ce \\
        --epochs 80 \\
        --lr 5e-4 \\
        --batch_size 16 \\
        --output_dir "g:/DataSet for CNN/landslide_dl_project/outputs/baseline_wce"

    # Load config from YAML
    python src/train.py --config configs/default_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# ── Ensure project root is on sys.path ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

from src.datamodule import LandslideDataModule
from src.lightning_module import LandslideClassifier
from src.losses import get_loss_fn
from src.model import get_model
from src.utils import (
    plot_confusion_matrix,
    plot_roc_curve,
    save_metrics_csv,
    set_seed,
    threshold_optimization,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ─────────────────────────────────────────────────────────────────── #
#  Argument Parsing                                                    #
# ─────────────────────────────────────────────────────────────────── #
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train landslide classification model on Sentinel-2 patches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser.add_argument(
        "--data_root",
        type=str,
        default="g:/DataSet for CNN/LandslideData",
        help="Root folder containing images/ and labels/ sub-directories.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="g:/DataSet for CNN/landslide_dl_project/outputs/run",
        help="Directory for checkpoints, logs, and output plots.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a YAML config file. CLI flags override YAML values.",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "baseline_cnn"],
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights for ResNet18.",
    )
    parser.add_argument(
        "--num_bands",
        type=int,
        default=4,
        help="Number of input spectral bands.",
    )
    parser.add_argument(
        "--unfreeze_epoch",
        type=int,
        default=10,
        help="Epoch at which to unfreeze ResNet backbone. -1 to disable.",
    )

    # Loss
    parser.add_argument(
        "--loss",
        type=str,
        default="focal",
        choices=["focal", "weighted_ce"],
        help="Loss function for handling class imbalance.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for Focal Loss.",
    )

    # Optimizer & Scheduler
    parser.add_argument(
        "--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"]
    )
    parser.add_argument(
        "--scheduler", type=str, default="cosine", choices=["cosine", "step", "none"]
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Training
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument(
        "--patience", type=int, default=12, help="EarlyStopping patience (epochs)."
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Run 1 batch per step for smoke testing.",
    )

    args = parser.parse_args(argv)

    # ── Override from YAML config ──────────────────────────────────
    if args.config is not None:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        # Apply YAML values for any args that still hold their default
        for key, value in cfg.items():
            # Only set if the arg exists and user didn't explicitly pass it
            if hasattr(args, key):
                setattr(args, key, value)

    return args


# ─────────────────────────────────────────────────────────────────── #
#  Post-training Evaluation                                            #
# ─────────────────────────────────────────────────────────────────── #
def evaluate_and_save(
    trainer: "Trainer",
    module: LandslideClassifier,
    datamodule: LandslideDataModule,
    output_dir: Path,
    model_tag: str,
    threshold: float = 0.5,
) -> None:
    """
    Runs final validation, computes all metrics, and saves plots + CSV.
    """
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    logger.info("Running final evaluation on validation set …")
    module.eval()
    val_loader = datamodule.val_dataloader()
    device = next(module.parameters()).device

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = module(images)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)

    # ── Threshold optimization ─────────────────────────────────────
    best_threshold, best_f1 = threshold_optimization(y_true, y_prob, metric="f1")
    y_pred = (y_prob >= best_threshold).astype(int)

    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc": round(roc_auc_score(y_true, y_prob), 4),
        "best_threshold": round(best_threshold, 4),
    }

    logger.info("── Final Metrics (%s) ──", model_tag)
    for k, v in metrics.items():
        logger.info("  %-18s : %s", k, v)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_roc_curve(
        y_true,
        y_prob,
        out_path=plots_dir / f"roc_curve_{model_tag}.png",
        title=f"ROC Curve – {model_tag.replace('_', ' ').title()}",
    )
    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names=["Non-LS", "Landslide"],
        out_path=plots_dir / f"confusion_matrix_{model_tag}.png",
        title=f"Confusion Matrix – {model_tag.replace('_', ' ').title()}",
    )

    csv_path = output_dir.parent / "metrics_summary.csv"
    save_metrics_csv(metrics, csv_path, model_name=model_tag)
    logger.info("Metrics CSV updated → %s", csv_path)


# ─────────────────────────────────────────────────────────────────── #
#  Main training loop                                                  #
# ─────────────────────────────────────────────────────────────────── #
def main(argv=None) -> None:
    args = parse_args(argv)

    # ── Reproducibility ───────────────────────────────────────────
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive a short tag for naming outputs
    model_tag = f"{args.model}_{args.loss}"

    logger.info("=" * 60)
    logger.info("  Landslide Detection  |  %s  |  Loss: %s", args.model, args.loss)
    logger.info("=" * 60)

    # ── DataModule ────────────────────────────────────────────────
    datamodule = LandslideDataModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        num_bands=args.num_bands,
        seed=args.seed,
        pin_memory=torch.cuda.is_available(),
    )
    datamodule.setup()

    # ── Model ─────────────────────────────────────────────────────
    pretrained = not args.no_pretrained
    model = get_model(
        name=args.model,
        num_bands=args.num_bands,
        pretrained=pretrained,
        freeze_backbone=(args.model == "resnet18"),
    )

    # ── Loss ──────────────────────────────────────────────────────
    class_weights = datamodule.class_weights
    loss_fn = get_loss_fn(
        name=args.loss,
        class_weights=class_weights,
        focal_gamma=args.focal_gamma,
    )

    # ── Lightning Module ──────────────────────────────────────────
    unfreeze_ep = args.unfreeze_epoch if args.unfreeze_epoch >= 0 else None
    module = LandslideClassifier(
        model=model,
        loss_fn=loss_fn,
        lr=args.lr,
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        unfreeze_epoch=unfreeze_ep,
    )

    # ── Callbacks ─────────────────────────────────────────────────
    ckpt_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename=f"{model_tag}_best",
        monitor="val_auroc",
        mode="max",
        save_top_k=1,
        verbose=True,
    )
    early_stop = EarlyStopping(
        monitor="val_auroc",
        mode="max",
        patience=args.patience,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = RichProgressBar()

    # ── Logger ────────────────────────────────────────────────────
    tb_logger = TensorBoardLogger(
        save_dir=str(output_dir / "logs"),
        name=model_tag,
        version="v1",
    )

    # ── Trainer ───────────────────────────────────────────────────
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        callbacks=[ckpt_callback, early_stop, lr_monitor, progress_bar],
        logger=tb_logger,
        log_every_n_steps=1,
        deterministic=True,
        fast_dev_run=args.fast_dev_run,
        enable_model_summary=True,
    )

    # ── Train ─────────────────────────────────────────────────────
    logger.info("Starting training … [%d epochs max]", args.epochs)
    trainer.fit(module, datamodule=datamodule)

    if not args.fast_dev_run:
        # ── Post-training evaluation ──────────────────────────────
        best_ckpt = ckpt_callback.best_model_path
        logger.info("Best checkpoint: %s", best_ckpt)

        if best_ckpt:
            best_module = LandslideClassifier.load_from_checkpoint(
                best_ckpt,
                model=model,
                loss_fn=loss_fn,
                strict=False,
            )
        else:
            best_module = module

        best_module.to("cuda" if torch.cuda.is_available() else "cpu")
        evaluate_and_save(
            trainer=trainer,
            module=best_module,
            datamodule=datamodule,
            output_dir=output_dir,
            model_tag=model_tag,
        )

    logger.info("Training complete. Outputs saved to: %s", output_dir)


if __name__ == "__main__":
    main()
