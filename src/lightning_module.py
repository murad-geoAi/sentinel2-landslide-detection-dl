"""
lightning_module.py
====================
PyTorch Lightning LightningModule for landslide binary classification.

Features:
    • Unified training/validation loop using torchmetrics
    • Metrics: Accuracy, Precision, Recall, F1, AUROC
    • Confusion matrix logged to TensorBoard each validation epoch
    • Grad-CAM visualization support via pytorch_grad_cam
    • Configurable optimizer (Adam/SGD) + LR scheduler (CosineAnnealingLR)
    • Threshold optimization for best F1-score
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)

logger = logging.getLogger(__name__)

OptimizerName = Literal["adam", "sgd", "adamw"]
SchedulerName = Literal["cosine", "step", "none"]


# ─────────────────────────────────────────────────────────────────── #
#  LandslideClassifier                                                 #
# ─────────────────────────────────────────────────────────────────── #
class LandslideClassifier(LightningModule):
    """
    PyTorch Lightning module for binary landslide classification.

    Args:
        model          : nn.Module (ResNet18Classifier or BaselineCNN)
        loss_fn        : Loss function (WeightedCrossEntropyLoss or FocalLoss)
        lr             : Initial learning rate
        optimizer_name : 'adam' | 'sgd' | 'adamw'
        scheduler_name : 'cosine' | 'step' | 'none'
        weight_decay   : L2 regularization
        max_epochs     : Used by CosineAnnealingLR
        unfreeze_epoch : Epoch at which to unfreeze the ResNet backbone
                         (set to None to keep frozen throughout).
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        lr: float = 1e-3,
        optimizer_name: OptimizerName = "adamw",
        scheduler_name: SchedulerName = "cosine",
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
        unfreeze_epoch: Optional[int] = 10,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.unfreeze_epoch = unfreeze_epoch

        # Save hyper-params to checkpoint (excludes bulky model/loss objects)
        self.save_hyperparameters(
            ignore=["model", "loss_fn"],
            logger=False,
        )

        # ── Metrics ─────────────────────────────────────────────────
        metric_kwargs = {"threshold": 0.5}
        shared_metrics = MetricCollection(
            {
                "accuracy": BinaryAccuracy(**metric_kwargs),
                "precision": BinaryPrecision(**metric_kwargs),
                "recall": BinaryRecall(**metric_kwargs),
                "f1": BinaryF1Score(**metric_kwargs),
                "auroc": BinaryAUROC(),
            }
        )
        self.train_metrics = shared_metrics.clone(prefix="train_")
        self.val_metrics = shared_metrics.clone(prefix="val_")

        # Storage for per-epoch val outputs (for confusion matrix)
        self._val_probs: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []

    # ── Forward ──────────────────────────────────────────────────── #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ── Training step ────────────────────────────────────────────── #

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)

        probs = torch.softmax(logits, dim=1)[:, 1]
        self.train_metrics.update(probs, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, prog_bar=False)
        self.train_metrics.reset()

    # ── Validation step ──────────────────────────────────────────── #

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)

        probs = torch.softmax(logits, dim=1)[:, 1]
        self.val_metrics.update(probs, labels)

        # Store for confusion matrix
        self._val_probs.append(probs.detach().cpu())
        self._val_labels.append(labels.detach().cpu())

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        # Compute and log scalar metrics
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, prog_bar=True)
        self.val_metrics.reset()

        # Compute confusion matrix and log as figure
        if self._val_probs:
            all_probs = torch.cat(self._val_probs)
            all_labels = torch.cat(self._val_labels)
            self._log_confusion_matrix(all_probs, all_labels)
            self._val_probs.clear()
            self._val_labels.clear()

    # ── Confusion matrix helper ──────────────────────────────────── #

    def _log_confusion_matrix(self, probs: torch.Tensor, labels: torch.Tensor) -> None:
        """Computes confusion matrix and logs it as a TensorBoard figure."""
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        preds = (probs >= 0.5).long().numpy()
        y_true = labels.numpy()
        cm = confusion_matrix(y_true, preds, labels=[0, 1])

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Non-LS", "Landslide"],
            yticklabels=["Non-LS", "Landslide"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix (Epoch {self.current_epoch})")
        plt.tight_layout()

        if self.logger is not None:
            self.logger.experiment.add_figure(
                "val/confusion_matrix", fig, global_step=self.current_epoch
            )
        plt.close(fig)

    # ── Backbone unfreezing ──────────────────────────────────────── #

    def on_train_epoch_start(self) -> None:
        if (
            self.unfreeze_epoch is not None
            and self.current_epoch == self.unfreeze_epoch
            and hasattr(self.model, "unfreeze_all")
        ):
            logger.info(
                "Epoch %d: Unfreezing ResNet backbone for full fine-tuning.",
                self.current_epoch,
            )
            self.model.unfreeze_all()

    # ── Optimizer & Scheduler ────────────────────────────────────── #

    def configure_optimizers(self) -> Dict[str, Any]:
        params = filter(lambda p: p.requires_grad, self.model.parameters())

        opt_name = self.optimizer_name.lower()
        if opt_name == "adam":
            optimizer = torch.optim.Adam(
                params, lr=self.lr, weight_decay=self.weight_decay
            )
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(
                params, lr=self.lr, weight_decay=self.weight_decay
            )
        elif opt_name == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
                nesterov=True,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        sched_name = self.scheduler_name.lower()
        if sched_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_epochs, eta_min=1e-6
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_auroc",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        elif sched_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.5
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    # ── Grad-CAM helper (called from inference.py) ───────────────── #

    def get_gradcam_target_layer(self) -> Optional[nn.Module]:
        """
        Returns the target layer for Grad-CAM visualization.

        For ResNet18: returns layer4[-1].
        For BaselineCNN: returns the last Conv2d block before AdaptiveAvgPool.
        """
        model = self.model
        # Try ResNet18 backbone
        if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
            return model.backbone.layer4[-1]
        # Try BaselineCNN features
        if hasattr(model, "features"):
            for layer in reversed(list(model.features.children())):
                if isinstance(layer, (nn.Sequential, nn.Conv2d)):
                    return layer
        return None
