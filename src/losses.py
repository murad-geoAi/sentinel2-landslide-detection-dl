"""
losses.py
=========
Loss functions designed for class-imbalanced binary classification:

  1. WeightedCrossEntropyLoss  – Standard CE with inverse-frequency class weights.
  2. FocalLoss                  – Lin et al. (2017) focal loss; down-weights easy
                                   negatives so the model focuses on hard examples.

Both classes follow the same interface: forward(logits, targets) → scalar loss.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

LossName = Literal["weighted_ce", "focal"]


# ------------------------------------------------------------------ #
#  Weighted Cross-Entropy                                              #
# ------------------------------------------------------------------ #
class WeightedCrossEntropyLoss(nn.Module):
    """
    Wraps nn.CrossEntropyLoss with pre-computed class weights.

    Args:
        class_weights: Tensor of shape (num_classes,).
                       Typically inverse-frequency: w_c = N / (C * n_c)
        reduction    : Reduction method ('mean', 'sum', 'none').
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.reduction = reduction
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, C) raw model outputs.
            targets : (B,)  class indices (long tensor).

        Returns:
            Scalar loss value.
        """
        weight = self.class_weights if self.class_weights is not None else None
        return F.cross_entropy(logits, targets, weight=weight, reduction=self.reduction)


# ------------------------------------------------------------------ #
#  Focal Loss                                                           #
# ------------------------------------------------------------------ #
class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) for addressing class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha  : Weighting factor (scalar or tensor of shape (C,)).
                 If float, applied uniformly. Use class_weights for per-class alpha.
        gamma  : Focusing parameter (≥0). gamma=0 reduces to cross-entropy.
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (float, int)):
                alpha = torch.tensor([alpha, 1 - alpha], dtype=torch.float32)
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, C) raw model outputs.
            targets : (B,)  class indices (long tensor).

        Returns:
            Scalar focal loss.
        """
        # Convert logits to log-probabilities
        log_probs = F.log_softmax(logits, dim=1)  # (B, C)
        probs = torch.exp(log_probs)  # (B, C)

        # Gather probabilities for the target class
        targets = targets.view(-1, 1)  # (B, 1)
        log_pt = log_probs.gather(1, targets).squeeze(1)  # (B,)
        pt = probs.gather(1, targets).squeeze(1)  # (B,)

        # Focal weighting
        focal_weight = (1 - pt) ** self.gamma  # (B,)

        # Alpha weighting (per-class)
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets.squeeze(1))  # (B,)
            focal_weight = alpha_t * focal_weight

        loss = -focal_weight * log_pt  # (B,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ------------------------------------------------------------------ #
#  Factory                                                             #
# ------------------------------------------------------------------ #
def get_loss_fn(
    name: LossName,
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """
    Factory for loss functions.

    Args:
        name          : "weighted_ce" or "focal"
        class_weights : Tensor of class weights (used by both loss types).
        focal_gamma   : Focusing parameter for FocalLoss.

    Returns:
        Configured nn.Module loss instance.
    """
    name = name.lower()
    if name == "weighted_ce":
        return WeightedCrossEntropyLoss(class_weights=class_weights)
    elif name == "focal":
        return FocalLoss(alpha=class_weights, gamma=focal_gamma)
    else:
        raise ValueError(f"Unknown loss '{name}'. Choose from: 'weighted_ce', 'focal'.")
