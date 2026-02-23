"""
model.py
========
Model architectures for landslide detection:

  1. ResNet18Classifier  – Transfer learning from ImageNet pretrained ResNet18.
                           The first conv layer is patched to accept 4-band input.
                           The final FC layer is replaced for binary classification.

  2. BaselineCNN         – Lightweight 4-layer CNN trained from scratch as a
                           comparison baseline.

Both models output raw logits of shape (B, 2) for use with CrossEntropyLoss.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

logger = logging.getLogger(__name__)

# Supported model names
ModelName = Literal["resnet18", "baseline_cnn"]

NUM_CLASSES = 2


# ------------------------------------------------------------------ #
#  ResNet18 Transfer Learning                                          #
# ------------------------------------------------------------------ #
class ResNet18Classifier(nn.Module):
    """
    ResNet18 adapted for 4-band Sentinel-2 input with binary classification head.

    Modifications vs. vanilla ResNet18:
        • conv1: in_channels 3 → 4 (extra NIR band).  Pre-trained RGB weights are
          averaged across channels and replicated; the 4th channel is initialized
          from the mean of R, G, B weights (Ayush et al., 2021 strategy).
        • layer1..layer3: frozen during first training phase (feature extraction).
        • layer4 + fc: trainable (fine-tuning head).
        • fc: (512) → Linear(512, 2)

    Args:
        num_bands      : Number of input channels (default 4).
        freeze_backbone: If True, freeze layers 1-3 for initial training.
        pretrained     : Load ImageNet weights for the RGB backbone.
    """

    def __init__(
        self,
        num_bands: int = 4,
        freeze_backbone: bool = True,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # ── Patch conv1 for 4-band input ────────────────────────────
        old_conv = backbone.conv1  # (64, 3, 7, 7)
        new_conv = nn.Conv2d(
            in_channels=num_bands,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        if pretrained and num_bands != 3:
            with torch.no_grad():
                # Replicate mean RGB weight for the extra band(s)
                mean_rgb = old_conv.weight.mean(dim=1, keepdim=True)  # (64,1,7,7)
                new_conv.weight[:, :3, :, :] = old_conv.weight
                for i in range(3, num_bands):
                    new_conv.weight[:, i : i + 1, :, :] = mean_rgb

        backbone.conv1 = new_conv

        # ── Replace classification head ──────────────────────────────
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, NUM_CLASSES),
        )

        self.backbone = backbone

        # ── Optionally freeze feature layers ────────────────────────
        if freeze_backbone:
            frozen_layers = [
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
            ]
            for layer in frozen_layers:
                for param in layer.parameters():
                    param.requires_grad = False
            logger.info("ResNet18: layers 1-3 frozen, layers 4 + head trainable.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def unfreeze_all(self) -> None:
        """Unfreeze entire backbone for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("ResNet18: all layers unfrozen for fine-tuning.")


# ------------------------------------------------------------------ #
#  Baseline CNN (trained from scratch)                                 #
# ------------------------------------------------------------------ #
class BaselineCNN(nn.Module):
    """
    Simple 4-block CNN trained from scratch as a performance baseline.

    Architecture:
        Block 1: Conv(4→32, 3×3) → BN → ReLU → MaxPool(2)
        Block 2: Conv(32→64, 3×3) → BN → ReLU → MaxPool(2)
        Block 3: Conv(64→128, 3×3) → BN → ReLU → MaxPool(2)
        Block 4: Conv(128→256, 3×3) → BN → ReLU → AdaptiveAvgPool(1)
        Head   : Dropout(0.5) → Linear(256, 2)

    Args:
        num_bands: Number of input channels (default 4).
    """

    def __init__(self, num_bands: int = 4) -> None:
        super().__init__()

        def _block(in_ch: int, out_ch: int, pool: bool = True) -> nn.Sequential:
            layers: list = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2, 2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            _block(num_bands, 32),  # (B, 32, H/2, W/2)
            _block(32, 64),  # (B, 64, H/4, W/4)
            _block(64, 128),  # (B,128, H/8, W/8)
            _block(128, 256, pool=False),  # (B,256, H/8, W/8)
            nn.AdaptiveAvgPool2d(1),  # (B,256, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ------------------------------------------------------------------ #
#  Factory                                                             #
# ------------------------------------------------------------------ #
def get_model(name: ModelName, num_bands: int = 4, **kwargs) -> nn.Module:
    """
    Factory function to create a model by name.

    Args:
        name      : "resnet18" or "baseline_cnn"
        num_bands : Number of spectral input bands
        **kwargs  : Additional keyword arguments passed to the model constructor.

    Returns:
        nn.Module instance

    Raises:
        ValueError: if name is not recognized.
    """
    name = name.lower()
    if name == "resnet18":
        model = ResNet18Classifier(num_bands=num_bands, **kwargs)
        logger.info("Created ResNet18Classifier (pretrained, 4-band patched).")
    elif name in ("baseline_cnn", "baseline"):
        model = BaselineCNN(num_bands=num_bands)
        logger.info("Created BaselineCNN (from scratch).")
    else:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: 'resnet18', 'baseline_cnn'."
        )
    return model
