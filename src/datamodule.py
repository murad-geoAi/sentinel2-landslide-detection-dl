"""
datamodule.py
=============
PyTorch Lightning DataModule for Sentinel-2 landslide patch classification.

Handles:
    • Dataset building (image-label pairing from EMD-exported tiles)
    • Stratified 80/20 train/validation split
    • Augmentation pipelines:
        Training   – RandomHorizontalFlip, RandomVerticalFlip, RandomRotation,
                     ColorJitter (on RGB bands), RandomResizedCrop
        Validation – CenterCrop only (no augmentation)
    • DataLoader creation with appropriate num_workers
    • Class weight computation for imbalanced datasets
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import LandslideDataset, build_dataframe

logger = logging.getLogger(__name__)

# Input size expected by ResNet18 (and used for Baseline CNN)
IMAGE_SIZE = 224


class LandslideDataModule(LightningDataModule):
    """
    LightningDataModule for Sentinel-2 landslide patch classification.

    Args:
        data_root   : Path to the ESRI Classified Tiles root dir
                      (must contain 'images/' and 'labels/' sub-dirs).
        batch_size  : Batch size for all DataLoaders.
        num_workers : Number of DataLoader worker processes.
        val_split   : Fraction of data reserved for validation (default 0.2).
        num_bands   : Number of spectral bands to load (default 4).
        seed        : Random seed for reproducible stratified split.
        pin_memory  : Pin memory for GPU training.
    """

    def __init__(
        self,
        data_root: str,
        batch_size: int = 8,
        num_workers: int = 0,
        val_split: float = 0.2,
        num_bands: int = 4,
        seed: int = 42,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.num_bands = num_bands
        self.seed = seed
        self.pin_memory = pin_memory

        self._class_weights: Optional[torch.Tensor] = None

    # ── Transforms ───────────────────────────────────────────────── #

    @property
    def train_transform(self) -> transforms.Compose:
        """Augmentation pipeline for training (applied AFTER normalization)."""
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    IMAGE_SIZE, scale=(0.7, 1.0), ratio=(0.9, 1.1)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=90),
                # ColorJitter applied only to first 3 bands (RGB surrogate).
                # Full 4-band tensor jitter is done with RandomApply over a
                # custom lambda to avoid library incompatibility.
                transforms.RandomApply(
                    [transforms.Lambda(lambda x: _jitter_rgb_bands(x))],
                    p=0.3,
                ),
            ]
        )

    @property
    def val_transform(self) -> transforms.Compose:
        """Deterministic crop for validation (no augmentation)."""
        return transforms.Compose(
            [
                transforms.CenterCrop(IMAGE_SIZE),
            ]
        )

    # ── DataModule lifecycle ─────────────────────────────────────── #

    def setup(self, stage: Optional[str] = None) -> None:
        """Called on every process in DDP. Builds datasets for train/val."""
        df = build_dataframe(self.data_root)

        # Stratified split
        train_df, val_df = train_test_split(
            df,
            test_size=self.val_split,
            stratify=df["class_idx"],
            random_state=self.seed,
        )

        logger.info(
            "Split → Train: %d | Val: %d  (Landslide: %d train / %d val)",
            len(train_df),
            len(val_df),
            (train_df["class_idx"] == 1).sum(),
            (val_df["class_idx"] == 1).sum(),
        )

        self.train_dataset = LandslideDataset(
            dataframe=train_df,
            transform=self.train_transform,
            num_bands=self.num_bands,
            normalize=True,
        )
        self.val_dataset = LandslideDataset(
            dataframe=val_df,
            transform=self.val_transform,
            num_bands=self.num_bands,
            normalize=True,
        )

        # Compute class weights from the training set
        self._class_weights = self.train_dataset.get_class_weights()
        logger.info("Class weights (from train set): %s", self._class_weights.tolist())

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    # ── Public accessors ──────────────────────────────────────────  #

    @property
    def class_weights(self) -> torch.Tensor:
        """Returns class weights (available after setup())."""
        if self._class_weights is None:
            raise RuntimeError("Call setup() before accessing class_weights.")
        return self._class_weights


# ── Module-level helpers ─────────────────────────────────────────── #


def _jitter_rgb_bands(tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies a lightweight brightness/contrast jitter to the first 3 bands
    (RGB surrogate) of a multi-band tensor. The NIR band is kept unchanged.

    Args:
        tensor: Float32 tensor of shape (C, H, W) where C >= 3.

    Returns:
        Jittered tensor of the same shape.
    """
    factor_b = 1.0 + (torch.rand(1).item() - 0.5) * 0.4  # brightness [-0.2, 0.2]
    factor_c = 1.0 + (torch.rand(1).item() - 0.5) * 0.4  # contrast

    out = tensor.clone()
    for i in range(min(3, tensor.shape[0])):
        band = out[i]
        # Contrast: scale around mean
        mean = band.mean()
        out[i] = (band - mean) * factor_c + mean * factor_b

    return torch.clamp(out, min=tensor.min(), max=tensor.max())
