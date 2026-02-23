"""
dataset.py
==========
Custom PyTorch Dataset for Sentinel-2 image patches exported from ArcGIS Pro
(ESRI Classified Tiles format).

Dataset structure:
    data_root/
        images/   ← 4-band 16-bit GeoTIFF patches (*.tif)
        labels/   ← single-band GeoTIFF semantic masks (1=Non-landslide, 2=Landslide)

Patch-level label is derived from the **dominant class** in the mask:
    • If ≥50% of non-zero pixels are class 2 → Landslide  (class_idx = 1)
    • Else → Non-landslide                                 (class_idx = 0)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Band statistics from ESRI EMD (used for per-band normalization)    #
# ------------------------------------------------------------------ #
BAND_STATS = {
    "mean": [722.26, 762.28, 589.94, 2166.67],  # Red, Green, Blue, NIR
    "std": [325.39, 231.94, 237.08, 533.91],
}

# Class mapping (internal 0-indexed labels)
CLASS_NAMES = ["Non-landslide", "Landslide"]
ESRI_CLASS_VALUES = {1: 0, 2: 1}  # ESRI value → internal idx


# ------------------------------------------------------------------ #
#  Helper: read a label TIF and return patch-level class              #
# ------------------------------------------------------------------ #
def get_patch_label(label_path: Path) -> int:
    """
    Derives a single patch-level binary label from a semantic mask TIF.

    Args:
        label_path: Path to single-band label GeoTIFF.

    Returns:
        0 → Non-landslide, 1 → Landslide
    """
    with rasterio.open(label_path) as src:
        mask = src.read(1).astype(np.int32)

    # Count pixels per ESRI class (ignore nodata zeros)
    valid = mask[mask > 0]
    if valid.size == 0:
        return 0  # default to non-landslide if no valid pixels

    landslide_ratio = np.mean(valid == 2)  # fraction of landslide pixels
    return int(landslide_ratio >= 0.5)


# ------------------------------------------------------------------ #
#  Helper: build a DataFrame of (image_path, label_path, class_idx)  #
# ------------------------------------------------------------------ #
def build_dataframe(data_root: str | Path) -> pd.DataFrame:
    """
    Scans data_root/images/ and data_root/labels/ to build a paired
    DataFrame with patch-level class labels.

    Args:
        data_root: Root directory containing 'images/' and 'labels/' sub-dirs.

    Returns:
        pd.DataFrame with columns:
            image_path  : str path to image TIF
            label_path  : str path to label TIF
            class_idx   : int (0 or 1)
            class_name  : str
    """
    data_root = Path(data_root)
    images_dir = data_root / "images"
    labels_dir = data_root / "labels"

    image_files = sorted(images_dir.glob("*.tif"))
    if not image_files:
        raise FileNotFoundError(f"No .tif files found in {images_dir}")

    records: List[Dict] = []
    for img_path in image_files:
        lbl_path = labels_dir / img_path.name
        if not lbl_path.exists():
            logger.warning(f"Label not found for {img_path.name}, skipping.")
            continue

        class_idx = get_patch_label(lbl_path)
        records.append(
            {
                "image_path": str(img_path),
                "label_path": str(lbl_path),
                "class_idx": class_idx,
                "class_name": CLASS_NAMES[class_idx],
            }
        )

    df = pd.DataFrame(records)
    logger.info(
        "Dataset built: %d total | %d Non-landslide | %d Landslide",
        len(df),
        (df["class_idx"] == 0).sum(),
        (df["class_idx"] == 1).sum(),
    )
    return df


# ------------------------------------------------------------------ #
#  PyTorch Dataset                                                     #
# ------------------------------------------------------------------ #
class LandslideDataset(Dataset):
    """
    PyTorch Dataset for Sentinel-2 landslide patches.

    Each item returns a (tensor, label) pair:
        • tensor  : Float32 tensor of shape (C, H, W) where C = num_bands
        • label   : Long scalar tensor (0 or 1)

    Args:
        dataframe    : DataFrame from build_dataframe()
        transform    : torchvision transform pipeline (training/validation)
        num_bands    : Number of spectral bands to load (default 4)
        normalize    : Apply per-band z-score normalization using BAND_STATS
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: Optional[transforms.Compose] = None,
        num_bands: int = 4,
        normalize: bool = True,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.num_bands = num_bands
        self.normalize = normalize

        # Build normalization transform
        if normalize:
            mean = BAND_STATS["mean"][:num_bands]
            std = BAND_STATS["std"][:num_bands]
            self._norm = transforms.Normalize(mean=mean, std=std)
        else:
            self._norm = None

    # ── dunders ────────────────────────────────────────────────────── #

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        # ── Read image ──────────────────────────────────────────────
        image = self._load_image(row["image_path"])  # (C, H, W) float32

        # ── Per-band normalization (before spatial transforms) ───────
        if self._norm is not None:
            image = self._norm(image)

        # ── Spatial / photometric transforms ────────────────────────
        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(row["class_idx"], dtype=torch.long)
        return image, label

    # ── Private helpers ─────────────────────────────────────────────  #

    def _load_image(self, path: str) -> torch.Tensor:
        """
        Loads a multi-band GeoTIFF and returns a float32 tensor in (C, H, W).
        16-bit DN values are scaled to [0, 1] by dividing by 10000 (S2 typical).
        """
        with rasterio.open(path) as src:
            bands = src.read(list(range(1, self.num_bands + 1)))  # (C, H, W)

        img = bands.astype(np.float32) / 10000.0  # scale to ~[0, 1]
        return torch.from_numpy(img)

    # ── Class utilities ─────────────────────────────────────────────  #

    def get_class_weights(self) -> torch.Tensor:
        """
        Computes inverse-frequency class weights for WeightedCrossEntropyLoss.

        Returns:
            Tensor of shape (num_classes,)
        """
        counts = self.df["class_idx"].value_counts().sort_index()
        n_total = len(self.df)
        weights = torch.tensor(
            [n_total / (len(counts) * c) for c in counts], dtype=torch.float32
        )
        return weights

    def get_rgb_image(self, idx: int) -> np.ndarray:
        """
        Returns uint8 RGB (3-channel) image for display purposes.
        Uses bands 0 (R), 1 (G), 2 (B).
        """
        row = self.df.iloc[idx]
        with rasterio.open(row["image_path"]) as src:
            rgb = src.read([1, 2, 3])  # (3, H, W) uint16

        # Percentile-clip + scale to 0-255
        p2, p98 = np.percentile(rgb, [2, 98])
        rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0, 1)
        rgb = (rgb * 255).astype(np.uint8)
        return rgb.transpose(1, 2, 0)  # (H, W, 3)
