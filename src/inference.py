"""
inference.py
============
Inference engine for the landslide detection model.

Provides:
    • LandslideInferencer         – Main inference class
        predict_single()          – Predict a single image TIF
        predict_batch()           – Predict all TIFs in a directory
        predict_with_gradcam()    – Predict + overlay Grad-CAM heatmap
    • load_inferencer_from_ckpt() – Factory to instantiate from a checkpoint path

All predictions return landslide probability alongside the class label so the
user can apply a custom threshold if needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from src.dataset import BAND_STATS, LandslideDataset, build_dataframe
from src.lightning_module import LandslideClassifier

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Non-landslide", "Landslide"]


class LandslideInferencer:
    """
    High-level inference wrapper for the trained landslide classifier.

    Args:
        ckpt_path   : Path to PyTorch Lightning checkpoint (.ckpt).
        model       : Pre-instantiated nn.Module (if not loading from ckpt).
        device      : 'cuda', 'cpu', or 'auto' (default).
        num_bands   : Number of spectral bands (must match model training config).
        threshold   : Default classification threshold for positive class.
    """

    def __init__(
        self,
        ckpt_path: Optional[str | Path] = None,
        model: Optional[nn.Module] = None,
        device: str = "auto",
        num_bands: int = 4,
        threshold: float = 0.5,
    ) -> None:
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.num_bands = num_bands
        self.threshold = threshold

        # ── Load model ───────────────────────────────────────────────
        if ckpt_path is not None:
            self.lightning_module = LandslideClassifier.load_from_checkpoint(
                str(ckpt_path), map_location=self.device, strict=False
            )
            self.net = self.lightning_module.model
        elif model is not None:
            self.net = model
            self.lightning_module = None
        else:
            raise ValueError("Provide either ckpt_path or model.")

        self.net.to(self.device)
        self.net.eval()

        # ── Normalization ────────────────────────────────────────────
        from torchvision.transforms import Normalize, CenterCrop, Compose

        mean = BAND_STATS["mean"][:num_bands]
        std = BAND_STATS["std"][:num_bands]
        self._preprocess = Compose(
            [
                Normalize(mean=mean, std=std),
                CenterCrop(224),
            ]
        )

        logger.info(
            "LandslideInferencer ready on %s | threshold=%.2f",
            self.device,
            self.threshold,
        )

    # ── Single image prediction ──────────────────────────────────── #

    def predict_single(
        self,
        image_path: str | Path,
        threshold: Optional[float] = None,
    ) -> Dict[str, Union[str, float, int]]:
        """
        Predict the class of a single image TIF patch.

        Args:
            image_path : Path to 4-band GeoTIFF image patch.
            threshold  : Override the default classification threshold.

        Returns:
            Dict with keys:
                'image_path'  – str path
                'probability' – float probability of Landslide class
                'class_idx'   – int (0 or 1)
                'class_name'  – str ('Non-landslide' or 'Landslide')
                'threshold'   – float threshold used
        """
        thresh = threshold if threshold is not None else self.threshold
        tensor = self._load_and_preprocess(image_path)  # (1, C, H, W)

        with torch.no_grad():
            logits = self.net(tensor)  # (1, 2)
            prob = torch.softmax(logits, dim=1)[0, 1].item()

        class_idx = int(prob >= thresh)
        return {
            "image_path": str(image_path),
            "probability": round(prob, 4),
            "class_idx": class_idx,
            "class_name": CLASS_NAMES[class_idx],
            "threshold": thresh,
        }

    # ── Batch prediction ─────────────────────────────────────────── #

    def predict_batch(
        self,
        image_dir: str | Path,
        threshold: Optional[float] = None,
        recursive: bool = False,
    ) -> pd.DataFrame:
        """
        Predict classes for all .tif images in a directory.

        Args:
            image_dir : Directory containing image TIF files.
            threshold : Override the default classification threshold.
            recursive : If True, search sub-directories recursively.

        Returns:
            pd.DataFrame with one row per image:
                columns – image_path, probability, class_idx, class_name, threshold
        """
        image_dir = Path(image_dir)
        pattern = "**/*.tif" if recursive else "*.tif"
        image_files = sorted(image_dir.glob(pattern))

        if not image_files:
            logger.warning("No .tif files found in %s", image_dir)
            return pd.DataFrame()

        results = []
        for img_path in image_files:
            try:
                res = self.predict_single(img_path, threshold=threshold)
                results.append(res)
            except Exception as exc:
                logger.warning("Failed to predict %s: %s", img_path, exc)

        df = pd.DataFrame(results)
        logger.info(
            "Batch inference complete: %d images | %d Landslide | %d Non-landslide",
            len(df),
            (df["class_idx"] == 1).sum(),
            (df["class_idx"] == 0).sum(),
        )
        return df

    # ── Grad-CAM visualization ────────────────────────────────────── #

    def predict_with_gradcam(
        self,
        image_path: str | Path,
        threshold: Optional[float] = None,
        save_path: Optional[str | Path] = None,
    ) -> Tuple[Dict, np.ndarray]:
        """
        Predict class AND generate a Grad-CAM saliency overlay.

        Args:
            image_path : Path to 4-band GeoTIFF image patch.
            threshold  : Override default classification threshold.
            save_path  : If provided, saves the overlay PNG to this path.

        Returns:
            Tuple of (prediction_dict, overlay_rgb_array).
            overlay_rgb_array has shape (H, W, 3), dtype uint8.
        """
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_gradcam_on_image
        except ImportError:
            raise ImportError("Install grad-cam: pip install grad-cam")

        thresh = threshold if threshold is not None else self.threshold

        # ── Identify target layer ─────────────────────────────────
        if self.lightning_module is not None:
            target_layer = self.lightning_module.get_gradcam_target_layer()
        else:
            target_layer = self._auto_detect_target_layer()

        if target_layer is None:
            raise RuntimeError("Could not identify a target layer for Grad-CAM.")

        # ── Prepare input ──────────────────────────────────────────
        tensor = self._load_and_preprocess(image_path)  # (1, C, H, W)

        # ── Prediction ──────────────────────────────────────────────
        with torch.no_grad():
            logits = self.net(tensor)
            prob = torch.softmax(logits, dim=1)[0, 1].item()
        class_idx = int(prob >= thresh)

        # ── Grad-CAM ────────────────────────────────────────────────
        cam = GradCAM(model=self.net, target_layers=[target_layer])
        # Target class for CAM: always landslide class (idx=1)
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        targets = [ClassifierOutputTarget(1)]
        grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]  # (H, W)

        # Build RGB background from first 3 bands (percentile-clipped)
        import rasterio

        with rasterio.open(image_path) as src:
            rgb = src.read([1, 2, 3]).astype(np.float32)
        p2, p98 = np.percentile(rgb, [2, 98])
        rgb_norm = np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0, 1)
        rgb_norm = rgb_norm.transpose(1, 2, 0)  # (H, W, 3)

        # Resize CAM to match image spatial dims if needed
        from PIL import Image as PILImage

        if grayscale_cam.shape != (rgb_norm.shape[0], rgb_norm.shape[1]):
            cam_pil = PILImage.fromarray((grayscale_cam * 255).astype(np.uint8))
            cam_pil = cam_pil.resize(
                (rgb_norm.shape[1], rgb_norm.shape[0]), PILImage.BILINEAR
            )
            grayscale_cam = np.array(cam_pil).astype(np.float32) / 255.0

        overlay = show_gradcam_on_image(rgb_norm, grayscale_cam, use_rgb=True)

        if save_path is not None:
            PILImage.fromarray(overlay).save(save_path)
            logger.info("Grad-CAM overlay saved → %s", save_path)

        prediction = {
            "image_path": str(image_path),
            "probability": round(prob, 4),
            "class_idx": class_idx,
            "class_name": CLASS_NAMES[class_idx],
            "threshold": thresh,
        }
        return prediction, overlay

    # ── Private helpers ──────────────────────────────────────────── #

    def _load_and_preprocess(self, image_path: str | Path) -> torch.Tensor:
        """
        Loads a GeoTIFF, scales, normalizes, and returns a batched tensor.

        Returns:
            Tensor of shape (1, C, H, W) on self.device.
        """
        import rasterio

        with rasterio.open(image_path) as src:
            bands = src.read(list(range(1, self.num_bands + 1)))  # (C, H, W)

        img = torch.from_numpy(bands.astype(np.float32) / 10000.0)  # (C, H, W)
        img = self._preprocess(img)  # (C, 224, 224)
        return img.unsqueeze(0).to(self.device)  # (1, C, H, W)

    def _auto_detect_target_layer(self) -> Optional[nn.Module]:
        """Fallback target layer detection when lightning_module is unavailable."""
        if hasattr(self.net, "backbone") and hasattr(self.net.backbone, "layer4"):
            return self.net.backbone.layer4[-1]
        if hasattr(self.net, "features"):
            for layer in reversed(list(self.net.features.children())):
                if isinstance(layer, (nn.Sequential, nn.Conv2d)):
                    return layer
        return None


# ─────────────────────────────────────────────────────────────────── #
#  Convenience factory                                                 #
# ─────────────────────────────────────────────────────────────────── #
def load_inferencer_from_ckpt(
    ckpt_path: str | Path,
    device: str = "auto",
    threshold: float = 0.5,
    num_bands: int = 4,
) -> LandslideInferencer:
    """
    Convenience factory to build a LandslideInferencer from a checkpoint path.

    Example:
        >>> inf = load_inferencer_from_ckpt("outputs/best_model.ckpt")
        >>> result = inf.predict_single("LandslideData/images/000000000000.tif")
        >>> print(result)

    Args:
        ckpt_path  : Path to .ckpt file.
        device     : 'cuda', 'cpu', or 'auto'.
        threshold  : Classification threshold.
        num_bands  : Spectral bands (must match training config).

    Returns:
        LandslideInferencer instance.
    """
    return LandslideInferencer(
        ckpt_path=ckpt_path,
        device=device,
        threshold=threshold,
        num_bands=num_bands,
    )
