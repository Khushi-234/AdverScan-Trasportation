"""
Preprocessing defenses package for Module 7 (Hardening).
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from app.hardening.defenses.base import BaseDefense
from app.hardening.defenses.wrapper import HardenedModelWrapper
from app.hardening.defenses.preprocessing.normalization import (
    NormalizationDefense,
)
from app.hardening.defenses.preprocessing.gaussian_filter import (
    GaussianFilterDefense,
    SpatialSmoothingDefense,
    apply_spatial_smoothing,
)
from app.hardening.defenses.preprocessing.median_filter import (
    MedianFilterDefense,
    apply_median_filter_2d,
)
from app.hardening.defenses.preprocessing.image_denoising import (
    ImageDenoisingDefense,
)
from app.hardening.defenses.preprocessing.feature_squeezing import (
    FeatureSqueezingDefense,
    reduce_bit_depth,
)
from app.hardening.hardening_result import HardeningMetadata, HardeningResult
from app.hardening.exceptions import DefenseExecutionError, HardeningConfigurationError
import torch.nn.functional as F


def simulate_jpeg_compression(
    inputs: torch.Tensor,
    quality: int = 75,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> torch.Tensor:
    """
    Simulate JPEG compression artefacts using block-wise discrete cosine transform (DCT) thresholding / quantization.

    Args:
        inputs: Input image tensor (B, C, H, W).
        quality: JPEG compression quality factor (1 to 100). Lower quality = higher compression defense.
        clip_min: Min value clip bound.
        clip_max: Max value clip bound.

    Returns:
        torch.Tensor: Compressed/preprocessed image tensor.
    """
    if inputs.ndim != 4:
        return inputs

    # Approximate lossy quantization scale factor based on quality
    quality = max(1, min(100, quality))
    scale = 50.0 / quality if quality < 50 else (200.0 - 2.0 * quality) / 100.0

    # Downsample high frequency noise via average pooling and bilinear upsampling
    block_size = 2 if quality > 70 else (4 if quality > 30 else 8)
    h, w = inputs.shape[2], inputs.shape[3]

    downsampled = F.adaptive_avg_pool2d(inputs, (max(1, h // block_size), max(1, w // block_size)))
    reconstructed = F.interpolate(downsampled, size=(h, w), mode="bilinear", align_corners=False)

    # Mix original and downsampled according to quality factor
    alpha = min(1.0, max(0.2, quality / 100.0))
    compressed = alpha * inputs + (1.0 - alpha) * reconstructed

    return torch.clamp(compressed, min=clip_min, max=clip_max)


class JPEGCompressionDefense(BaseDefense):
    """
    JPEG Compression Defensive Preprocessing (backward compatible).
    """

    supported_domains: List[str] = ["image"]
    defense_family: str = "preprocessing"

    def __init__(
        self,
        quality: int = 75,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = config or {}
        super().__init__(
            name="jpeg_compression",
            defense_type="preprocessing",
            config=cfg,
            supported_domains=self.supported_domains,
            defense_family=self.defense_family,
        )
        self.quality = int(cfg.get("quality", quality))
        self.clip_min = float(cfg.get("clip_min", clip_min))
        self.clip_max = float(cfg.get("clip_max", clip_max))

    def preprocess_tensor(self, inputs: torch.Tensor) -> torch.Tensor:
        return simulate_jpeg_compression(inputs, quality=self.quality, clip_min=self.clip_min, clip_max=self.clip_max)

    def apply(
        self,
        model: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> HardeningResult:
        start_time = time.time()
        try:
            hardened_inputs = self.preprocess_tensor(inputs) if inputs is not None else None
            self.mark_hardened_input(hardened_inputs, self.preprocess_tensor)
            wrapped_model = HardenedModelWrapper(model, self.preprocess_tensor, defense_name=self.name)
            exec_time = time.time() - start_time
            meta = HardeningMetadata(
                defense_name=self.name,
                defense_type=self.defense_type,
                parameters={"quality": self.quality, "clip_min": self.clip_min, "clip_max": self.clip_max},
                execution_time_seconds=exec_time,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                extra_metadata={"defense_family": self.defense_family},
            )
            return HardeningResult(
                hardened_model=wrapped_model,
                metadata=meta,
                hardened_inputs=hardened_inputs,
                success=True,
                recommendations=[f"Evaluated model with JPEG compression quality={self.quality} defense."],
            )
        except Exception as e:
            raise DefenseExecutionError(f"JPEGCompressionDefense failed: {str(e)}") from e


class PreprocessingDefense(BaseDefense):
    """
    Unified Preprocessing coordinator capable of chaining multiple preprocessing steps.
    """

    supported_domains: List[str] = ["image"]
    defense_family: str = "preprocessing"

    def __init__(
        self,
        methods: Optional[List[str]] = None,
        kernel_size: int = 3,
        sigma: float = 1.0,
        bit_depth: int = 4,
        jpeg_quality: int = 75,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = config or {}
        super().__init__(
            name="preprocessing",
            defense_type="preprocessing",
            config=cfg,
            supported_domains=self.supported_domains,
            defense_family=self.defense_family,
        )
        self.methods = methods or ["spatial_smoothing"]
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.bit_depth = bit_depth
        self.jpeg_quality = jpeg_quality

    def preprocess_tensor(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        for method in self.methods:
            m = method.lower().strip()
            if m in ("spatial_smoothing", "gaussian_filter"):
                x = apply_spatial_smoothing(x, kernel_size=self.kernel_size, sigma=self.sigma)
            elif m == "feature_squeezing":
                x = reduce_bit_depth(x, bit_depth=self.bit_depth)
            elif m in ("jpeg_compression", "jpeg"):
                x = simulate_jpeg_compression(x, quality=self.jpeg_quality)
            elif m == "median_filter":
                x = apply_median_filter_2d(x, kernel_size=self.kernel_size)
            else:
                raise HardeningConfigurationError(f"Unknown preprocessing method: '{method}'")
        return x

    def apply(
        self,
        model: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> HardeningResult:
        start_time = time.time()
        try:
            hardened_inputs = self.preprocess_tensor(inputs) if inputs is not None else None
            self.mark_hardened_input(hardened_inputs, self.preprocess_tensor)
            wrapped_model = HardenedModelWrapper(model, self.preprocess_tensor, defense_name=self.name)
            exec_time = time.time() - start_time
            meta = HardeningMetadata(
                defense_name=self.name,
                defense_type=self.defense_type,
                parameters={
                    "methods": self.methods,
                    "kernel_size": self.kernel_size,
                    "sigma": self.sigma,
                    "bit_depth": self.bit_depth,
                    "jpeg_quality": self.jpeg_quality,
                },
                execution_time_seconds=exec_time,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                extra_metadata={"defense_family": self.defense_family},
            )
            return HardeningResult(
                hardened_model=wrapped_model,
                metadata=meta,
                hardened_inputs=hardened_inputs,
                success=True,
                recommendations=[f"Applied preprocessing pipeline ({', '.join(self.methods)}) to model inputs."],
            )
        except Exception as e:
            if isinstance(e, HardeningConfigurationError):
                raise
            raise DefenseExecutionError(f"PreprocessingDefense failed: {str(e)}") from e


__all__ = [
    "NormalizationDefense",
    "GaussianFilterDefense",
    "SpatialSmoothingDefense",
    "MedianFilterDefense",
    "apply_median_filter_2d",
    "ImageDenoisingDefense",
    "FeatureSqueezingDefense",
    "JPEGCompressionDefense",
    "simulate_jpeg_compression",
    "PreprocessingDefense",
]
