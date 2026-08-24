"""
Defensive Preprocessing Module for Module 7 (Hardening).

This Preprocessing Defense: does not update model weights. 
It tries to remove or weaken adversarial perturbations before the input reaches the model.

Implements input preprocessing defenses:
- Spatial Smoothing (Gaussian blur filtering)
- Bit-Depth Reduction (Feature Squeezing)
- JPEG Compression simulation
- PreprocessedModelWrapper (nn.Module wrapping base model with runtime preprocessing)

Adversarial Image
        ↓
Gaussian Blur / Feature Squeezing / JPEG
        ↓
Reduced adversarial perturbation
        ↓
Original Model
        ↓
Prediction
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

from app.hardening.defenses.base import BaseDefense
from app.hardening.hardening_result import HardeningMetadata, HardeningResult
from app.hardening.utils import (
    apply_spatial_smoothing,
    reduce_bit_depth,
    simulate_jpeg_compression,
)
from app.hardening.exceptions import DefenseExecutionError, HardeningConfigurationError


class PreprocessedModelWrapper(nn.Module):
    """
    PyTorch nn.Module wrapper that applies defensive preprocessing transforms to input tensors
    prior to executing the underlying model's forward pass.
    """

    def __init__(self, model: nn.Module, preprocessing_fn: Any) -> None:
        """
        Initialize wrapper.

        Args:
            model: Original PyTorch model.
            preprocessing_fn: Function taking input tensor and returning preprocessed tensor.
        """
        super().__init__()
        self.model = model
        self.preprocessing_fn = preprocessing_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input tensor and execute underlying model.
        """
        preprocessed_x = self.preprocessing_fn(x)
        return self.model(preprocessed_x)


class SpatialSmoothingDefense(BaseDefense):
    """
    Spatial Smoothing Defensive Preprocessing (Gaussian Filter).
    Removes high-frequency adversarial perturbations.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        sigma: float = 1.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name="spatial_smoothing", defense_type="preprocessing", config=config)
        self.kernel_size = kernel_size
        self.sigma = sigma

    def preprocess_tensor(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply spatial smoothing filter to tensor."""
        return apply_spatial_smoothing(inputs, kernel_size=self.kernel_size, sigma=self.sigma)

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
            wrapped_model = PreprocessedModelWrapper(model, self.preprocess_tensor)

            exec_time = time.time() - start_time
            meta = HardeningMetadata(
                defense_name=self.name,
                defense_type=self.defense_type,
                parameters={"kernel_size": self.kernel_size, "sigma": self.sigma},
                execution_time_seconds=exec_time,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            return HardeningResult(
                hardened_model=wrapped_model,
                metadata=meta,
                hardened_inputs=hardened_inputs,
                success=True,
                recommendations=["Evaluated model with Spatial Smoothing (Gaussian Kernel) defense."],
            )
        except Exception as e:
            raise DefenseExecutionError(f"SpatialSmoothingDefense failed: {str(e)}") from e


class BitDepthReductionDefense(BaseDefense):
    """
    Bit-Depth Reduction Defensive Preprocessing (Feature Squeezing).
    Quantizes features to reduce small gradient adversarial perturbations.
    """

    def __init__(
        self,
        bit_depth: int = 4,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name="bit_depth_reduction", defense_type="preprocessing", config=config)
        self.bit_depth = bit_depth
        self.clip_min = clip_min
        self.clip_max = clip_max

    def preprocess_tensor(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply bit-depth reduction to tensor."""
        return reduce_bit_depth(inputs, bit_depth=self.bit_depth, clip_min=self.clip_min, clip_max=self.clip_max)

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
            wrapped_model = PreprocessedModelWrapper(model, self.preprocess_tensor)

            exec_time = time.time() - start_time
            meta = HardeningMetadata(
                defense_name=self.name,
                defense_type=self.defense_type,
                parameters={"bit_depth": self.bit_depth, "clip_min": self.clip_min, "clip_max": self.clip_max},
                execution_time_seconds=exec_time,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            return HardeningResult(
                hardened_model=wrapped_model,
                metadata=meta,
                hardened_inputs=hardened_inputs,
                success=True,
                recommendations=[f"Evaluated model with {self.bit_depth}-bit Feature Squeezing preprocessing."],
            )
        except Exception as e:
            raise DefenseExecutionError(f"BitDepthReductionDefense failed: {str(e)}") from e


class JPEGCompressionDefense(BaseDefense):
    """
    JPEG Compression Defensive Preprocessing.
    Simulates lossy compression to disrupt adversarial perturbations.
    """

    def __init__(
        self,
        quality: int = 75,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name="jpeg_compression", defense_type="preprocessing", config=config)
        self.quality = quality
        self.clip_min = clip_min
        self.clip_max = clip_max

    def preprocess_tensor(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply JPEG compression simulation to tensor."""
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
            wrapped_model = PreprocessedModelWrapper(model, self.preprocess_tensor)

            exec_time = time.time() - start_time
            meta = HardeningMetadata(
                defense_name=self.name,
                defense_type=self.defense_type,
                parameters={"quality": self.quality, "clip_min": self.clip_min, "clip_max": self.clip_max},
                execution_time_seconds=exec_time,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
    Unified Defensive Preprocessing coordinator capable of running multiple preprocessing steps.
    """

    def __init__(
        self,
        methods: Optional[List[str]] = None,
        kernel_size: int = 3,
        sigma: float = 1.0,
        bit_depth: int = 4,
        jpeg_quality: int = 75,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name="preprocessing", defense_type="preprocessing", config=config)
        self.methods = methods or ["spatial_smoothing"]
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.bit_depth = bit_depth
        self.jpeg_quality = jpeg_quality

    def preprocess_tensor(self, inputs: torch.Tensor) -> torch.Tensor:
        """Sequential execution of specified preprocessing methods."""
        x = inputs
        for method in self.methods:
            m = method.lower().strip()
            if m == "spatial_smoothing":
                x = apply_spatial_smoothing(x, kernel_size=self.kernel_size, sigma=self.sigma)
            elif m in ("bit_depth_reduction", "feature_squeezing"):
                x = reduce_bit_depth(x, bit_depth=self.bit_depth)
            elif m in ("jpeg_compression", "jpeg"):
                x = simulate_jpeg_compression(x, quality=self.jpeg_quality)
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
            wrapped_model = PreprocessedModelWrapper(model, self.preprocess_tensor)

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
