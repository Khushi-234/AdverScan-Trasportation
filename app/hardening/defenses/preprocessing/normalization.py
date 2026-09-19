"""
Input Normalization Preprocessing Defense for Module 7 (Hardening).

Normalizes input tensors using expected channel mean/std or min-max bounds,
ensuring inputs conform to the model's expected range and dynamic distribution.
Can be applied standalone to tensors or wrapped around an nn.Module.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union
import torch
import torch.nn as nn

from app.hardening.defenses.base import BaseDefense
from app.hardening.defenses.wrapper import HardenedModelWrapper
from app.hardening.hardening_result import HardeningMetadata, HardeningResult
from app.hardening.exceptions import DefenseExecutionError, HardeningConfigurationError


class NormalizationDefense(BaseDefense):
    """
    Input Normalization Defense.

    Supports:
    - Channel-wise mean/std normalization (e.g., ImageNet or custom)
    - Min-max scaling / feature standardization
    - Configurable clipping bounds [clip_min, clip_max]
    """

    supported_domains: List[str] = ["image", "tabular", "audio"]
    defense_family: str = "preprocessing"

    def __init__(
        self,
        mean: Optional[Union[float, Sequence[float]]] = None,
        std: Optional[Union[float, Sequence[float]]] = None,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
        norm_type: str = "mean_std",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize NormalizationDefense.

        Args:
            mean: Expected mean per channel (or global float). Defaults to None (or ImageNet if image).
            std: Expected std per channel (or global float).
            clip_min: Optional lower bound clipping.
            clip_max: Optional upper bound clipping.
            norm_type: Normalization type ('mean_std', 'min_max', 'standardize').
            config: Optional configuration dictionary.
        """
        cfg = config or {}
        super().__init__(
            name="normalization",
            defense_type="preprocessing",
            config=cfg,
            supported_domains=self.supported_domains,
            defense_family=self.defense_family,
        )
        self.mean = cfg.get("mean", mean)
        self.std = cfg.get("std", std)
        self.clip_min = cfg.get("clip_min", clip_min)
        self.clip_max = cfg.get("clip_max", clip_max)
        self.norm_type = str(cfg.get("norm_type", norm_type)).lower().strip()

    def normalize(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Normalize input tensor without modifying original tensor in-place.
        """
        if not isinstance(inputs, torch.Tensor):
            raise HardeningConfigurationError(f"Expected torch.Tensor, got {type(inputs)}")

        x = inputs.clone()

        if self.norm_type == "mean_std":
            if self.mean is not None and self.std is not None:
                device = x.device
                dtype = x.dtype

                # Format mean and std tensors matching input dimensions
                if isinstance(self.mean, (int, float)):
                    mean_t = torch.tensor(float(self.mean), device=device, dtype=dtype)
                else:
                    mean_t = torch.tensor(list(self.mean), device=device, dtype=dtype)

                if isinstance(self.std, (int, float)):
                    std_t = torch.tensor(float(self.std), device=device, dtype=dtype)
                else:
                    std_t = torch.tensor(list(self.std), device=device, dtype=dtype)

                # Broadcast to (1, C, 1, 1) if 4D image tensor, or (1, C) if 2D
                if x.ndim == 4 and mean_t.ndim == 1:
                    mean_t = mean_t.view(1, -1, 1, 1)
                    std_t = std_t.view(1, -1, 1, 1)
                elif x.ndim == 2 and mean_t.ndim == 1:
                    mean_t = mean_t.view(1, -1)
                    std_t = std_t.view(1, -1)

                x = (x - mean_t) / (std_t + 1e-8)

        elif self.norm_type == "min_max":
            # Scale each sample to [0, 1] across feature dimensions
            dims = tuple(range(1, x.ndim))
            x_min = x.amin(dim=dims, keepdim=True)
            x_max = x.amax(dim=dims, keepdim=True)
            x = (x - x_min) / (x_max - x_min + 1e-8)

        elif self.norm_type == "standardize":
            # Zero mean, unit variance per sample
            dims = tuple(range(1, x.ndim))
            mean_val = x.mean(dim=dims, keepdim=True)
            std_val = x.std(dim=dims, keepdim=True)
            x = (x - mean_val) / (std_val + 1e-8)

        # Apply clipping if specified
        if self.clip_min is not None or self.clip_max is not None:
            c_min = float(self.clip_min) if self.clip_min is not None else float("-inf")
            c_max = float(self.clip_max) if self.clip_max is not None else float("inf")
            x = torch.clamp(x, min=c_min, max=c_max)

        return x

    def apply(
        self,
        model: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> HardeningResult:
        """
        Apply normalization defense as an input transformation and wrapped model.
        """
        start_time = time.time()
        try:
            hardened_inputs = self.normalize(inputs) if inputs is not None else None
            self.mark_hardened_input(hardened_inputs, self.normalize)
            wrapped_model = HardenedModelWrapper(model, self.normalize, defense_name=self.name)

            exec_time = time.time() - start_time
            meta = HardeningMetadata(
                defense_name=self.name,
                defense_type=self.defense_type,
                parameters={
                    "mean": self.mean,
                    "std": self.std,
                    "clip_min": self.clip_min,
                    "clip_max": self.clip_max,
                    "norm_type": self.norm_type,
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
                recommendations=[
                    f"Applied input normalization (type='{self.norm_type}') wrapper to model inference."
                ],
            )
        except Exception as e:
            if isinstance(e, HardeningConfigurationError):
                raise
            raise DefenseExecutionError(f"NormalizationDefense failed: {str(e)}") from e
