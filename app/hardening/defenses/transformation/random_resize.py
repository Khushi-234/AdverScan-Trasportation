"""
Random Resizing Transformation Defense for Module 7 (Hardening).

Applies random spatial scaling to input images prior to inference and restores
expected model input dimensions, disrupting perturbation grids.
"""

import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.hardening.defenses.base import BaseDefense
from app.hardening.defenses.wrapper import HardenedModelWrapper
from app.hardening.hardening_result import HardeningMetadata, HardeningResult
from app.hardening.exceptions import DefenseExecutionError, HardeningConfigurationError


class RandomResizeDefense(BaseDefense):
    """
    Random Resizing Transformation Defense.

    Randomly scales image tensors within a configurable scale factor range [min_scale, max_scale],
    then restores the original spatial dimensions via bilinear interpolation.
    """

    supported_domains: List[str] = ["image"]
    defense_family: str = "transformation"

    def __init__(
        self,
        min_scale: float = 0.85,
        max_scale: float = 1.15,
        interpolation: str = "bilinear",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize RandomResizeDefense.

        Args:
            min_scale: Lower bound scaling factor (e.g. 0.85).
            max_scale: Upper bound scaling factor (e.g. 1.15).
            interpolation: Interpolation mode ('bilinear', 'nearest', 'bicubic').
            config: Optional configuration dictionary.
        """
        cfg = config or {}
        super().__init__(
            name="random_resize",
            defense_type="transformation",
            config=cfg,
            supported_domains=self.supported_domains,
            defense_family=self.defense_family,
        )
        self.min_scale = float(cfg.get("min_scale", min_scale))
        self.max_scale = float(cfg.get("max_scale", max_scale))
        self.interpolation = str(cfg.get("interpolation", interpolation)).lower().strip()

        if self.min_scale <= 0.0 or self.max_scale <= 0.0:
            raise HardeningConfigurationError("Scales must be positive numbers.")
        if self.min_scale > self.max_scale:
            raise HardeningConfigurationError(f"min_scale ({self.min_scale}) cannot exceed max_scale ({self.max_scale})")

    def transform(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply random resizing transform without modifying input in-place."""
        if not isinstance(inputs, torch.Tensor):
            raise HardeningConfigurationError(f"Expected torch.Tensor, got {type(inputs)}")
        if inputs.ndim != 4:
            return inputs.clone()

        b, c, h, w = inputs.shape
        scale = random.uniform(self.min_scale, self.max_scale)
        target_h = max(1, int(round(h * scale)))
        target_w = max(1, int(round(w * scale)))

        x = inputs.clone()
        # Scale to intermediate random dimension
        align_corners = False if self.interpolation in ("bilinear", "bicubic") else None
        
        if self.interpolation in ("bilinear", "bicubic"):
            scaled = F.interpolate(
                x,
                size=(target_h, target_w),
                mode=self.interpolation,
                align_corners=False,
            )
        else:
            scaled = F.interpolate(
                x,
                size=(target_h, target_w),
                mode=self.interpolation,
            )
        # Restore original required dimensions (H, W)
        restored = F.interpolate(
            scaled,
            size=(h, w),
            mode=self.interpolation,
            align_corners=align_corners,
        )
        return restored

    def apply(
        self,
        model: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> HardeningResult:
        start_time = time.time()
        try:
            hardened_inputs = self.transform(inputs) if inputs is not None else None
            self.mark_hardened_input(hardened_inputs, self.transform)
            wrapped_model = HardenedModelWrapper(model, self.transform, defense_name=self.name)

            exec_time = time.time() - start_time
            meta = HardeningMetadata(
                defense_name=self.name,
                defense_type=self.defense_type,
                parameters={
                    "min_scale": self.min_scale,
                    "max_scale": self.max_scale,
                    "interpolation": self.interpolation,
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
                    f"Applied Random Resize defense (scale in [{self.min_scale:.2f}, {self.max_scale:.2f}])."
                ],
            )
        except Exception as e:
            if isinstance(e, HardeningConfigurationError):
                raise
            raise DefenseExecutionError(f"RandomResizeDefense failed: {str(e)}") from e
