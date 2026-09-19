"""
Feature Squeezing Preprocessing Defense for Module 7 (Hardening).

Reduces unnecessary input precision and dynamic range via bit-depth quantization,
stripping away subtle gradient perturbations crafted by adversarial attacks.
Modularly supports image tensors as well as tabular/feature vectors.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from app.hardening.defenses.base import BaseDefense
from app.hardening.defenses.wrapper import HardenedModelWrapper
from app.hardening.hardening_result import HardeningMetadata, HardeningResult
from app.hardening.exceptions import DefenseExecutionError, HardeningConfigurationError


def reduce_bit_depth(
    inputs: torch.Tensor,
    bit_depth: int = 4,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> torch.Tensor:
    """
    Reduce color/feature bit-depth (Feature Squeezing).

    Quantizes tensor values into 2^bit_depth discrete levels within [clip_min, clip_max].

    Args:
        inputs: Input tensor (B, C, H, W) or (B, D).
        bit_depth: Target bit depth (1 to 16).
        clip_min: Minimum bound.
        clip_max: Maximum bound.

    Returns:
        torch.Tensor: Quantized feature squeezed tensor.
    """
    bit_depth = max(1, min(16, int(bit_depth)))
    max_level = (2**bit_depth) - 1

    # Normalize to [0, 1]
    range_val = clip_max - clip_min
    if range_val <= 0:
        return inputs

    normalized = (inputs - clip_min) / range_val
    normalized = torch.clamp(normalized, 0.0, 1.0)

    # Quantize
    quantized = torch.round(normalized * max_level) / max_level

    # Scale back to original range
    squeezed = quantized * range_val + clip_min
    return squeezed


class FeatureSqueezingDefense(BaseDefense):
    """
    Feature Squeezing Defense.

    Quantizes continuous input values to a reduced number of discrete bins (e.g. 2^bit_depth levels),
    suppressing small adversarial noise while preserving dominant structure.
    """

    supported_domains: List[str] = ["image", "tabular"]
    defense_family: str = "preprocessing"

    def __init__(
        self,
        bit_depth: int = 4,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        name: str = "feature_squeezing",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize FeatureSqueezingDefense.

        Args:
            bit_depth: Target bit-depth (1 to 16). Quantizes into 2^bit_depth discrete steps.
            clip_min: Minimum input value bound.
            clip_max: Maximum input value bound.
            name: Defense registration identifier.
            config: Optional configuration dictionary.
        """
        cfg = config or {}
        defense_name = cfg.get("name", name)
        super().__init__(
            name=defense_name,
            defense_type="preprocessing",
            config=cfg,
            supported_domains=self.supported_domains,
            defense_family=self.defense_family,
        )
        self.bit_depth = int(cfg.get("bit_depth", bit_depth))
        self.clip_min = float(cfg.get("clip_min", clip_min))
        self.clip_max = float(cfg.get("clip_max", clip_max))

        if self.bit_depth < 1 or self.bit_depth > 16:
            raise HardeningConfigurationError(f"bit_depth must be between 1 and 16, got {self.bit_depth}")
        if self.clip_min >= self.clip_max:
            raise HardeningConfigurationError(f"clip_min ({self.clip_min}) must be strictly less than clip_max ({self.clip_max})")

    def squeeze(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply bit-depth feature quantization to tensor."""
        if not isinstance(inputs, torch.Tensor):
            raise HardeningConfigurationError(f"Expected torch.Tensor, got {type(inputs)}")
        return reduce_bit_depth(
            inputs,
            bit_depth=self.bit_depth,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
        )

    def apply(
        self,
        model: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> HardeningResult:
        start_time = time.time()
        try:
            hardened_inputs = self.squeeze(inputs) if inputs is not None else None
            self.mark_hardened_input(hardened_inputs, self.squeeze)
            wrapped_model = HardenedModelWrapper(model, self.squeeze, defense_name=self.name)

            exec_time = time.time() - start_time
            meta = HardeningMetadata(
                defense_name=self.name,
                defense_type=self.defense_type,
                parameters={
                    "bit_depth": self.bit_depth,
                    "clip_min": self.clip_min,
                    "clip_max": self.clip_max,
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
                    f"Applied Feature Squeezing ({self.bit_depth}-bit quantization, [{self.clip_min}, {self.clip_max}])."
                ],
            )
        except Exception as e:
            if isinstance(e, HardeningConfigurationError):
                raise
            raise DefenseExecutionError(f"FeatureSqueezingDefense failed: {str(e)}") from e
