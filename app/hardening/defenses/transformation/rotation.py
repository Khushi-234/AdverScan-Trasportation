"""
Random Rotation Transformation Defense for Module 7 (Hardening).

Applies small random rotational transformations using differentiable affine grid sampling,
preserving spatial dimensions and attenuating directional adversarial gradients.
"""

import math
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


class RotationDefense(BaseDefense):
    """
    Rotation Defense.

    Applies a random spatial rotation in [-max_angle, max_angle] degrees
    via affine grid sampling, strictly preserving required tensor dimensions.
    """

    supported_domains: List[str] = ["image"]
    defense_family: str = "transformation"

    def __init__(
        self,
        max_angle: float = 15.0,
        padding_mode: str = "reflection",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize RotationDefense.

        Args:
            max_angle: Maximum absolute rotation angle in degrees.
            padding_mode: Padding mode for sampling ('reflection', 'zeros', 'border').
            config: Optional configuration dictionary.
        """
        cfg = config or {}
        super().__init__(
            name="rotation",
            defense_type="transformation",
            config=cfg,
            supported_domains=self.supported_domains,
            defense_family=self.defense_family,
        )
        self.max_angle = float(cfg.get("max_angle", max_angle))
        self.padding_mode = str(cfg.get("padding_mode", padding_mode)).lower().strip()

        if self.max_angle < 0.0:
            raise HardeningConfigurationError(f"max_angle must be non-negative, got {self.max_angle}")

    def transform(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply random rotation preserving input dimensions."""
        if not isinstance(inputs, torch.Tensor):
            raise HardeningConfigurationError(f"Expected torch.Tensor, got {type(inputs)}")
        if inputs.ndim != 4:
            return inputs.clone()

        if self.max_angle == 0.0:
            return inputs.clone()

        b = inputs.size(0)
        device = inputs.device
        dtype = inputs.dtype

        # Sample independent rotation angle per batch item (or single angle across batch)
        angles_deg = [random.uniform(-self.max_angle, self.max_angle) for _ in range(b)]
        angles_rad = [deg * math.pi / 180.0 for deg in angles_deg]

        theta = torch.zeros(b, 2, 3, device=device, dtype=dtype)
        for i, rad in enumerate(angles_rad):
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            theta[i, 0, 0] = cos_a
            theta[i, 0, 1] = -sin_a
            theta[i, 1, 0] = sin_a
            theta[i, 1, 1] = cos_a

        grid = F.affine_grid(theta, inputs.size(), align_corners=False)
        rotated = F.grid_sample(
            inputs,
            grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=False,
        )
        return rotated

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
                    "max_angle": self.max_angle,
                    "padding_mode": self.padding_mode,
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
                    f"Applied Random Rotation defense (max_angle=+/-{self.max_angle:.1f} deg)."
                ],
            )
        except Exception as e:
            if isinstance(e, HardeningConfigurationError):
                raise
            raise DefenseExecutionError(f"RotationDefense failed: {str(e)}") from e
