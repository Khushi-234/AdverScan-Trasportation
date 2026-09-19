"""
Random Translation Transformation Defense for Module 7 (Hardening).

Applies small spatial translations to input images via differentiable grid sampling,
preserving spatial dimensions and disrupting localized adversarial perturbations.
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


class TranslationDefense(BaseDefense):
    """
    Translation Defense.

    Applies small random horizontal and vertical translations within [-max_dx, max_dx]
    and [-max_dy, max_dy] normalized coordinates [-1, 1], strictly preserving input dimensions.
    """

    supported_domains: List[str] = ["image"]
    defense_family: str = "transformation"

    def __init__(
        self,
        max_dx: float = 0.1,
        max_dy: float = 0.1,
        padding_mode: str = "reflection",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize TranslationDefense.

        Args:
            max_dx: Maximum fractional horizontal shift in normalized coordinates [-1, 1].
            max_dy: Maximum fractional vertical shift in normalized coordinates [-1, 1].
            padding_mode: Padding mode for border sampling ('reflection', 'zeros', 'border').
            config: Optional configuration dictionary.
        """
        cfg = config or {}
        super().__init__(
            name="translation",
            defense_type="transformation",
            config=cfg,
            supported_domains=self.supported_domains,
            defense_family=self.defense_family,
        )
        self.max_dx = float(cfg.get("max_dx", max_dx))
        self.max_dy = float(cfg.get("max_dy", max_dy))
        self.padding_mode = str(cfg.get("padding_mode", padding_mode)).lower().strip()

        if self.max_dx < 0.0 or self.max_dy < 0.0:
            raise HardeningConfigurationError("max_dx and max_dy must be non-negative.")

    def transform(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply random translation preserving input dimensions."""
        if not isinstance(inputs, torch.Tensor):
            raise HardeningConfigurationError(f"Expected torch.Tensor, got {type(inputs)}")
        if inputs.ndim != 4:
            return inputs.clone()

        if self.max_dx == 0.0 and self.max_dy == 0.0:
            return inputs.clone()

        b = inputs.size(0)
        device = inputs.device
        dtype = inputs.dtype

        theta = torch.zeros(b, 2, 3, device=device, dtype=dtype)
        for i in range(b):
            tx = random.uniform(-self.max_dx, self.max_dx)
            ty = random.uniform(-self.max_dy, self.max_dy)
            theta[i, 0, 0] = 1.0
            theta[i, 0, 1] = 0.0
            theta[i, 0, 2] = tx
            theta[i, 1, 0] = 0.0
            theta[i, 1, 1] = 1.0
            theta[i, 1, 2] = ty

        grid = F.affine_grid(theta, inputs.size(), align_corners=False)
        translated = F.grid_sample(
            inputs,
            grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=False,
        )
        return translated

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
                    "max_dx": self.max_dx,
                    "max_dy": self.max_dy,
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
                    f"Applied Random Translation defense (dx=+/-{self.max_dx:.2f}, dy=+/-{self.max_dy:.2f})."
                ],
            )
        except Exception as e:
            if isinstance(e, HardeningConfigurationError):
                raise
            raise DefenseExecutionError(f"TranslationDefense failed: {str(e)}") from e
