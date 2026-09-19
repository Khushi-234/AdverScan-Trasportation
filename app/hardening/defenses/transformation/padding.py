"""
Padding Transformation Defense for Module 7 (Hardening).

Adds configurable padding around image borders and scales back to target dimensions,
shifting feature alignment and mitigating spatial adversarial patterns.
"""

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


class PaddingDefense(BaseDefense):
    """
    Padding Transformation Defense.

    Pads input images with configurable border width and mode, then restores
    the expected input tensor size via bilinear interpolation.
    """

    supported_domains: List[str] = ["image"]
    defense_family: str = "transformation"

    def __init__(
        self,
        pad_size: int = 4,
        padding_mode: str = "reflect",
        fill_value: float = 0.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize PaddingDefense.

        Args:
            pad_size: Number of pixels to pad on each side.
            padding_mode: Padding style ('constant', 'reflect', 'replicate').
            fill_value: Constant value if padding_mode is 'constant'.
            config: Optional configuration dictionary.
        """
        cfg = config or {}
        super().__init__(
            name="padding",
            defense_type="transformation",
            config=cfg,
            supported_domains=self.supported_domains,
            defense_family=self.defense_family,
        )
        self.pad_size = int(cfg.get("pad_size", pad_size))
        self.padding_mode = str(cfg.get("padding_mode", padding_mode)).lower().strip()
        self.fill_value = float(cfg.get("fill_value", fill_value))

        if self.pad_size < 0:
            raise HardeningConfigurationError(f"pad_size must be non-negative, got {self.pad_size}")
        if self.padding_mode not in ("constant", "reflect", "replicate"):
            raise HardeningConfigurationError(
                f"Unsupported padding_mode '{self.padding_mode}'. Supported: ['constant', 'reflect', 'replicate']"
            )

    def transform(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply padding and restore original spatial dimensions."""
        if not isinstance(inputs, torch.Tensor):
            raise HardeningConfigurationError(f"Expected torch.Tensor, got {type(inputs)}")
        if inputs.ndim != 4:
            return inputs.clone()

        if self.pad_size == 0:
            return inputs.clone()

        b, c, h, w = inputs.shape
        p = self.pad_size

        mode = self.padding_mode
        if mode == "reflect" and (p >= h or p >= w):
            mode = "replicate"

        if mode == "constant":
            padded = F.pad(inputs, (p, p, p, p), mode="constant", value=self.fill_value)
        else:
            padded = F.pad(inputs, (p, p, p, p), mode=mode)

        # Interpolate back to required (H, W) to preserve model input size
        restored = F.interpolate(padded, size=(h, w), mode="bilinear", align_corners=False)
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
                    "pad_size": self.pad_size,
                    "padding_mode": self.padding_mode,
                    "fill_value": self.fill_value,
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
                    f"Applied Padding defense (pad_size={self.pad_size}, mode='{self.padding_mode}')."
                ],
            )
        except Exception as e:
            if isinstance(e, HardeningConfigurationError):
                raise
            raise DefenseExecutionError(f"PaddingDefense failed: {str(e)}") from e
