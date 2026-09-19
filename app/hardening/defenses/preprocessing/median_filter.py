"""
Median Filter Preprocessing Defense for Module 7 (Hardening).

Applies median filtering to input image tensors to suppress localized impulse-like
and salt-and-pepper adversarial perturbations.
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


def apply_median_filter_2d(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Apply 2D spatial median filtering across all channels of an image tensor.

    Args:
        x: Input image tensor (B, C, H, W).
        kernel_size: Odd integer filter size.

    Returns:
        Filtered tensor (B, C, H, W) matching original shape.
    """
    pad = kernel_size // 2
    b, c, h, w = x.shape
    # Reflect pad spatially
    x_padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")

    # Unfold extracts sliding local windows
    # patches shape: (B, C * kernel_size * kernel_size, H * W)
    patches = F.unfold(x_padded, kernel_size=(kernel_size, kernel_size), stride=1)
    # Reshape to (B, C, kernel_size * kernel_size, H, W)
    patches = patches.view(b, c, kernel_size * kernel_size, h, w)
    # Median along window elements dimension
    median_val = patches.median(dim=2).values
    return median_val


class MedianFilterDefense(BaseDefense):
    """
    Median Filtering Defense.

    Removes impulse-like or salt-and-pepper pixel perturbations by taking
    the median in a local neighborhood window.
    """

    supported_domains: List[str] = ["image"]
    defense_family: str = "preprocessing"

    def __init__(
        self,
        kernel_size: int = 3,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize MedianFilterDefense.

        Args:
            kernel_size: Positive odd integer (e.g. 3, 5).
            config: Optional configuration dictionary.
        """
        cfg = config or {}
        super().__init__(
            name="median_filter",
            defense_type="preprocessing",
            config=cfg,
            supported_domains=self.supported_domains,
            defense_family=self.defense_family,
        )
        self.kernel_size = int(cfg.get("kernel_size", kernel_size))

        if self.kernel_size % 2 == 0 or self.kernel_size < 1:
            raise HardeningConfigurationError(f"kernel_size must be a positive odd integer, got {self.kernel_size}")

    def filter(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Apply median filter to input tensor.
        """
        if not isinstance(inputs, torch.Tensor):
            raise HardeningConfigurationError(f"Expected torch.Tensor, got {type(inputs)}")
        if inputs.ndim != 4:
            return inputs.clone()
        return apply_median_filter_2d(inputs, kernel_size=self.kernel_size)

    def apply(
        self,
        model: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> HardeningResult:
        start_time = time.time()
        try:
            hardened_inputs = self.filter(inputs) if inputs is not None else None
            self.mark_hardened_input(hardened_inputs, self.filter)
            wrapped_model = HardenedModelWrapper(model, self.filter, defense_name=self.name)

            exec_time = time.time() - start_time
            meta = HardeningMetadata(
                defense_name=self.name,
                defense_type=self.defense_type,
                parameters={"kernel_size": self.kernel_size},
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
                    f"Applied Median Filter defense (kernel_size={self.kernel_size})."
                ],
            )
        except Exception as e:
            if isinstance(e, HardeningConfigurationError):
                raise
            raise DefenseExecutionError(f"MedianFilterDefense failed: {str(e)}") from e
