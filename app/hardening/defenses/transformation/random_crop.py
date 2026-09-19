"""
Random Cropping Transformation Defense for Module 7 (Hardening).

Applies random spatial cropping to input images and preserves required model input size,
breaking spatial alignment of adversarial perturbations.
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


class RandomCropDefense(BaseDefense):
    """
    Random Cropping Defense.

    Pads input image tensor and extracts a random crop matching the original spatial dimensions,
    thereby translating and shifting pixel alignments while strictly preserving model input size.
    """

    supported_domains: List[str] = ["image"]
    defense_family: str = "transformation"

    def __init__(
        self,
        pad_size: int = 4,
        padding_mode: str = "reflect",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize RandomCropDefense.

        Args:
            pad_size: Number of pixels padded to each border before cropping.
            padding_mode: Padding mode ('reflect', 'replicate', 'constant').
            config: Optional configuration dictionary.
        """
        cfg = config or {}
        super().__init__(
            name="random_crop",
            defense_type="transformation",
            config=cfg,
            supported_domains=self.supported_domains,
            defense_family=self.defense_family,
        )
        self.pad_size = int(cfg.get("pad_size", pad_size))
        self.padding_mode = str(cfg.get("padding_mode", padding_mode)).lower().strip()

        if self.pad_size < 0:
            raise HardeningConfigurationError(f"pad_size must be non-negative, got {self.pad_size}")
        if self.padding_mode not in ("reflect", "replicate", "constant"):
            raise HardeningConfigurationError(
                f"Unsupported padding_mode '{self.padding_mode}'. Supported: ['reflect', 'replicate', 'constant']"
            )

    def transform(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply random crop preserving original input dimensions."""
        if not isinstance(inputs, torch.Tensor):
            raise HardeningConfigurationError(f"Expected torch.Tensor, got {type(inputs)}")
        if inputs.ndim != 4:
            return inputs.clone()

        if self.pad_size == 0:
            return inputs.clone()

        b, c, h, w = inputs.shape
        p = self.pad_size

        # Reflect padding requires input dimensions > pad_size
        mode = self.padding_mode
        if mode == "reflect" and (p >= h or p >= w):
            mode = "replicate"

        x_padded = F.pad(inputs, (p, p, p, p), mode=mode)
        max_y = x_padded.shape[2] - h
        max_x = x_padded.shape[3] - w

        # Extract independent random crops for each image in the batch
        # ensuring stochastic diversity rather than sharing one offset across all samples
        crops = []
        for i in range(b):
            top_i = random.randint(0, max_y)
            left_i = random.randint(0, max_x)
            crops.append(x_padded[i, :, top_i : top_i + h, left_i : left_i + w])

        cropped = torch.stack(crops, dim=0)
        return cropped

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
                    f"Applied Random Crop defense (pad_size={self.pad_size}, mode='{self.padding_mode}')."
                ],
            )
        except Exception as e:
            if isinstance(e, HardeningConfigurationError):
                raise
            raise DefenseExecutionError(f"RandomCropDefense failed: {str(e)}") from e
