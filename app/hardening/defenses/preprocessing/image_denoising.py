"""
Image Denoising Preprocessing Defense for Module 7 (Hardening).

Provides a modular image denoising implementation capable of supporting multiple
denoising algorithms (Total Variation minimization, bilateral approximation, local mean)
to attenuate subtle adversarial perturbations before model inference.
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


def denoise_total_variation(x: torch.Tensor, weight: float = 0.05, iterations: int = 5) -> torch.Tensor:
    """
    Apply Total Variation (TV) gradient descent smoothing step to image tensor.
    TV denoising preserves sharp edges while smoothing small adversarial noise.
    """

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Input must be a torch.Tensor, got {type(x).__name__}")

    if x.ndim != 4:
        raise ValueError(
            f"Input tensor must be 4D (B, C, H, W), got {x.ndim} dimensions"
        )

    if weight <= 0:
        raise ValueError(f"weight must be positive, got {weight}")

    if iterations < 1 or not isinstance(iterations, int):
        raise ValueError(
            f"iterations must be a positive integer, got {iterations}"
        )
    denoised = x.clone().detach().requires_grad_(True)
    optimizer = torch.optim.SGD([denoised], lr=0.1)

    for _ in range(iterations):
        optimizer.zero_grad()
        # Compute TV penalty: sum of spatial gradients
        diff_h = torch.abs(denoised[:, :, 1:, :] - denoised[:, :, :-1, :]).mean()
        diff_w = torch.abs(denoised[:, :, :, 1:] - denoised[:, :, :, :-1]).mean()
        tv_loss = weight * (diff_h + diff_w)
        fidelity_loss = F.mse_loss(denoised, x)
        loss = fidelity_loss + tv_loss
        loss.backward()
        optimizer.step()

    return denoised.detach().clamp(0.0, 1.0)


def denoise_bilateral_approx(x: torch.Tensor, kernel_size: int = 5, sigma_s: float = 1.5, sigma_r: float = 0.1) -> torch.Tensor:
    """
    Fast bilateral filter approximation using unfolded spatial and range Gaussian weighting.
    """
    b, c, h, w = x.shape
    pad = kernel_size // 2
    x_padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")

    # Spatial Gaussian weights
    y, x_coord = torch.meshgrid(
        torch.arange(-pad, pad + 1, device=x.device, dtype=x.dtype),
        torch.arange(-pad, pad + 1, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    spatial_dist = (x_coord ** 2 + y ** 2)
    spatial_weights = torch.exp(-spatial_dist / (2 * sigma_s ** 2)).view(1, 1, kernel_size * kernel_size, 1, 1)

    # Patches: (B, C, K*K, H, W)
    patches = F.unfold(x_padded, kernel_size=(kernel_size, kernel_size), stride=1).view(b, c, kernel_size * kernel_size, h, w)
    center = x.unsqueeze(2)  # (B, C, 1, H, W)

    # Range weights: intensity difference
    range_weights = torch.exp(-((patches - center) ** 2) / (2 * sigma_r ** 2))
    weights = spatial_weights * range_weights
    weights_sum = weights.sum(dim=2, keepdim=True) + 1e-8

    denoised = (patches * weights).sum(dim=2) / weights_sum.squeeze(2)
    return denoised


def denoise_local_mean(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Local mean box filtering.
    """
    pad = kernel_size // 2
    b, c, h, w = x.shape
    weight = torch.ones((c, 1, kernel_size, kernel_size), device=x.device, dtype=x.dtype) / (kernel_size * kernel_size)
    return F.conv2d(x, weight, padding=pad, groups=c)


class ImageDenoisingDefense(BaseDefense):
    """
    Modular Image Denoising Defense.

    Supports configurable denoising methods:
    - 'tv' (Total Variation regularization)
    - 'bilateral' (Edge-preserving bilateral filter)
    - 'local_mean' (Local window averaging)
    """

    supported_domains: List[str] = ["image"]
    defense_family: str = "preprocessing"

    def __init__(
        self,
        method: str = "tv",
        strength: float = 0.05,
        iterations: int = 5,
        kernel_size: int = 3,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize ImageDenoisingDefense.

        Args:
            method: Denoising algorithm ('tv', 'bilateral', 'local_mean').
            strength: Denoising strength / weight parameter.
            iterations: Iteration count (for iterative methods like 'tv').
            kernel_size: Filter kernel size.
            config: Optional configuration dictionary.
        """
        cfg = config or {}
        super().__init__(
            name="image_denoising",
            defense_type="preprocessing",
            config=cfg,
            supported_domains=self.supported_domains,
            defense_family=self.defense_family,
        )
        self.method = str(cfg.get("method", method)).lower().strip()
        self.strength = float(cfg.get("strength", strength))
        self.iterations = int(cfg.get("iterations", iterations))
        self.kernel_size = int(cfg.get("kernel_size", kernel_size))

        if self.method not in ("tv", "bilateral", "local_mean"):
            raise HardeningConfigurationError(
                f"Unsupported denoising method '{self.method}'. Supported: ['tv', 'bilateral', 'local_mean']"
            )

    def denoise(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply selected denoising algorithm to input image tensor."""
        if not isinstance(inputs, torch.Tensor):
            raise HardeningConfigurationError(f"Expected torch.Tensor, got {type(inputs)}")
        if inputs.ndim != 4:
            return inputs.clone()

        if self.method == "tv":
            return denoise_total_variation(inputs, weight=self.strength, iterations=self.iterations)
        elif self.method == "bilateral":
            return denoise_bilateral_approx(
                inputs,
                kernel_size=self.kernel_size,
                sigma_s=max(self.strength * 20.0, 1.0),
                sigma_r=max(self.strength, 0.05),
            )
        elif self.method == "local_mean":
            return denoise_local_mean(inputs, kernel_size=self.kernel_size)
        else:
            return inputs

    def apply(
        self,
        model: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> HardeningResult:
        start_time = time.time()
        try:
            hardened_inputs = self.denoise(inputs) if inputs is not None else None
            self.mark_hardened_input(hardened_inputs, self.denoise)
            wrapped_model = HardenedModelWrapper(model, self.denoise, defense_name=self.name)

            exec_time = time.time() - start_time
            meta = HardeningMetadata(
                defense_name=self.name,
                defense_type=self.defense_type,
                parameters={
                    "method": self.method,
                    "strength": self.strength,
                    "iterations": self.iterations,
                    "kernel_size": self.kernel_size,
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
                    f"Applied Image Denoising defense (method='{self.method}', strength={self.strength:.3f})."
                ],
            )
        except Exception as e:
            if isinstance(e, HardeningConfigurationError):
                raise
            raise DefenseExecutionError(f"ImageDenoisingDefense failed: {str(e)}") from e
