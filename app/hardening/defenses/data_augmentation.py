"""
Data Augmentation Defense implementation for Module 7 (Hardening).

Transforms input batches before inference using stochastic transformations:
- Gaussian noise injection (single/few randomized transformations)
- Random crop / resize
- Brightness / contrast adjustments
- Random horizontal flip
"""

import time
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.hardening.defenses.base import BaseDefense
from app.hardening.hardening_result import HardeningMetadata, HardeningResult


class DataAugmentationDefense(BaseDefense):
    """
    Data Augmentation Defense.

    Mitigates adversarial perturbations by transforming input batches prior to inference
    with stochastic data augmentation operations (crop/resize, flip, brightness/contrast, single-pass Gaussian noise).
    """

    def __init__(
        self,
        noise_std: float = 0.02,
        flip_prob: float = 0.5,
        brightness_jitter: float = 0.1,
        contrast_jitter: float = 0.1,
        crop_scale: Tuple[float, float] = (0.9, 1.0),
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize DataAugmentationDefense.

        Args:
            noise_std: Standard deviation for light Gaussian noise perturbation.
            flip_prob: Probability of random horizontal flip.
            brightness_jitter: Scale for random brightness adjustments.
            contrast_jitter: Scale for random contrast adjustments.
            crop_scale: Min and max scale bounds for random cropping and resizing.
            config: Optional configuration dictionary.
        """
        cfg = config or {}
        super().__init__(
            name="data_augmentation",
            defense_type="preprocessing",
            config=cfg,
        )
        self.noise_std = float(cfg.get("noise_std", noise_std))
        self.flip_prob = float(cfg.get("flip_prob", flip_prob))
        self.brightness_jitter = float(cfg.get("brightness_jitter", brightness_jitter))
        self.contrast_jitter = float(cfg.get("contrast_jitter", contrast_jitter))
        self.crop_scale = cfg.get("crop_scale", crop_scale)

    def transform_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation transforms to input tensor batch.

        Args:
            x: Input PyTorch tensor [B, C, H, W] or [B, D].

        Returns:
            Transformed PyTorch tensor.
        """
        out = x.clone()
        device = out.device
        batch_size = out.size(0)

        # 1. Light Gaussian Noise Injection (single-pass perturbation transform)
        if self.noise_std > 0.0:
            noise = torch.randn_like(out) * self.noise_std
            out = out + noise

        # 2. Brightness & Contrast Adjustment
        if self.brightness_jitter > 0.0 or self.contrast_jitter > 0.0:
            if out.ndim == 4:
                # Brightness
                if self.brightness_jitter > 0.0:
                    b_factor = 1.0 + (torch.rand(batch_size, 1, 1, 1, device=device) * 2.0 - 1.0) * self.brightness_jitter
                    out = out * b_factor
                # Contrast
                if self.contrast_jitter > 0.0:
                    c_factor = 1.0 + (torch.rand(batch_size, 1, 1, 1, device=device) * 2.0 - 1.0) * self.contrast_jitter
                    mean = out.mean(dim=(2, 3), keepdim=True)
                    out = (out - mean) * c_factor + mean
            elif out.ndim == 2:
                if self.brightness_jitter > 0.0:
                    b_factor = 1.0 + (torch.rand(batch_size, 1, device=device) * 2.0 - 1.0) * self.brightness_jitter
                    out = out * b_factor

        # 3. Random Horizontal Flip (for 4D spatial image tensors)
        if self.flip_prob > 0.0 and out.ndim == 4 and out.size(3) > 1:
            flip_mask = torch.rand(batch_size, device=device) < self.flip_prob
            if flip_mask.any():
                out[flip_mask] = torch.flip(out[flip_mask], dims=[3])

        # 4. Random Crop and Resize (for 4D spatial image tensors)
        if out.ndim == 4 and out.size(2) > 4 and out.size(3) > 4:
            min_scale, max_scale = self.crop_scale
            if min_scale < 1.0:
                h, w = out.size(2), out.size(3)
                crop_h = max(1, int(h * (min_scale + torch.rand(1).item() * (max_scale - min_scale))))
                crop_w = max(1, int(w * (min_scale + torch.rand(1).item() * (max_scale - min_scale))))
                top = torch.randint(0, h - crop_h + 1, (1,)).item()
                left = torch.randint(0, w - crop_w + 1, (1,)).item()

                cropped = out[:, :, top : top + crop_h, left : left + crop_w]
                out = F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)

        return out

    def apply(
        self,
        model: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> HardeningResult:
        """
        Apply data augmentation defense to transform input batch before inference.

        Args:
            model: PyTorch target neural network module.
            inputs: Input tensor batch.
            labels: Optional ground truth labels.
            **kwargs: Additional parameters.

        Returns:
            Standardized HardeningResult containing transformed inputs and execution metadata.
        """
        start_time = time.time()
        parameters = {
            "noise_std": self.noise_std,
            "flip_prob": self.flip_prob,
            "brightness_jitter": self.brightness_jitter,
            "contrast_jitter": self.contrast_jitter,
            "crop_scale": self.crop_scale,
        }

        hardened_inputs: Optional[torch.Tensor] = None
        if inputs is not None:
            hardened_inputs = self.transform_tensor(inputs)

        exec_time = time.time() - start_time

        meta = HardeningMetadata(
            defense_name=self.name,
            defense_type=self.defense_type,
            parameters=parameters,
            execution_time_seconds=exec_time,
            extra_metadata={
                "input_transformed": hardened_inputs is not None,
                "transformations_applied": ["noise_injection", "brightness_contrast", "horizontal_flip", "crop_resize"],
            },
        )

        return HardeningResult(
            hardened_model=model,
            metadata=meta,
            hardened_inputs=hardened_inputs,
            success=True,
            recommendations=["Applied single-pass Data Augmentation transforms (crop/resize, flip, brightness/contrast, light noise) to input batch."],
        )
