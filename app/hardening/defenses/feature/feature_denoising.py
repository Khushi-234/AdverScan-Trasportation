"""
Feature Denoising Defense implementation for Module 7 (Hardening).

Reduces adversarial perturbations within intermediate feature representations
at deployment time using modular activation hooks or wrapper transformations.
Does not assume a fixed CNN architecture.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.hardening.defenses.base import BaseDefense
from app.hardening.hardening_result import HardeningMetadata, HardeningResult
from app.hardening.exceptions import DefenseExecutionError, HardeningConfigurationError


def denoise_features(features: torch.Tensor, method: str = "mean", strength: float = 0.1) -> torch.Tensor:
    """
    Denoise intermediate activation tensor.

    Args:
        features: Intermediate tensor (B, C, H, W) or (B, D).
        method: Denoising algorithm ('mean', 'soft_threshold').
        strength: Denoising strength/threshold parameter in [0.0, 1.0].

    Returns:
        Denoised feature tensor matching input shape.
    """
    if features.ndim == 4:
        # 4D feature map (B, C, H, W)
        b, c, h, w = features.shape
        if method == "mean":
            if h >= 3 and w >= 3:
                weight = torch.ones((c, 1, 3, 3), device=features.device, dtype=features.dtype) / 9.0
                smoothed = F.conv2d(features, weight, padding=1, groups=c)
            elif h > 1 or w > 1:
                # Small spatial maps (e.g., 2x2): smooth towards spatial average to avoid silent skip
                smoothed = features.mean(dim=(-2, -1), keepdim=True).expand_as(features)
            else:
                # 1x1 spatial feature maps: smooth towards cross-channel local average
                smoothed = features.mean(dim=1, keepdim=True).expand_as(features)
            # Convex combination between raw and smoothed features
            return (1.0 - strength) * features + strength * smoothed
        elif method == "soft_threshold":
            # Per-sample thresholding: reduce over (C, H, W) with keepdim=True (shape B, 1, 1, 1).
            # Prevents batch coupling and cross-sample threshold contamination.
            th = strength * features.abs().mean(dim=(1, 2, 3), keepdim=True)
            # Differentiable soft-thresholding (shrinkage) operator:
            # S_th(x) = x * max(0, 1 - th / (|x| + eps))
            # Avoids torch.sign() whose backward gradient is zero almost everywhere,
            # eliminating artificial gradient masking (Athalye et al., 2018).
            shrink = F.relu(1.0 - th / (features.abs() + 1e-12))
            return features * shrink
        else:
            return features
    elif features.ndim == 2:
        # 2D feature vector (B, D)
        if method == "soft_threshold":
            # Per-sample thresholding across feature dimension (shape B, 1)
            th = strength * features.abs().mean(dim=-1, keepdim=True)
            shrink = F.relu(1.0 - th / (features.abs() + 1e-12))
            return features * shrink
        else:
            # Local feature averaging across channels
            return (1.0 - strength) * features + strength * features.mean(dim=-1, keepdim=True)
    return features


class FeatureDenoisedModelWrapper(nn.Module):
    """
    nn.Module wrapper that applies intermediate feature denoising via PyTorch forward hooks.
    """

    def __init__(
        self,
        base_model: nn.Module,
        target_layer: Optional[str] = None,
        method: str = "mean",
        strength: float = 0.2,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.target_layer_name = target_layer
        self.method = method
        self.strength = strength
        self._hook_handle = None
        self._register_feature_hook()

    def _register_feature_hook(self) -> None:
        """Find target module and attach feature denoising forward hook."""
        target_mod: Optional[nn.Module] = None

        if self.target_layer_name:
            for name, mod in self.base_model.named_modules():
                if name == self.target_layer_name:
                    target_mod = mod
                    break
            if target_mod is None:
                raise HardeningConfigurationError(
                    f"Target layer '{self.target_layer_name}' not found in model."
                )
        else:
            # Auto-detect mid-to-late layer (Xie et al., 2019):
            # Adversarial perturbations amplify into semantically meaningful noise in late feature maps.
            # Denoising the very first layer operates on raw pixels, undermining the core principle.
            conv_candidates = [
                (name, mod) for name, mod in self.base_model.named_modules() if isinstance(mod, nn.Conv2d)
            ]
            linear_candidates = [
                (name, mod) for name, mod in self.base_model.named_modules() if isinstance(mod, nn.Linear)
            ]

            if conv_candidates:
                # Select the last Conv2d layer (highest-level representation before classifier head)
                self.target_layer_name, target_mod = conv_candidates[-1]
            elif linear_candidates:
                # If multiple linear layers, select penultimate (latent features before final logits)
                if len(linear_candidates) > 1:
                    self.target_layer_name, target_mod = linear_candidates[-2]
                else:
                    self.target_layer_name, target_mod = linear_candidates[-1]
            else:
                raise HardeningConfigurationError(
                    "Auto-detection failed: no Conv2d or Linear layer found in model."
                )

        def hook_fn(module: nn.Module, inputs: Any, output: torch.Tensor) -> torch.Tensor:
            if isinstance(output, torch.Tensor):
                return denoise_features(output, method=self.method, strength=self.strength)
            return output

        self._hook_handle = target_mod.register_forward_hook(hook_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)

    def remove_hook(self) -> None:
        """Detach registered forward hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


class FeatureDenoisingDefense(BaseDefense):
    """
    Feature Denoising Defense.

    Attenuates adversarial noise from intermediate activations during model forward execution.
    Hooks into mid-to-late representation layers (or user-specified layer) without model retraining.
    """

    supported_domains: List[str] = ["image", "tabular"]
    defense_family: str = "feature"

    def __init__(
        self,
        target_layer: Optional[str] = None,
        method: str = "mean",
        strength: float = 0.2,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize FeatureDenoisingDefense.

        Args:
            target_layer: Name of module layer to hook (or None for auto-detect).
            method: Feature denoising method ('mean', 'soft_threshold').
            strength: Denoising blending strength in [0.0, 1.0].
            config: Optional configuration dictionary.
        """
        cfg = config or {}
        super().__init__(
            name="feature_denoising",
            defense_type="feature",
            config=cfg,
            supported_domains=self.supported_domains,
            defense_family=self.defense_family,
        )
        self.target_layer = cfg.get("target_layer", target_layer)
        self.method = str(cfg.get("method", method)).lower().strip()
        self.strength = float(cfg.get("strength", strength))

        if not (0.0 <= self.strength <= 1.0):
            raise HardeningConfigurationError(f"strength must be in [0.0, 1.0], got {self.strength}")
        if self.method not in ("mean", "soft_threshold"):
            raise HardeningConfigurationError(
                f"Unsupported feature denoising method '{self.method}'. Supported: ['mean', 'soft_threshold']"
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
            wrapped_model = FeatureDenoisedModelWrapper(
                base_model=model,
                target_layer=self.target_layer,
                method=self.method,
                strength=self.strength,
            )

            hardened_inputs = inputs.clone() if inputs is not None else None
            exec_time = time.time() - start_time

            meta = HardeningMetadata(
                defense_name=self.name,
                defense_type=self.defense_type,
                parameters={
                    "target_layer": wrapped_model.target_layer_name,
                    "method": self.method,
                    "strength": self.strength,
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
                    f"Applied Feature Denoising ({self.method}, strength={self.strength:.2f}) on layer '{wrapped_model.target_layer_name}'."
                ],
            )
        except Exception as e:
            if isinstance(e, HardeningConfigurationError):
                raise
            raise DefenseExecutionError(f"FeatureDenoisingDefense failed: {str(e)}") from e
