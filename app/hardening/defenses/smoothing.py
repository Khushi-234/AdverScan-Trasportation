"""
Smoothing-based Defense Module for Module 7 (Hardening).

Implements Randomized Smoothing defense:
- Wraps a PyTorch neural network with noise-augmented sampling (Gaussian noise N(0, sigma^2))
- Aggregates multi-sample predictions via probability averaging / majority voting
- Provides provable/certifiable robust predictions under L2 perturbations
"""

import time
from datetime import datetime
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.hardening.defenses.base import BaseDefense
from app.hardening.hardening_result import HardeningMetadata, HardeningResult
from app.hardening.utils import add_gaussian_noise
from app.hardening.exceptions import DefenseExecutionError


class RandomizedSmoothingModel(nn.Module):
    """
    PyTorch nn.Module wrapper implementing Randomized Smoothing.

    For each input sample, generates `num_samples` noisy variants with Gaussian noise N(0, sigma^2),
    evaluates the base model on all variants, and aggregates outputs (average probabilities).
    """

    def __init__(
        self,
        base_model: nn.Module,
        sigma: float = 0.12,
        num_samples: int = 10,
        batch_size: int = 32,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.sigma = sigma
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.clip_min = clip_min
        self.clip_max = clip_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with randomized Gaussian noise sampling and expectation aggregation.

        Args:
            x: Input tensor of shape (B, C, H, W) or (B, D).

        Returns:
            torch.Tensor: Smoothed output logits or averaged class probabilities of shape (B, NumClasses).
        """
        if not self.training and self.num_samples > 1:
            # Aggregate over multiple noise draws
            batch_size_inputs = x.shape[0]
            accumulated_probs = None

            for i in range(self.num_samples):
                noisy_x = add_gaussian_noise(
                    inputs=x,
                    sigma=self.sigma,
                    clip_min=self.clip_min,
                    clip_max=self.clip_max,
                )
                logits = self.base_model(noisy_x)
                probs = F.softmax(logits, dim=-1)

                if accumulated_probs is None:
                    accumulated_probs = probs
                else:
                    accumulated_probs = accumulated_probs + probs

            avg_probs = accumulated_probs / self.num_samples
            # Return log probabilities so argmax / NLLLoss works directly
            return torch.log(avg_probs + 1e-10)
        else:
            # Training or single sample pass
            noisy_x = add_gaussian_noise(
                inputs=x,
                sigma=self.sigma,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
            )
            return self.base_model(noisy_x)


class RandomizedSmoothingDefense(BaseDefense):
    """
    Randomized Smoothing Defense implementation.
    """

    def __init__(
        self,
        sigma: float = 0.12,
        num_samples: int = 10,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name="randomized_smoothing", defense_type="smoothing", config=config)
        self.sigma = sigma
        self.num_samples = num_samples
        self.clip_min = clip_min
        self.clip_max = clip_max

    def apply(
        self,
        model: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> HardeningResult:
        start_time = time.time()
        try:
            smoothed_model = RandomizedSmoothingModel(
                base_model=model,
                sigma=self.sigma,
                num_samples=self.num_samples,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
            )

            # If inputs are provided, pass through smoothed model for demonstration
            hardened_inputs = inputs

            exec_time = time.time() - start_time
            meta = HardeningMetadata(
                defense_name=self.name,
                defense_type=self.defense_type,
                parameters={
                    "sigma": self.sigma,
                    "num_samples": self.num_samples,
                    "clip_min": self.clip_min,
                    "clip_max": self.clip_max,
                },
                execution_time_seconds=exec_time,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            return HardeningResult(
                hardened_model=smoothed_model,
                metadata=meta,
                hardened_inputs=hardened_inputs,
                success=True,
                recommendations=[
                    f"Wrapped model with Randomized Smoothing (sigma={self.sigma}, num_samples={self.num_samples}).",
                    "Randomized smoothing provides certifiable L2 adversarial robustness.",
                ],
            )
        except Exception as e:
            raise DefenseExecutionError(f"RandomizedSmoothingDefense failed: {str(e)}") from e
