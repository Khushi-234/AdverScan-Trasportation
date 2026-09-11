"""
Randomized Smoothing Defense for Module 7 (Hardening).

Implements empirical randomized smoothing by:
1. Creating multiple Gaussian-perturbed copies of each input.
2. Evaluating the base model on those samples in mini-batches using StochasticInferenceWrapper.
3. Converting model outputs to class probabilities.
4. Averaging probabilities using Monte Carlo estimation.
5. Returning the aggregated prediction during inference.

This implementation uses the common HardenedModelWrapper / StochasticInferenceWrapper
hierarchy, providing duplicate-execution prevention and batched Monte Carlo inference.
It does NOT provide certified robustness guarantees.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from app.hardening.defenses.base import BaseDefense
from app.hardening.defenses.wrapper import (
    HardenedModelWrapper,
    StochasticInferenceWrapper,
)
from app.hardening.hardening_result import HardeningMetadata, HardeningResult
from app.hardening.exceptions import (
    DefenseExecutionError,
    HardeningConfigurationError,
)


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------

def add_gaussian_noise(
    inputs: torch.Tensor,
    sigma: float = 0.1,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> torch.Tensor:
    """
    Add Gaussian noise N(0, sigma^2) to the input and clip the result.

    Args:
        inputs: Input tensor of shape (B, C, H, W), (B, D), etc.
        sigma: Standard deviation of Gaussian noise.
        clip_min: Minimum valid input value. Must be strictly less than clip_max.
        clip_max: Maximum valid input value.

    Returns:
        torch.Tensor: Noisy and clipped tensor.
    """
    if clip_min >= clip_max:
        raise ValueError(
            f"clip_min must be strictly less than clip_max, "
            f"got clip_min={clip_min}, clip_max={clip_max}"
        )
    if sigma <= 0.0:
        return torch.clamp(inputs, min=clip_min, max=clip_max)

    noise = torch.randn_like(inputs) * sigma
    noisy_inputs = inputs + noise

    return torch.clamp(
        noisy_inputs,
        min=clip_min,
        max=clip_max,
    )


# ---------------------------------------------------------------------
# Randomized Smoothing Model (uses common StochasticInferenceWrapper)
# ---------------------------------------------------------------------

class RandomizedSmoothingModel(StochasticInferenceWrapper):
    """
    Empirical Randomized Smoothing model wrapper.

    Subclasses StochasticInferenceWrapper (which extends HardenedModelWrapper)
    to provide empirical randomized smoothing:
    - Base model is evaluated on multiple Gaussian-perturbed versions of the input.
    - Resulting class probabilities are averaged via Monte Carlo estimation.
    - Evaluated in mini-batches of size `batch_size` under `torch.inference_mode()`.
    - Inherits duplicate-application protection from HardenedModelWrapper.
    """

    def __init__(
        self,
        base_model: nn.Module,
        sigma: float = 0.12,
        num_samples: int = 10,
        batch_size: int = 32,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        output_type: str = "prob",
        defense_name: str = "randomized_smoothing",
    ) -> None:
        super().__init__(
            base_model=base_model,
            num_samples=num_samples,
            batch_size=batch_size,
            output_type=output_type,
            defense_name=defense_name,
        )

        if sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        if clip_min >= clip_max:
            raise ValueError(
                f"clip_min must be strictly less than clip_max, "
                f"got clip_min={clip_min}, clip_max={clip_max}"
            )

        self.sigma = float(sigma)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

        # Indicates that this implementation is empirical rather than certified
        self.is_certified: bool = False

    def _perturb(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian perturbation for randomized smoothing."""
        return add_gaussian_noise(
            inputs=inputs,
            sigma=self.sigma,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
        )


# ---------------------------------------------------------------------
# Hardening Defense
# ---------------------------------------------------------------------

class RandomizedSmoothingDefense(BaseDefense):
    """
    AdverScan hardening defense using empirical randomized smoothing.

    The defense does not modify model weights. Instead, it wraps the original model
    using RandomizedSmoothingModel (which inherits from HardenedModelWrapper and
    StochasticInferenceWrapper) to perform stochastic inference with Gaussian perturbations.
    """

    supported_domains: List[str] = [
        "image",
        "tabular",
        "audio",
    ]

    defense_family: str = "smoothing"

    def __init__(
        self,
        sigma: float = 0.12,
        num_samples: int = 10,
        batch_size: int = 32,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        output_type: str = "prob",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = config or {}

        super().__init__(
            name="randomized_smoothing",
            defense_type="smoothing",
            config=cfg,
            supported_domains=self.supported_domains,
            defense_family=self.defense_family,
        )

        self.sigma = float(cfg.get("sigma", sigma))
        self.num_samples = int(cfg.get("num_samples", num_samples))
        self.batch_size = int(cfg.get("batch_size", batch_size))
        self.clip_min = float(cfg.get("clip_min", clip_min))
        self.clip_max = float(cfg.get("clip_max", clip_max))
        self.output_type = str(cfg.get("output_type", output_type)).lower()

        # Validation
        if self.sigma <= 0.0:
            raise HardeningConfigurationError(
                f"sigma must be positive, got {self.sigma}"
            )

        if self.num_samples < 1:
            raise HardeningConfigurationError(
                f"num_samples must be at least 1, got {self.num_samples}"
            )

        if self.batch_size < 1:
            raise HardeningConfigurationError(
                f"batch_size must be at least 1, got {self.batch_size}"
            )

        if self.clip_min >= self.clip_max:
            raise HardeningConfigurationError(
                f"clip_min must be strictly less than clip_max, "
                f"got clip_min={self.clip_min}, clip_max={self.clip_max}"
            )

        if self.output_type not in {"prob", "log_prob", "logits"}:
            raise HardeningConfigurationError(
                "output_type must be either 'prob', 'log_prob', or 'logits'"
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
            # Create stochastic wrapper (HardenedModelWrapper subclass)
            smoothed_model = RandomizedSmoothingModel(
                base_model=model,
                sigma=self.sigma,
                num_samples=self.num_samples,
                batch_size=self.batch_size,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
                output_type=self.output_type,
                defense_name=self.name,
            )

            # Randomized smoothing does not permanently modify inputs
            hardened_inputs = inputs.clone() if inputs is not None else None
            setup_time = time.time() - start_time

            metadata = HardeningMetadata(
                defense_name=self.name,
                defense_type=self.defense_type,
                parameters={
                    "sigma": self.sigma,
                    "num_samples": self.num_samples,
                    "batch_size": self.batch_size,
                    "clip_min": self.clip_min,
                    "clip_max": self.clip_max,
                    "output_type": self.output_type,
                    "smoothing_mode": "empirical",
                    "is_certified": False,
                    "setup_time_seconds": setup_time,
                },
                execution_time_seconds=setup_time,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                extra_metadata={
                    "defense_family": self.defense_family,
                    "inference_behavior": (
                        f"During evaluation, each input is evaluated over {self.num_samples} "
                        f"independent Gaussian-perturbed copies (batch_size={self.batch_size}). "
                        "Predictions are aggregated using Monte Carlo probability averaging."
                    ),
                    "latency_notice": (
                        "execution_time_seconds measures only defense wrapper initialization time. "
                        "Runtime smoothing inference latency must be measured during model evaluation/retest."
                    ),
                    "certification_notice": (
                        "This implementation provides empirical randomized smoothing only. "
                        "It does not provide a certified robustness radius."
                    ),
                },
            )

            return HardeningResult(
                hardened_model=smoothed_model,
                metadata=metadata,
                hardened_inputs=hardened_inputs,
                success=True,
                recommendations=[
                    (
                        f"Applied empirical randomized smoothing with sigma={self.sigma:.3f} and "
                        f"num_samples={self.num_samples}."
                    ),
                    (
                        f"Monte Carlo evaluations are processed in chunks of batch_size={self.batch_size} "
                        "to control memory usage."
                    ),
                    (
                        "Runtime inference latency should be measured during retesting because "
                        "randomized smoothing performs multiple model evaluations per input."
                    ),
                    (
                        "Certified robustness claims should not be made without a separate statistical "
                        "certification procedure."
                    ),
                ],
            )

        except HardeningConfigurationError:
            raise
        except Exception as exc:
            raise DefenseExecutionError(
                f"RandomizedSmoothingDefense failed: {str(exc)}"
            ) from exc


__all__ = [
    "HardenedModelWrapper",
    "StochasticInferenceWrapper",
    "RandomizedSmoothingModel",
    "RandomizedSmoothingDefense",
    "add_gaussian_noise",
]
