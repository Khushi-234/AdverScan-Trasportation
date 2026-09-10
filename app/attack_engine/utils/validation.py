"""
Validation utilities for inputs, bounds, configs, and model compatibility.
"""

from typing import Any, Optional
import torch
import torch.nn as nn

from app.attack_engine.config import AttackConfig
from app.attack_engine.exceptions import (
    AttackConfigurationError,
    AttackExecutionError,
    UnsupportedModelError,
)
from app.attack_engine.base.attack_metadata import AttackMetadata
from app.ingestion.adapters.base_adapter import BaseModelAdapter


def validate_inputs(
    inputs: Any, labels: Any, attack_name: str = "Attack"
) -> None:
    """
    Validate that inputs and labels are valid tensors and have matching batch sizes.

    Args:
        inputs: Input data.
        labels: True labels data.
        attack_name: Name of attack for error formatting.

    Raises:
        AttackExecutionError: If inputs/labels are invalid or have mismatched batch dimensions.
    """
    if not isinstance(inputs, torch.Tensor):
        raise AttackExecutionError(
            f"{attack_name} expects inputs to be torch.Tensor, got {type(inputs)}"
        )

    if not isinstance(labels, (torch.Tensor, int, list, tuple)):
        raise AttackExecutionError(
            f"{attack_name} expects labels to be torch.Tensor or iterable, got {type(labels)}"
        )

    if isinstance(labels, torch.Tensor):
        if inputs.size(0) != labels.size(0):
            raise AttackExecutionError(
                f"{attack_name} input batch size ({inputs.size(0)}) does not match labels batch size ({labels.size(0)})."
            )


def validate_bounds(
    clip_min: Optional[float] = None, clip_max: Optional[float] = None
) -> None:
    """
    Validate that clip bounds are consistent.

    Raises:
        AttackConfigurationError: If clip_min > clip_max.
    """
    if clip_min is not None and clip_max is not None and clip_min > clip_max:
        raise AttackConfigurationError(
            f"clip_min ({clip_min}) cannot be greater than clip_max ({clip_max})"
        )


def validate_attack_config(config: Optional[AttackConfig]) -> None:
    """
    Validate an AttackConfig object.

    Raises:
        AttackConfigurationError: If config values are invalid.
    """
    if config is None:
        return

    if config.epsilon < 0:
        raise AttackConfigurationError(
            f"Epsilon cannot be negative, got {config.epsilon}"
        )
    validate_bounds(config.clip_min, config.clip_max)


def check_model_compatibility(model: Any, metadata: AttackMetadata) -> None:
    """
    Check if a model satisfies the capability requirements of an attack.

    Args:
        model: Target model or adapter.
        metadata: Static metadata of the attack.

    Raises:
        UnsupportedModelError: If model cannot support the attack.
    """
    if model is None:
        raise AttackConfigurationError("Target model cannot be None.")

    raw_model = model.get_model() if isinstance(model, BaseModelAdapter) else model

    if metadata.requires_gradient or metadata.requires_model_weights:
        if not isinstance(raw_model, nn.Module):
            # Check if model has callable and parameter support
            if not (callable(raw_model) and hasattr(raw_model, "parameters")):
                raise UnsupportedModelError(
                    f"Attack '{metadata.name}' requires a differentiable PyTorch model with parameters, got {type(raw_model)}"
                )
