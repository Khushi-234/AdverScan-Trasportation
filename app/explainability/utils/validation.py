"""
Input validation and shape checking utilities for AdverScan explainability module.
Guarantees consistent dimensionality, single-sample constraints for explainer.explain(),
and shape compatibility between clean and adversarial inputs.
"""

from typing import Any, Tuple
import numpy as np
import torch


def get_shape(x: Any) -> Tuple[int, ...]:
    """
    Get the shape tuple of a tensor, numpy array, dataframe, or sequence.

    Args:
        x: Input tensor, array, list, or dataframe.

    Returns:
        Tuple representing dimensions, or empty tuple if scalar/unknown.
    """
    if hasattr(x, "shape"):
        return tuple(x.shape)
    if isinstance(x, (list, tuple)):
        return (len(x),)
    return ()


def ensure_batch(inputs: Any, modality: str = "tabular") -> Tuple[Any, bool]:
    """
    Ensure input has a batch dimension (B, ...), expanding dimensions if missing.

    Args:
        inputs: Input tensor, array, list, or text.
        modality: Data modality string ('image', 'text', 'tabular', 'timeseries').

    Returns:
        Tuple of (batched_inputs, was_single: bool).
    """
    modality_lower = str(modality).lower()

    if isinstance(inputs, str):
        return [inputs], True

    if isinstance(inputs, (list, tuple)):
        if len(inputs) > 0 and isinstance(inputs[0], str):
            # Already list of strings
            return list(inputs), False

    arr_or_tensor = inputs
    is_tensor = isinstance(arr_or_tensor, torch.Tensor)
    is_np = isinstance(arr_or_tensor, np.ndarray)

    if is_tensor or is_np:
        ndim = arr_or_tensor.ndim
        if modality_lower == "image":
            # Image: 3D (C, H, W) or (H, W, C) -> add batch dimension (1, C, H, W)
            if ndim == 3:
                return (arr_or_tensor.unsqueeze(0) if is_tensor else np.expand_dims(arr_or_tensor, 0), True)
            return arr_or_tensor, False
        elif modality_lower in ("timeseries", "time_series"):
            # Time-series: 2D (T, F) -> (1, T, F)
            if ndim == 2:
                return (arr_or_tensor.unsqueeze(0) if is_tensor else np.expand_dims(arr_or_tensor, 0), True)
            return arr_or_tensor, False
        elif modality_lower == "tabular":
            # Tabular: 1D (D,) -> (1, D)
            if ndim == 1:
                return (arr_or_tensor.unsqueeze(0) if is_tensor else np.expand_dims(arr_or_tensor, 0), True)
            return arr_or_tensor, False

    return inputs, False


def validate_single_sample(inputs: Any, modality: str = "tabular") -> None:
    """
    Ensure input contains exactly one sample. Raises ValueError if batch size N > 1.

    Enforces that `explain()` is invoked on single instances, requiring callers to use
    `explain_batch()` for multi-sample datasets.

    Args:
        inputs: Model input sample or batch.
        modality: Target data modality string.

    Raises:
        ValueError: If input contains more than one sample.
    """
    modality_lower = str(modality).lower()
    if isinstance(inputs, (list, tuple)) and all(isinstance(s, str) for s in inputs):
        if len(inputs) > 1:
            raise ValueError(
                f"explain() accepts exactly one sample. Received batch size N={len(inputs)}. "
                f"Use explain_batch() for batches."
            )
        return

    shape = get_shape(inputs)
    if not shape:
        return

    n_samples = None
    if modality_lower == "image":
        if len(shape) == 4:
            n_samples = shape[0]
    elif modality_lower in ("timeseries", "time_series"):
        if len(shape) == 3:
            n_samples = shape[0]
    elif modality_lower == "tabular":
        if len(shape) == 2:
            n_samples = shape[0]

    if n_samples is not None and n_samples > 1:
        raise ValueError(
            f"explain() accepts exactly one sample. Received batch size N={n_samples}. "
            f"Use explain_batch() for batches."
        )


def validate_matching_shapes(clean: Any, adversarial: Any) -> bool:
    """
    Validate that clean and adversarial inputs have matching shapes and formats.

    Args:
        clean: Clean input sample or batch.
        adversarial: Adversarial input sample or batch.

    Returns:
        True if shapes match or cannot be determined; False if shapes mismatch.
    """
    if isinstance(clean, (list, tuple)) and isinstance(adversarial, (list, tuple)):
        return len(clean) == len(adversarial)
    shape_clean = get_shape(clean)
    shape_adv = get_shape(adversarial)
    if not shape_clean or not shape_adv:
        return True
    return shape_clean == shape_adv
