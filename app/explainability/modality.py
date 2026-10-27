"""
Modality definition and auto-detection utilities for AdverScan XAI.

Supports image, text, tabular, and time-series modalities.
"""

from enum import Enum
from typing import Any, Optional, Union
import warnings
import numpy as np


class Modality(str, Enum):
    """Supported data modalities in AdverScan explainability module."""

    IMAGE = "image"
    TEXT = "text"
    TABULAR = "tabular"
    TIMESERIES = "timeseries"

    @classmethod
    def from_str(cls, value: Union[str, "Modality"]) -> "Modality":
        """Parse string or Modality enum safely."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ValueError(f"Expected str or Modality, got {type(value)}")
        val = value.strip().lower()
        # Normalization aliases
        aliases = {
            "time_series": cls.TIMESERIES,
            "time-series": cls.TIMESERIES,
            "timeseries": cls.TIMESERIES,
            "ts": cls.TIMESERIES,
            "img": cls.IMAGE,
            "vision": cls.IMAGE,
            "txt": cls.TEXT,
            "tab": cls.TABULAR,
            "table": cls.TABULAR,
        }
        if val in aliases:
            return aliases[val]
        for member in cls:
            if member.value == val:
                return member
        raise ValueError(
            f"Unsupported modality '{value}'. Valid options are: {[m.value for m in cls]}"
        )


def is_valid_modality(value: Any) -> bool:
    """Check if value corresponds to a valid Modality."""
    try:
        Modality.from_str(value)
        return True
    except (ValueError, TypeError):
        return False


def detect_modality(inputs: Any, hint: Optional[Union[str, Modality]] = None) -> Modality:
    """
    Identify the input data modality and select the appropriate handler.

    Args:
        inputs: Input data (tensor, numpy array, list of strings, DataFrame, etc.).
        hint: Optional explicit modality hint ("image", "text", "tabular", "timeseries").

    Returns:
        Modality enum.
    """
    if hint is not None:
        return Modality.from_str(hint)

    # 1. Text checks: string or collection of strings
    if isinstance(inputs, str):
        return Modality.TEXT
    if isinstance(inputs, (list, tuple)) and len(inputs) > 0 and all(isinstance(x, str) for x in inputs):
        return Modality.TEXT

    # 2. Tabular checks: pandas DataFrame / Series
    try:
        import pandas as pd
        if isinstance(inputs, (pd.DataFrame, pd.Series)):
            return Modality.TABULAR
    except ImportError:
        pass

    # 3. Dictionary checks
    if isinstance(inputs, dict):
        keys = {k.lower() for k in inputs.keys()}
        if keys & {"input_ids", "text", "tokens", "token_ids"}:
            return Modality.TEXT
        if keys & {"pixel_values", "image", "images"}:
            return Modality.IMAGE
        if keys & {"timeseries", "series", "sequence"}:
            return Modality.TIMESERIES
        if keys & {"features", "tabular", "columns"}:
            return Modality.TABULAR

    # 4. Array / Tensor checks
    try:
        import torch
        is_torch = isinstance(inputs, torch.Tensor)
    except ImportError:
        torch = None
        is_torch = False

    is_np = isinstance(inputs, np.ndarray)

    if is_torch or is_np:
        ndim = inputs.ndim
        shape = tuple(inputs.shape)

        # 4D: Typically batched images (B, C, H, W) or (B, H, W, C)
        if ndim == 4:
            warnings.warn("Modality not specified; inferred 'image' from 4D tensor. Specify 'modality' explicitly for best accuracy.", UserWarning, stacklevel=2)
            return Modality.IMAGE

        # 3D: Could be a single image (C, H, W) / (H, W, C) or batched timeseries (B, T, F)
        if ndim == 3:
            # Only treat as single image if spatial dimensions H, W are >= 16 and channels are exactly 1, 3, or 4
            is_channels_first_img = (shape[0] in (1, 3, 4) and shape[1] >= 16 and shape[2] >= 16)
            is_channels_last_img = (shape[2] in (1, 3, 4) and shape[0] >= 16 and shape[1] >= 16)
            if is_channels_first_img or is_channels_last_img:
                warnings.warn("Modality not specified; inferred 'image' from 3D tensor. Specify 'modality' explicitly.", UserWarning, stacklevel=2)
                return Modality.IMAGE
            warnings.warn("Modality not specified; inferred 'timeseries' from 3D tensor (B, T, F). Specify 'modality' explicitly.", UserWarning, stacklevel=2)
            return Modality.TIMESERIES

        # 2D: Tabular data (batch_size, features)
        if ndim == 2:
            warnings.warn("Modality not specified; inferred 'tabular' from 2D tensor. Specify 'modality' explicitly.", UserWarning, stacklevel=2)
            return Modality.TABULAR

        # 1D: Single tabular record or flat feature vector
        if ndim == 1:
            warnings.warn("Modality not specified; inferred 'tabular' from 1D tensor. Specify 'modality' explicitly.", UserWarning, stacklevel=2)
            return Modality.TABULAR

    # Fallback default: if input is a nested numeric list
    if isinstance(inputs, (list, tuple)):
        if len(inputs) > 0 and isinstance(inputs[0], (int, float)):
            return Modality.TABULAR
        if len(inputs) > 0 and isinstance(inputs[0], (list, tuple)):
            return Modality.TABULAR

    raise ValueError(f"Unable to auto-detect modality for input of type {type(inputs)}. Please specify 'modality' explicitly.")
