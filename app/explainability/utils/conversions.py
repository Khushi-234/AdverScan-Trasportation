"""
Data type conversion and model unwrapping utilities for AdverScan explainability module.
Provides robust helpers for bridging PyTorch tensors, NumPy arrays, pandas structures,
and framework model wrappers.
"""

from typing import Any, Optional, Union
import numpy as np
import torch


def unwrap_model(model: Any) -> Any:
    """
    Unwrap any adapter or wrapper (e.g., PyTorchAdapter, M1 wrappers) to obtain the underlying model.

    Traverses through common wrapper attributes (`get_model()`, `_model`, `model`)
    while guarding against circular references.

    Args:
        model: Model instance, adapter, or callable.

    Returns:
        Underlying PyTorch nn.Module, scikit-learn estimator, or callable.
    """
    curr = model
    visited = set()
    while curr is not None and id(curr) not in visited:
        visited.add(id(curr))
        if hasattr(curr, "get_model") and callable(curr.get_model):
            curr = curr.get_model()
        elif hasattr(curr, "_model") and curr._model is not None and curr._model is not curr:
            curr = curr._model
        elif hasattr(curr, "model") and curr.model is not None and curr.model is not curr:
            curr = curr.model
        else:
            break
    return curr


def to_numpy(x: Any, dtype: Optional[Any] = None) -> np.ndarray:
    """
    Safely convert inputs (Tensor, DataFrame, Series, list, tuple, scalar) to NumPy ndarray.

    Args:
        x: Input data of diverse formats.
        dtype: Optional target numpy data type.

    Returns:
        np.ndarray of converted data.

    Raises:
        TypeError: If input object cannot be converted to a NumPy array.
    """
    if isinstance(x, np.ndarray):
        arr = x
    elif isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    elif hasattr(x, "to_numpy"):  # pandas DataFrame or Series
        arr = x.to_numpy()
    elif isinstance(x, (list, tuple)):
        arr = np.asarray(x)
    elif np.isscalar(x):
        arr = np.array([x])
    else:
        try:
            arr = np.asarray(x)
        except Exception as e:
            raise TypeError(f"Cannot convert object of type {type(x)} to numpy array: {e}")

    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def to_tensor(
    x: Any,
    dtype: torch.dtype = torch.float32,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Safely convert inputs (NumPy array, DataFrame, list, scalar, or tensor) to a PyTorch Tensor.

    Args:
        x: Input data.
        dtype: Target PyTorch data type (default: torch.float32).
        device: Target PyTorch device.

    Returns:
        torch.Tensor of converted data.
    """
    if isinstance(x, torch.Tensor):
        t = x.detach().clone().to(dtype=dtype).contiguous()
    elif isinstance(x, np.ndarray):
        arr = np.ascontiguousarray(x.copy())
        t = torch.from_numpy(arr).to(dtype=dtype).contiguous()
    elif hasattr(x, "to_numpy"):  # pandas DataFrame/Series
        t = torch.from_numpy(np.ascontiguousarray(x.to_numpy().copy())).to(dtype=dtype).contiguous()
    elif isinstance(x, (list, tuple)):
        t = torch.tensor(x, dtype=dtype).contiguous()
    elif np.isscalar(x):
        t = torch.tensor([x], dtype=dtype).contiguous()
    else:
        arr = np.ascontiguousarray(to_numpy(x).copy())
        t = torch.from_numpy(arr).to(dtype=dtype).contiguous()

    if device is not None:
        t = t.to(device)
    return t
