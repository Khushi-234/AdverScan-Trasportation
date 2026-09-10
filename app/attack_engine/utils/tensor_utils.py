"""
Tensor manipulation and device utility functions for the Adversarial Attack Engine.
"""

from typing import Any, Optional, Union
import torch
import torch.nn as nn


def get_device(model: Any) -> torch.device:
    """
    Extract the primary device on which model parameters reside.

    Args:
        model: PyTorch model or object with parameters.

    Returns:
        torch.device instance (defaults to cpu if no parameters found).
    """
    if hasattr(model, "parameters"):
        try:
            params = list(model.parameters())
            if params:
                return params[0].device
        except Exception:
            pass
    return torch.device("cpu")


def to_tensor(
    data: Any,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Safely convert data to a torch.Tensor and place it on target device.

    Args:
        data: Tensor, numpy array, list, or scalar.
        device: Target torch.device.
        dtype: Target torch.dtype.

    Returns:
        torch.Tensor on target device.
    """
    if isinstance(data, torch.Tensor):
        tensor = data.clone().detach()
    else:
        tensor = torch.as_tensor(data)

    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)

    return tensor


def clip_tensor(
    tensor: torch.Tensor,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> torch.Tensor:
    """
    Clamp tensor values within [clip_min, clip_max] bounds if specified.

    Args:
        tensor: Input tensor.
        clip_min: Lower bound (or None for -inf).
        clip_max: Upper bound (or None for +inf).

    Returns:
        Clamped tensor.
    """
    if clip_min is None and clip_max is None:
        return tensor

    c_min = clip_min if clip_min is not None else float("-inf")
    c_max = clip_max if clip_max is not None else float("inf")
    return torch.clamp(tensor, min=c_min, max=c_max)


def project_lp(
    x_adv: torch.Tensor,
    x_orig: torch.Tensor,
    epsilon: float,
    norm: str = "Linf",
) -> torch.Tensor:
    """
    Project adversarial tensor onto the Lp-ball of radius epsilon around x_orig.

    Args:
        x_adv: Adversarial tensor.
        x_orig: Original clean tensor.
        epsilon: Radius of the perturbation ball.
        norm: Norm type ('Linf', 'inf', 'L2', '2').

    Returns:
        Projected adversarial tensor.
    """
    if epsilon <= 0:
        return x_orig.clone().detach()

    delta = x_adv - x_orig
    norm_upper = norm.upper()

    if norm_upper in ("LINF", "INF"):
        delta = torch.clamp(delta, min=-epsilon, max=epsilon)
        return x_orig + delta

    elif norm_upper in ("L2", "2"):
        batch_size = x_orig.size(0)
        delta_flat = delta.view(batch_size, -1)
        l2_norms = torch.norm(delta_flat, p=2, dim=1)
        factor = torch.clamp(epsilon / (l2_norms + 1e-12), max=1.0)
        view_shape = [batch_size] + [1] * (delta.dim() - 1)
        factor = factor.view(*view_shape)
        return x_orig + delta * factor

    return x_adv
