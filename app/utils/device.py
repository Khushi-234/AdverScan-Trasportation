"""
Device helper functions for PyTorch execution devices (CPU, CUDA, MPS).
"""

from typing import Any, Optional, Union
import torch


def get_device(preferred_device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Determine and return the appropriate execution device.
    """
    if isinstance(preferred_device, torch.device):
        return preferred_device

    if isinstance(preferred_device, str):
        dev_str = preferred_device.lower().strip()
        if dev_str.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(dev_str)
            return torch.device("cpu")
        if dev_str == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if dev_str == "cpu":
            return torch.device("cpu")

    # Auto-detection
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(obj: Any, device: Union[str, torch.device]) -> Any:
    """
    Move a model, tensor, or collection of tensors to the specified device.
    """
    if isinstance(device, str):
        device = get_device(device)

    if hasattr(obj, "to"):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_device(v, device) for v in obj)
    return obj


def get_execution_device_info(preferred: Optional[str] = "auto") -> dict:
    """
    Get detailed device execution dictionary.
    """
    dev = get_device(preferred)
    gpu_available = torch.cuda.is_available()
    gpu_model = torch.cuda.get_device_name(0) if gpu_available else "N/A"
    return {
        "device": dev,
        "device_str": str(dev),
        "gpu_available": gpu_available,
        "gpu_model": gpu_model,
    }
