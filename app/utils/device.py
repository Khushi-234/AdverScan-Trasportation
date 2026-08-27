"""
Device management utilities for hardware execution (CPU, CUDA, MPS) across AdverScan.
"""

from typing import Any, Dict, Optional, Union
import torch


def resolve_device(preferred_device: Optional[Union[str, torch.device]] = None) -> str:
    """
    Auto-detect CUDA GPU or MPS if available or requested, else fall back to CPU.

    Args:
        preferred_device: Target device preference (e.g., 'auto', 'cuda', 'gpu', 'mps', 'cpu', None).

    Returns:
        Device string ('cuda', 'mps', or 'cpu').
    """
    if preferred_device is None or str(preferred_device).lower() in ["auto", "none", ""]:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    dev_str = str(preferred_device).lower()
    if dev_str.startswith("cuda") or dev_str == "gpu":
        if torch.cuda.is_available():
            return "cuda" if dev_str in ["cuda", "gpu"] else dev_str
        return "cpu"
    elif dev_str.startswith("mps"):
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    elif dev_str == "cpu":
        return "cpu"

    return dev_str


def get_device(preferred_device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Determine and return the appropriate PyTorch torch.device instance.

    Args:
        preferred_device: Requested device string or torch.device.

    Returns:
        torch.device instance.
    """
    if isinstance(preferred_device, torch.device):
        dev_str = str(preferred_device)
    else:
        dev_str = resolve_device(preferred_device)
    
    return torch.device(dev_str)


def get_execution_device_info(preferred_device: Optional[Union[str, torch.device]] = None) -> Dict[str, Any]:
    """
    Retrieve comprehensive details regarding the current execution device and environment.

    Args:
        preferred_device: Target device setting.

    Returns:
        Dictionary containing gpu_available, gpu_model, device_str, and device object.
    """
    dev_str = resolve_device(preferred_device)
    gpu_available = torch.cuda.is_available() and dev_str.startswith("cuda")
    gpu_model = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

    return {
        "gpu_available": gpu_available,
        "gpu_model": gpu_model,
        "device_str": dev_str,
        "device": torch.device(dev_str),
    }


def to_device(obj: Any, device: Union[str, torch.device]) -> Any:
    """
    Move a model, tensor, or structure to the specified target device.

    Args:
        obj: Model, tensor, or object with a `.to()` method.
        device: Target device string or torch.device.

    Returns:
        Object moved to target device.
    """
    target_device = get_device(device)
    if hasattr(obj, "to"):
        return obj.to(target_device)
    return obj


class DeviceManager:
    """
    Static manager class providing hardware device resolution services.
    Maintained for backward compatibility across modules.
    """

    @staticmethod
    def get_device(preferred_device: Optional[Union[str, torch.device]] = None) -> torch.device:
        return get_device(preferred_device)

    @staticmethod
    def resolve_device_string(preferred_device: Optional[Union[str, torch.device]] = None) -> str:
        return resolve_device(preferred_device)

    @staticmethod
    def to_device(obj: Any, device: Union[str, torch.device]) -> Any:
        return to_device(obj, device)

    @staticmethod
    def get_device_info(preferred_device: Optional[Union[str, torch.device]] = None) -> Dict[str, Any]:
        return get_execution_device_info(preferred_device)
