"""
Device management module for handling PyTorch execution devices (CPU, CUDA, MPS).
"""

from typing import Any, Optional, Union
import torch


class DeviceManager:
    """
    Manages hardware execution devices for machine learning models.
    """

    @staticmethod
    def get_device(preferred_device: Optional[Union[str, torch.device]] = None) -> torch.device:
        """
        Determine and return the appropriate execution device.

        Args:
            preferred_device: Requested device string (e.g. 'cuda', 'cuda:0', 'mps', 'cpu').

        Returns:
            torch.device instance.
        """
        if preferred_device is not None:
            dev_str = str(preferred_device).lower()
            if dev_str.startswith("cuda"):
                if torch.cuda.is_available():
                    return torch.device(preferred_device)
                else:
                    return torch.device("cpu")
            elif dev_str.startswith("mps"):
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return torch.device(preferred_device)
                else:
                    return torch.device("cpu")
            elif dev_str == "cpu":
                return torch.device("cpu")
            else:
                return torch.device(preferred_device)

        # Automatic detection logic
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    @staticmethod
    def to_device(obj: Any, device: Union[str, torch.device]) -> Any:
        """
        Move a model or tensor object to the specified target device.

        Args:
            obj: Model, tensor, or structure containing tensors.
            device: Target device.

        Returns:
            The object moved to target device.
        """
        target_device = DeviceManager.get_device(device)
        if hasattr(obj, "to"):
            return obj.to(target_device)
        return obj
