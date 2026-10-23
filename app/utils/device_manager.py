"""
Device management module for handling PyTorch execution devices (CPU, CUDA, MPS).
Delegates core logic to app.utils.device.
"""

from typing import Any, Optional, Union
import torch

from app.utils.device import get_device as util_get_device, to_device as util_to_device


class DeviceManager:
    """
    Manages hardware execution devices for machine learning models.
    Delegates to app.utils.device.
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
        return util_get_device(preferred_device)

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
        return util_to_device(obj, device)

