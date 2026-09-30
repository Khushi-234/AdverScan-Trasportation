"""
PyTorch adapter implementation for standardizing PyTorch models.
"""

from typing import Any, Union
import numpy as np
import torch
import torch.nn as nn

from app.ingestion.adapters.base_adapter import BaseModelAdapter


class PyTorchAdapter(BaseModelAdapter):
    """
    Standardized adapter for PyTorch models.
    """

    def __init__(self, model: nn.Module, device: Union[str, torch.device] = "cpu"):
        """
        Initialize PyTorchAdapter with a PyTorch model.

        Args:
            model: PyTorch model instance (nn.Module).
            device: Initial target device.
        """
        if not isinstance(model, (nn.Module, torch.jit.ScriptModule)):
            raise TypeError(f"Expected torch.nn.Module or ScriptModule, got {type(model)}")

        self._model = model
        self._device = torch.device(device) if isinstance(device, str) else device
        self._model.to(self._device)

    @property
    def device(self) -> torch.device:
        """Get current model device."""
        return self._device

    def get_model(self) -> nn.Module:
        """Return the wrapped PyTorch model."""
        return self._model

    def to(self, device: Union[str, torch.device]) -> "PyTorchAdapter":
        """Move model to the target device."""
        self._device = torch.device(device) if isinstance(device, str) else device
        self._model.to(self._device)
        return self

    def eval(self) -> "PyTorchAdapter":
        """Set wrapped model to evaluation mode."""
        self._model.eval()
        return self

    def train(self, mode: bool = True) -> "PyTorchAdapter":
        """Set wrapped model to training or evaluation mode."""
        self._model.train(mode)
        return self

    def _prepare_tensor_input(self, inputs: Any) -> torch.Tensor:
        """Utility to convert input data to PyTorch Tensor on target device."""
        if isinstance(inputs, torch.Tensor):
            return inputs.to(self._device)
        elif isinstance(inputs, np.ndarray):
            return torch.from_numpy(inputs).to(self._device)
        elif isinstance(inputs, (list, tuple)):
            tensor = torch.tensor(inputs)
            return tensor.to(self._device)
        else:
            raise TypeError(f"Unsupported input type for PyTorchAdapter: {type(inputs)}")

    def predict(self, inputs: Any, return_numpy: bool = False) -> Union[torch.Tensor, np.ndarray]:
        """
        Perform forward inference on input data.

        Args:
            inputs: Tensor, numpy array, or list/tuple of inputs.
            return_numpy: If True, returns predictions as numpy array.

        Returns:
            Model prediction tensor or numpy array.
        """
        self._model.eval()
        tensor_input = self._prepare_tensor_input(inputs)
        with torch.no_grad():
            outputs = self._model(tensor_input)

        if hasattr(outputs, "logits"):
            outputs = outputs.logits

        if return_numpy:
            return outputs.cpu().numpy()
        return outputs


    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Forward call passing inputs to the model.
        Converts first positional input if appropriate, maintaining gradient context if required.
        """
        if args and isinstance(args[0], (torch.Tensor, np.ndarray, list, tuple)):
            prepared_first = self._prepare_tensor_input(args[0])
            new_args = (prepared_first, *args[1:])
            return self._model(*new_args, **kwargs)
        return self._model(*args, **kwargs)
