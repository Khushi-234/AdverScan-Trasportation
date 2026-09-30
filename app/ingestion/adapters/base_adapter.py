"""
Base adapter interface for standardized model interactions in AdverScan.
"""

from abc import ABC, abstractmethod
from typing import Any, Union


class BaseModelAdapter(ABC):
    """
    Abstract base class wrapping framework-specific models into a unified interface
    for attack, vulnerability assessment, scoring, and hardening engines.
    """

    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        """
        Perform model prediction on inputs.

        Args:
            inputs: Input data (e.g. tensors, numpy arrays, or batches).

        Returns:
            Model outputs / predictions / logits.
        """
        pass

    @abstractmethod
    def get_model(self) -> Any:
        """
        Return the underlying raw framework model.

        Returns:
            The framework-specific model instance.
        """
        pass

    @abstractmethod
    def to(self, device: Any) -> "BaseModelAdapter":
        """
        Move the model to the target execution device.

        Args:
            device: Target device (string or framework device object).

        Returns:
            Self instance.
        """
        pass

    @abstractmethod
    def eval(self) -> "BaseModelAdapter":
        """
        Set the model to evaluation mode.

        Returns:
            Self instance.
        """
        pass

    @abstractmethod
    def train(self, mode: bool = True) -> "BaseModelAdapter":
        """
        Set the model training/eval mode.

        Args:
            mode: True for training mode, False for evaluation mode.

        Returns:
            Self instance.
        """
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Allow calling the adapter instance directly as a callable.
        """
        pass
