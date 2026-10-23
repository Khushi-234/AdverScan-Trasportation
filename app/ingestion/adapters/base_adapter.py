"""
Base adapter interface for standardized model interactions in AdverScan.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Union


@dataclass
class AdapterCapabilities:
    """
    Operational and analytical capabilities of a model adapter.
    Exposes gradient, prediction, batching, and logits support for attack compatibility.
    """
    supports_gradients: bool = False
    supports_predict: bool = True
    supports_batch_predict: bool = True
    supports_logits: bool = True


class BaseModelAdapter(ABC):
    """
    Abstract base class wrapping framework-specific models into a unified interface
    for attack, vulnerability assessment, scoring, and hardening engines.
    """

    @property
    def capabilities(self) -> AdapterCapabilities:
        """
        Return the operational capabilities of this adapter.
        Default base implementation assumes standard prediction without gradients.
        """
        return AdapterCapabilities(
            supports_gradients=False,
            supports_predict=True,
            supports_batch_predict=True,
            supports_logits=True,
        )

    def get_capabilities(self) -> AdapterCapabilities:
        """Return the operational capabilities of this adapter."""
        return self.capabilities

    @abstractmethod
    def predict(self, inputs: Any, **kwargs: Any) -> Any:
        """
        Perform model prediction on inputs.

        Args:
            inputs: Input data (e.g. tensors, numpy arrays, or batches).
            **kwargs: Extra prediction parameters.

        Returns:
            Model outputs / predictions / logits.
        """
        pass

    def predict_batch(self, inputs: Any, **kwargs: Any) -> Any:
        """
        Perform batch prediction on inputs. Default implementation delegates to predict.

        Args:
            inputs: Batch of input data.
            **kwargs: Extra prediction parameters.

        Returns:
            Batch model outputs or logits.
        """
        return self.predict(inputs, **kwargs)

    def get_logits(self, inputs: Any, **kwargs: Any) -> Any:
        """
        Compute and return raw unnormalized model logits. Default implementation delegates to predict.

        Args:
            inputs: Model inputs.
            **kwargs: Extra inference parameters.

        Returns:
            Logits tensor or array.
        """
        outputs = self.predict(inputs, **kwargs)
        return getattr(outputs, "logits", outputs)



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
