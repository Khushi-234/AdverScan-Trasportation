"""
Common attack interface for the Adversarial Attack Engine.
"""

from abc import ABC, abstractmethod
from typing import Any
from app.ingestion.adapters.base_adapter import BaseModelAdapter


class BaseAttack(ABC):
    """
    Abstract base class defining the standardized interface for adversarial attacks.
    """

    def __init__(self, model: Any):
        """
        Initialize base attack.

        Args:
            model: Target model (raw framework model like PyTorch nn.Module or BaseModelAdapter instance).
        """
        self.model = model

    def _get_raw_model(self) -> Any:
        """
        Extract raw framework model if model is wrapped in a BaseModelAdapter.
        """
        if isinstance(self.model, BaseModelAdapter):
            return self.model.get_model()
        return self.model

    @abstractmethod
    def generate(self, inputs: Any, labels: Any, config: Any = None) -> Any:
        """
        Generate adversarial examples.

        Args:
            inputs: Original input data (e.g., torch.Tensor).
            labels: True class labels for inputs.
            config: Attack parameters configuration (AttackConfig).

        Returns:
            Adversarially perturbed input data.
        """
        raise NotImplementedError
