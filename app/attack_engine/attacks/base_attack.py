"""
Common attack interface for the Adversarial Attack Engine.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from app.ingestion.adapters.base_adapter import BaseModelAdapter
from app.attack_engine.models import AttackMetadata


class BaseAttack(ABC):
    """
    Abstract base class defining the standardized interface for all adversarial attacks.
    """

    metadata: AttackMetadata

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

    @classmethod
    def get_metadata(cls) -> AttackMetadata:
        """
        Retrieve standardized metadata describing this attack class.
        """
        return getattr(cls, "metadata", None)

    @property
    def attack_metadata(self) -> AttackMetadata:
        """
        Instance property shortcut for attack metadata.
        """
        return self.get_metadata()

    def validate_config(self, config: Optional[Any] = None) -> None:
        """
        Validate attack configuration parameters against this attack's requirements.

        Subclasses may override this to perform attack-specific parameter validation.
        """
        pass

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
