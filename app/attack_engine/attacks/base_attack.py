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

        Raises:
            NotImplementedError: If the attack class does not define metadata.
        """
        meta = getattr(cls, "metadata", None)
        if meta is None:
            raise NotImplementedError(
                f"Attack class '{cls.__name__}' must define class-level 'metadata' attribute of type AttackMetadata."
            )
        return meta

    @property
    def attack_metadata(self) -> AttackMetadata:
        """
        Instance property shortcut for attack metadata.
        """
        return self.get_metadata()

    @staticmethod
    def _extract_logits(outputs: Any) -> Any:
        """
        Extract classification logits tensor from model outputs.
        Supports standard Tensors, HuggingFace ModelOutputs (.logits, .prediction_logits, .prediction_outputs),
        and dictionary/tuple wrappers.
        """
        if hasattr(outputs, "logits") and outputs.logits is not None:
            return outputs.logits
        if hasattr(outputs, "prediction_logits") and outputs.prediction_logits is not None:
            return outputs.prediction_logits
        if hasattr(outputs, "prediction_outputs") and outputs.prediction_outputs is not None:
            return outputs.prediction_outputs
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            return outputs[0]
        if hasattr(outputs, "get") and callable(getattr(outputs, "get")):
            for k in ("logits", "prediction_logits", "prediction_outputs"):
                val = outputs.get(k)
                if val is not None:
                    return val
        return outputs

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
