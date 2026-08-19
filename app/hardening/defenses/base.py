"""
Abstract Base Class for all defense implementations in AdverScan Module 7.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn

from app.hardening.hardening_result import HardeningResult


class BaseDefense(ABC):
    """
    Abstract Base Class for all defense implementations.
    """

    def __init__(self, name: str, defense_type: str, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize base defense.

        Args:
            name: Defense registration identifier.
            defense_type: Defense category ('preprocessing', 'smoothing', 'adversarial_training').
            config: Optional configuration dictionary.
        """
        self.name = name
        self.defense_type = defense_type
        self.config = config or {}

    @abstractmethod
    def apply(
        self,
        model: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> HardeningResult:
        """
        Apply defense mechanism to model or inputs.

        Args:
            model: PyTorch neural network module.
            inputs: Optional input tensor.
            labels: Optional ground truth target labels.
            **kwargs: Additional runtime parameters.

        Returns:
            HardeningResult: Dataclass containing hardened model/inputs and metadata.
        """
        pass

    def get_name(self) -> str:
        """Get defense identifier."""
        return self.name

    def get_defense_type(self) -> str:
        """Get defense category."""
        return self.defense_type
