"""
Abstract Base Class for all defense implementations in AdverScan Module 7 (Hardening).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from app.hardening.hardening_result import HardeningResult


class BaseDefense(ABC):
    """
    Abstract Base Class for all defense implementations in AdverScan.

    All post-training / deployment-time defenses subclass this and declare their
    family, domain compatibility, and operational metadata.
    """

    # Default metadata contract for post-training / deployment-time defenses
    requires_training: bool = False
    requires_retraining: bool = False
    supported_domains: List[str] = ["image"]
    defense_family: str = "general"

    def __init__(
        self,
        name: str,
        defense_type: str,
        config: Optional[Dict[str, Any]] = None,
        supported_domains: Optional[List[str]] = None,
        defense_family: Optional[str] = None,
    ) -> None:
        """
        Initialize base defense.

        Args:
            name: Defense registration identifier.
            defense_type: Defense category ('preprocessing', 'smoothing', 'detection', 'transformation', 'feature', 'ensemble').
            config: Optional configuration dictionary.
            supported_domains: Optional list of supported data domains.
            defense_family: Optional defense family name.
        """
        self.name = name
        self.defense_type = defense_type
        self.config = config or {}
        if supported_domains is not None:
            self.supported_domains = supported_domains
        if defense_family is not None:
            self.defense_family = defense_family
        self.requires_training = False
        self.requires_retraining = False

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

    def mark_hardened_input(
        self,
        tensor: Optional[torch.Tensor],
        transform_fn: Optional[Any] = None,
    ) -> Optional[torch.Tensor]:
        """
        Mark a tensor as having been hardened by this defense to prevent duplicate
        application when later passed into a HardenedModelWrapper.
        """
        if tensor is not None and isinstance(tensor, torch.Tensor):
            try:
                if not hasattr(tensor, "_hardened_by") or not isinstance(tensor._hardened_by, set):
                    tensor._hardened_by = set()
                if self.name:
                    tensor._hardened_by.add(self.name)
                if transform_fn is not None:
                    tensor._hardened_by.add(id(transform_fn))
                tensor._is_hardened = True
            except Exception:
                pass
        return tensor

    def get_defense_type(self) -> str:
        """Get defense category."""
        return self.defense_type

    def get_supported_domains(self) -> List[str]:
        """Get list of supported data domains."""
        return list(self.supported_domains)

    def get_metadata(self) -> Dict[str, Any]:
        """Get capability and operational metadata for defense selector and registry."""
        return {
            "name": self.name,
            "defense_type": self.defense_type,
            "defense_family": self.defense_family,
            "supported_domains": self.supported_domains,
            "requires_training": self.requires_training,
            "requires_retraining": self.requires_retraining,
            "config": self.config,
        }
