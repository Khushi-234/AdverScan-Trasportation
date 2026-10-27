"""
Base explainer interface for all XAI techniques in AdverScan.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import torch

from app.explainability.modality import Modality


class BaseExplainer(ABC):
    """
    Abstract base class for explanation technique implementations.
    """

    name: str = "base"
    supported_modalities: List[Modality] = []
    is_available: bool = True

    @abstractmethod
    def explain(
        self,
        model: Any,
        inputs: Any,
        target_class: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate attribution map or feature importance scores.

        Args:
            model: Target model or prediction function.
            inputs: Model input sample or batch.
            target_class: Target class index to explain.
            **kwargs: Additional technique-specific parameters.

        Returns:
            Structured dictionary containing status, executed flag, technique name, and attribution.
        """
        pass

    def is_compatible(self, modality: Modality, model: Any) -> bool:
        """
        Check whether this explainer is compatible with the given modality and model.
        """
        if self.supported_modalities and modality not in self.supported_modalities:
            return False
        return True
