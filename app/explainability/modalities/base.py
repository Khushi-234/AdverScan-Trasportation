"""
Base interface for modality-specific input handlers and explanation processors.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from app.explainability.modality import Modality


class BaseModalityHandler(ABC):
    """
    Abstract base class for domain/modality handlers in AdverScan XAI.

    Each handler manages:
    1. Input preparation, validation, and batching.
    2. Modality-specific attribution post-processing.
    3. Extraction of top salient features/regions/tokens/time steps.
    """

    def __init__(self, modality: Modality):
        self.modality = modality

    @abstractmethod
    def prepare_input(self, inputs: Any) -> Any:
        """
        Validate and format inputs for model inference and explainers.
        """
        pass

    @abstractmethod
    def prepare_batch(self, clean: Any, adversarial: Any, **kwargs: Any) -> Tuple[Any, Any]:
        """
        Validate and format both clean and adversarial inputs to ensure compatibility.
        """
        pass

    @abstractmethod
    def postprocess_attribution(
        self,
        attribution: Any,
        inputs: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Convert raw explainer output into a standardized, modality-rich dictionary.

        Returns:
            Dictionary containing:
                - "attribution": Normalized attribution array/list
                - "top_features": Ranked list of key features/regions/tokens
                - "summary": Modality-specific statistical summary
        """
        pass

    @abstractmethod
    def get_top_features(
        self,
        attribution: Any,
        inputs: Any,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Extract the top-k most influential features according to attribution magnitude.
        """
        pass
