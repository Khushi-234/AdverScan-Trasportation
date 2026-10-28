"""
Base loader interface for model loaders in AdverScan.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union


class BaseModelLoader(ABC):
    """
    Abstract base class defining the common interface for loading machine learning models.
    """

    @abstractmethod
    def load(self, model_path: Union[str, Path, Any], **kwargs: Any) -> Any:
        """
        Load a model from a file path or model artifact.

        Args:
            model_path: Path to the model file, checkpoint, or an already instantiated model object.
            **kwargs: Framework-specific loading options.

        Returns:
            The loaded raw model object.
        """
        pass
