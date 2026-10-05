"""
Base processor interface for AdverScan model ingestion.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseProcessor(ABC):
    """
    Abstract base class for input processors.
    Prepares model-compatible inputs using configured domain processors or tokenizers.
    """

    @property
    @abstractmethod
    def data_domain(self) -> str:
        """Return the target data domain ('image' or 'text')."""
        pass

    @abstractmethod
    def process(self, inputs: Any, **kwargs: Any) -> Any:
        """
        Process raw inputs into standard model-compatible tensor or dictionary format.

        Args:
            inputs: Raw input data (e.g. image, text string, numpy array, or tensor).
            **kwargs: Extra processing options.

        Returns:
            Processed tensor or dict of tensors ready for model consumption.
        """
        pass

    @abstractmethod
    def get_processor(self) -> Any:
        """Return the underlying raw processor or tokenizer object."""
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Allow calling the processor instance directly."""
        return self.process(*args, **kwargs)
