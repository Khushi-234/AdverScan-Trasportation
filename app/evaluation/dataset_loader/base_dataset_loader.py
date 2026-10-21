"""
Abstract base class for dataset loaders in AdverScan evaluation.
"""

from abc import ABC, abstractmethod
from typing import Any, Generator, List, Tuple


class BaseDatasetLoader(ABC):
    """
    Abstract base dataset loader for AdverScan evaluation modules.
    Allows generic domain support (ITS, Financial, Medical, Vision, NLP, Time-Series).
    """

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Get dataset identifier."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get total number of samples in dataset split."""
        pass

    @abstractmethod
    def iterate_batches(
        self,
    ) -> Generator[Tuple[Any, Any, List[int]], None, None]:
        """Yield (batch_inputs, batch_targets, target_labels_list)."""
        pass
