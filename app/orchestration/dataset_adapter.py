"""
Internal dataset loader adapter for translating in-memory tensors to M2 BaseDatasetLoader.
"""

from collections.abc import Mapping
from typing import Any, Generator, List, Tuple
import numpy as np
import torch

from app.evaluation.dataset_loader import BaseDatasetLoader


class InMemoryDatasetLoader(BaseDatasetLoader):
    """
    Internal technical helper adapter for bridging M3 adversarial tensors or dicts
    to M2 BaselineEvaluator without modifying M2 code.
    """

    def __init__(
        self,
        inputs: Any,
        targets: Any,
        dataset_name: str = "adversarial_batch",
        batch_size: int = 32,
    ):
        """
        Initialize InMemoryDatasetLoader.

        Args:
            inputs: In-memory input tensor or dict batch (clean or adversarial).
            targets: Ground-truth target label tensor or list.
            dataset_name: Dataset identifier name.
            batch_size: Evaluation mini-batch size.
        """
        self._inputs = inputs
        self._targets = targets
        self._dataset_name = dataset_name
        self.batch_size = batch_size

    @property
    def dataset_name(self) -> str:
        """Get dataset identifier."""
        return self._dataset_name

    def __len__(self) -> int:
        """Get total number of samples in batch."""
        if isinstance(self._inputs, (dict, Mapping)):
            if "input_ids" in self._inputs:
                return len(self._inputs["input_ids"])
            for v in self._inputs.values():
                if hasattr(v, "__len__"):
                    return len(v)
            return 0
        return len(self._inputs)

    def iterate_batches(
        self,
    ) -> Generator[Tuple[Any, Any, List[int]], None, None]:
        """
        Yield mini-batches of inputs and targets matching BaseDatasetLoader contract.

        Yields:
            Tuple of (batch_inputs, batch_targets, target_labels_list)
        """
        total_samples = len(self)
        for i in range(0, total_samples, self.batch_size):
            if isinstance(self._inputs, (dict, Mapping)):
                batch_pixels = {
                    k: v[i : i + self.batch_size] if hasattr(v, "__getitem__") else v
                    for k, v in self._inputs.items()
                }
            else:
                batch_pixels = self._inputs[i : i + self.batch_size]

            if hasattr(self._targets, "__getitem__"):
                batch_targets = self._targets[i : i + self.batch_size]
            else:
                batch_targets = self._targets

            if isinstance(batch_targets, torch.Tensor):
                targets_list = batch_targets.cpu().numpy().tolist()
            elif isinstance(batch_targets, np.ndarray):
                targets_list = batch_targets.tolist()
            elif isinstance(batch_targets, list):
                targets_list = batch_targets
            else:
                targets_list = list(batch_targets)

            yield batch_pixels, batch_targets, targets_list
