"""
Internal dataset loader adapter for translating in-memory tensors to M2 BaseDatasetLoader.
"""

from typing import Any, Generator, List, Tuple
import torch

from app.evaluation.dataset_loader import BaseDatasetLoader


class InMemoryDatasetLoader(BaseDatasetLoader):
    """
    Internal technical helper adapter for bridging M3 adversarial tensors
    to M2 BaselineEvaluator without modifying M2 code.
    """

    def __init__(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        dataset_name: str = "adversarial_batch",
        batch_size: int = 32,
    ):
        """
        Initialize InMemoryDatasetLoader.

        Args:
            inputs: In-memory input tensor batch (clean or adversarial).
            targets: Ground-truth target label tensor batch.
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
        """Get total number of samples in tensor batch."""
        return len(self._inputs)

    def iterate_batches(
        self,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, List[int]], None, None]:
        """
        Yield mini-batches of tensors and targets matching BaseDatasetLoader contract.

        Yields:
            Tuple of (batch_inputs, batch_targets, target_labels_list)
        """
        total_samples = len(self._inputs)
        for i in range(0, total_samples, self.batch_size):
            batch_pixels = self._inputs[i : i + self.batch_size]
            batch_targets = self._targets[i : i + self.batch_size]
            if isinstance(batch_targets, torch.Tensor):
                targets_list = batch_targets.cpu().numpy().tolist()
            else:
                targets_list = list(batch_targets)
            yield batch_pixels, batch_targets, targets_list
