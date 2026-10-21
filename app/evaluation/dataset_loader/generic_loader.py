"""
Generic dataset loader for arbitrary PyTorch tensors, arrays, or custom inputs.
"""

from typing import Generator, List, Tuple
import torch

from app.evaluation.dataset_loader.base_dataset_loader import BaseDatasetLoader


class GenericDatasetLoader(BaseDatasetLoader):
    """
    Generic dataset loader for arbitrary PyTorch tensors, arrays, or custom inputs.
    """

    def __init__(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        dataset_name: str = "Custom_Dataset",
        batch_size: int = 32,
    ):
        self._dataset_name = dataset_name
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    def __len__(self) -> int:
        return len(self.inputs)

    def iterate_batches(
        self,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, List[int]], None, None]:
        total_samples = len(self.inputs)
        for i in range(0, total_samples, self.batch_size):
            batch_inputs = self.inputs[i : i + self.batch_size]
            batch_targets = self.targets[i : i + self.batch_size]
            targets_list = batch_targets.cpu().numpy().tolist()
            yield batch_inputs, batch_targets, targets_list
