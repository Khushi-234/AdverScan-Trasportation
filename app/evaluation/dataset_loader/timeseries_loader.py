"""
Dataset loader for time-series and sequential data.
"""

from typing import Any, Generator, List, Tuple, Union
import torch

from app.evaluation.dataset_loader.base_dataset_loader import BaseDatasetLoader


class TimeSeriesDatasetLoader(BaseDatasetLoader):
    """
    Dataset loader for time-series data (sequences, 2D/3D numerical arrays or tensors).
    """

    def __init__(
        self,
        inputs: Union[torch.Tensor, Any],
        targets: Union[torch.Tensor, Any],
        dataset_name: str = "TimeSeries_Dataset",
        batch_size: int = 32,
    ):
        self._dataset_name = dataset_name
        self.batch_size = batch_size

        if not isinstance(inputs, torch.Tensor):
            self.inputs = torch.tensor(inputs, dtype=torch.float32)
        else:
            self.inputs = inputs.to(torch.float32)

        if not isinstance(targets, torch.Tensor):
            self.targets = torch.tensor(targets, dtype=torch.long)
        else:
            self.targets = targets.to(torch.long)

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
