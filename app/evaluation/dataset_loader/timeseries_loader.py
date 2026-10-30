"""
Dataset loader for time-series and sequential data.
"""

from typing import Any, Generator, List, Tuple, Union , Optional
import torch

from app.evaluation.dataset_loader.base_dataset_loader import BaseDatasetLoader


class TimeSeriesDatasetLoader(BaseDatasetLoader):
    """
    Dataset loader for time-series data (sequences, 2D/3D numerical arrays or tensors).
    """

    def __init__(
        self,
        inputs: Union[torch.Tensor, str, Any],
        targets: Optional[Union[torch.Tensor, Any]] = None,
        dataset_name: str = "TimeSeries_Dataset",
        batch_size: int = 32,
    ):
        from pathlib import Path

        self._dataset_name = dataset_name
        self.batch_size = batch_size

        if isinstance(inputs, str):
            from app.evaluation.dataset_loader.factory import resolve_dataset_file
            resolved_p = resolve_dataset_file(inputs, [".pt", ".pth", ".npz", ".npy"])
            if resolved_p and Path(resolved_p).is_file():
                loaded = torch.load(resolved_p, map_location="cpu", weights_only=False)
            else:
                raise ValueError(f"Time-series dataset file '{inputs}' not found.")

            if isinstance(loaded, dict):
                inputs = loaded.get("inputs", loaded.get("X"))
                if targets is None:
                    targets = loaded.get("targets", loaded.get("y"))
            elif isinstance(loaded, tuple) and len(loaded) == 2:
                inputs, targets = loaded

        if targets is None:
            raise ValueError("TimeSeriesDatasetLoader requires targets tensor or a data dict with 'inputs' and 'targets'.")

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
