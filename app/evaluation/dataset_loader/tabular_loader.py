"""
Dataset loader for tabular / structured numerical data.
"""

from typing import Any, Generator, List, Tuple, Union
import torch

from app.evaluation.dataset_loader.base_dataset_loader import BaseDatasetLoader


class TabularDatasetLoader(BaseDatasetLoader):
    """
    Dataset loader for tabular / structured numerical data (CSV files, DataFrames, Tensors).
    """

    def __init__(
        self,
        data: Union[str, Any],
        target_col: Union[str, int] = "target",
        dataset_name: str = "Tabular_Dataset",
        batch_size: int = 32,
    ):
        self._dataset_name = dataset_name
        self.batch_size = batch_size

        if isinstance(data, str):
            import pandas as pd
            df = pd.read_csv(data)
            y = df[target_col].values
            X = df.drop(columns=[target_col]).values
            self.inputs = torch.tensor(X, dtype=torch.float32)
            self.targets = torch.tensor(y, dtype=torch.long)
        elif isinstance(data, tuple) and len(data) == 2:
            X, y = data
            self.inputs = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
            self.targets = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long)
        else:
            raise ValueError("Data for TabularDatasetLoader must be a CSV file path or tuple of (X, y).")

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
