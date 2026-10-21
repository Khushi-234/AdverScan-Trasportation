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
        from pathlib import Path
        import pandas as pd

        self._dataset_name = dataset_name
        self.batch_size = batch_size

        df = None
        if isinstance(data, str):
            from app.evaluation.dataset_loader.factory import resolve_dataset_file
            resolved_p = resolve_dataset_file(data, [".csv", ".tsv", ".txt"])
            if resolved_p and Path(resolved_p).is_file():
                df = pd.read_csv(resolved_p)
            else:
                raise ValueError(f"Tabular dataset file '{data}' could not be found.")
        elif isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, tuple) and len(data) == 2:
            X, y = data
            self.inputs = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
            self.targets = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long)
            return
        else:
            raise ValueError("Data for TabularDatasetLoader must be a CSV file path, DataFrame, or tuple of (X, y).")

        if df is not None:
            # Resolve target column
            resolved_col = None
            if target_col in df.columns:
                resolved_col = target_col
            elif "target" in df.columns:
                resolved_col = "target"
            elif "label" in df.columns:
                resolved_col = "label"
            elif "class" in df.columns:
                resolved_col = "class"
            elif isinstance(target_col, int) and 0 <= target_col < len(df.columns):
                resolved_col = df.columns[target_col]
            else:
                resolved_col = df.columns[-1]

            y = df[resolved_col].values
            X = df.drop(columns=[resolved_col]).values
            self.inputs = torch.tensor(X, dtype=torch.float32)
            self.targets = torch.tensor(y, dtype=torch.long)

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
