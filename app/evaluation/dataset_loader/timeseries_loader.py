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
        inputs: Optional[Union[torch.Tensor, str, Any]] = None,
        targets: Optional[Union[torch.Tensor, Any]] = None,
        dataset_name: str = "TimeSeries_Dataset",
        batch_size: int = 32,
        data_path: Optional[Union[str, Any]] = None,
        target_column: Optional[str] = None,
        **kwargs: Any,
    ):
        from pathlib import Path

        self._dataset_name = dataset_name
        self.batch_size = batch_size

        if inputs is None:
            inputs = data_path or kwargs.get("data") or kwargs.get("path")

        if isinstance(inputs, (str, Path)):
            from pathlib import Path
            p = Path(inputs)
            resolved_p = None
            if p.is_file():
                resolved_p = str(p)
            else:
                from app.evaluation.dataset_loader.factory import resolve_dataset_file
                resolved_p = resolve_dataset_file(
                    str(inputs),
                    [".pt", ".pth", ".npz", ".npy", ".csv", ".tsv", ".parquet"],
                )

            if not resolved_p or not Path(resolved_p).is_file():
                raise ValueError(f"Time-series dataset file '{inputs}' not found.")

            file_path = Path(resolved_p)
            suffix = file_path.suffix.lower()

            if suffix in (".pt", ".pth"):
                loaded = torch.load(file_path, map_location="cpu", weights_only=False)
                if isinstance(loaded, dict):
                    inputs = loaded.get("inputs", loaded.get("X", loaded.get("data")))
                    if targets is None:
                        targets = loaded.get("targets", loaded.get("y", loaded.get("labels")))
                elif isinstance(loaded, tuple) and len(loaded) == 2:
                    inputs, targets = loaded
                else:
                    inputs = loaded

            elif suffix == ".npy":
                import numpy as np
                arr = np.load(file_path)
                inputs = arr

            elif suffix == ".npz":
                import numpy as np
                npz = np.load(file_path)
                for k in ("inputs", "X", "arr_0", "data"):
                    if k in npz:
                        inputs = npz[k]
                        break
                if targets is None:
                    for k in ("targets", "y", "labels", "arr_1"):
                        if k in npz:
                            targets = npz[k]
                            break

            elif suffix in (".csv", ".tsv", ".parquet"):
                import pandas as pd
                if suffix == ".parquet":
                    df = pd.read_parquet(file_path)
                else:
                    df = pd.read_csv(file_path, sep="\t" if suffix == ".tsv" else ",")

                if df.empty:
                    raise ValueError(f"Time-series dataset file '{file_path}' is empty.")

                target_candidates = [target_column] if target_column else []
                target_candidates.extend(["target", "label", "class", "y"])
                found_target = None
                for c in target_candidates:
                    if c and c in df.columns:
                        found_target = c
                        break

                if found_target and targets is None:
                    targets = df[found_target].values
                    inputs = df.drop(columns=[found_target]).values
                else:
                    inputs = df.select_dtypes(include=["number"]).values
            else:
                raise ValueError(
                    f"Unsupported time-series file format '{suffix}'. "
                    f"Supported formats: .pt, .pth, .npz, .npy, .csv, .tsv, .parquet"
                )

        if not isinstance(inputs, torch.Tensor):
            self.inputs = torch.tensor(inputs, dtype=torch.float32)
        else:
            self.inputs = inputs.to(torch.float32)

        if len(self.inputs) == 0:
            raise ValueError(f"Time-series dataset contains 0 samples.")

        if targets is None:
            targets = torch.zeros(len(self.inputs), dtype=torch.long)

        # Standardize 2D sequence to [batch, 1, sequence_length] if needed
        if len(self.inputs.shape) == 2:
            self.inputs = self.inputs.unsqueeze(1)

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
