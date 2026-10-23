"""
Dataset loader for tabular / structured numerical data.
"""

from typing import Any, Generator, List, Optional, Tuple, Union
import torch

from app.evaluation.dataset_loader.base_dataset_loader import BaseDatasetLoader


class TabularDatasetLoader(BaseDatasetLoader):
    """
    Dataset loader for tabular / structured numerical data (CSV files, DataFrames, Tensors).
    """

    def __init__(
        self,
        data: Optional[Union[str, Any]] = None,
        target_col: Optional[Union[str, int]] = "target",
        dataset_name: str = "Tabular_Dataset",
        batch_size: int = 32,
        data_path: Optional[Union[str, Any]] = None,
        target_column: Optional[Union[str, int]] = None,
        **kwargs: Any,
    ):
        from pathlib import Path
        import pandas as pd

        if data is None:
            data = data_path or kwargs.get("path")
        if target_column is not None:
            target_col = target_column
        if target_col is None:
            target_col = "target"

        self._dataset_name = dataset_name
        self.batch_size = batch_size

        df = None
        if isinstance(data, (str, Path)):
            from pathlib import Path
            p = Path(data)
            resolved_p = None
            if p.is_file():
                resolved_p = str(p)
            else:
                from app.evaluation.dataset_loader.factory import resolve_dataset_file
                resolved_p = resolve_dataset_file(
                    str(data),
                    [".csv", ".tsv", ".parquet", ".xlsx", ".json", ".jsonl", ".txt"],
                )

            if not resolved_p or not Path(resolved_p).is_file():
                raise ValueError(f"Tabular dataset file '{data}' could not be found.")

            file_path = Path(resolved_p)
            suffix = file_path.suffix.lower()

            try:
                if suffix == ".csv":
                    df = pd.read_csv(file_path)
                elif suffix == ".tsv":
                    df = pd.read_csv(file_path, sep="\t")
                elif suffix == ".parquet":
                    df = pd.read_parquet(file_path)
                elif suffix == ".xlsx":
                    try:
                        import openpyxl
                    except ImportError as err:
                        from app.ingestion.exceptions import MissingDependencyError
                        raise MissingDependencyError(
                            "Reading Excel (.xlsx) tabular datasets requires 'openpyxl'. "
                            "Please install it via 'pip install openpyxl'."
                        ) from err
                    try:
                        df = pd.read_excel(file_path)
                    except Exception as err:
                        from app.ingestion.exceptions import DatasetFormatError
                        raise DatasetFormatError(f"Failed to read Excel file '{file_path}': {err}") from err
                elif suffix in (".json", ".jsonl"):
                    lines_mode = suffix == ".jsonl"
                    try:
                        df = pd.read_json(file_path, lines=lines_mode)
                    except Exception:
                        df = pd.read_json(file_path, lines=not lines_mode)
                elif suffix == ".txt":
                    # Try comma-delimited then tab-delimited
                    try:
                        df = pd.read_csv(file_path)
                    except Exception:
                        df = pd.read_csv(file_path, sep="\t")
                else:
                    raise ValueError(
                        f"Unsupported tabular file format '{suffix}'. "
                        f"Supported formats: .csv, .tsv, .parquet, .xlsx, .json, .jsonl"
                    )
            except Exception as e:
                from app.ingestion.exceptions import DatasetFormatError, MissingDependencyError
                if isinstance(e, (MissingDependencyError, DatasetFormatError)):
                    raise
                import pandas.errors as pe
                if isinstance(e, (pe.EmptyDataError, pe.ParserError)):
                    raise DatasetFormatError(f"Tabular dataset '{file_path}' is empty or malformed: {e}") from e
                raise ValueError(f"Failed to parse tabular dataset from '{file_path}': {e}") from e

        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, tuple) and len(data) == 2:
            X, y = data
            self.inputs = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
            self.targets = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long)
            return
        else:
            raise ValueError(
                "Data for TabularDatasetLoader must be a file path (.csv, .tsv, .parquet, .xlsx, .json, .jsonl), "
                "DataFrame, or tuple of (X, y)."
            )

        if df is not None:
            if df.empty or len(df) == 0:
                from app.ingestion.exceptions import DatasetFormatError
                raise DatasetFormatError(f"Tabular dataset from '{data}' is empty (0 samples).")

            # Resolve target column strictly
            resolved_col = None
            if target_col in df.columns:
                resolved_col = target_col
            elif target_col is not None and target_col != "target":
                raise ValueError(
                    f"Specified target column '{target_col}' not found in tabular dataset columns: {list(df.columns)}"
                )
            elif "target" in df.columns:
                resolved_col = "target"
            elif "label" in df.columns:
                resolved_col = "label"
            elif "class" in df.columns:
                resolved_col = "class"
            elif isinstance(target_col, int) and 0 <= target_col < len(df.columns):
                resolved_col = df.columns[target_col]
            else:
                if len(df.columns) < 2:
                    raise ValueError(
                        f"Tabular dataset contains only {len(df.columns)} column. "
                        f"Cannot separate features from target label."
                    )
            resolved_col_str = str(resolved_col)
            y_raw = df[resolved_col_str].values
            X_df = df.drop(columns=[resolved_col_str])

            # Convert target labels: handle strings/categoricals or integers
            import numpy as np
            if y_raw.dtype == object or isinstance(y_raw[0] if len(y_raw) > 0 else None, str):
                y_series = pd.Series(y_raw)
                y_codes, _ = pd.factorize(y_series)
                self.targets = torch.tensor(y_codes, dtype=torch.long)
            else:
                try:
                    self.targets = torch.tensor(y_raw.astype(np.int64), dtype=torch.long)
                except Exception:
                    self.targets = torch.tensor(y_raw, dtype=torch.long)

            # Convert features: check for non-numeric columns
            non_numeric = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric:
                raise ValueError(
                    f"Non-numeric feature columns detected in tabular dataset: {non_numeric}. "
                    f"Features must be preprocessed or encoded to numerical values before evaluation."
                )

            X_arr = X_df.values.astype(np.float32)
            self.inputs = torch.tensor(X_arr, dtype=torch.float32)

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
