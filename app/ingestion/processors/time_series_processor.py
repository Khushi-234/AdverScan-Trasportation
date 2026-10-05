"""
Time-series processor implementation for time-series model inputs.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import torch

from app.ingestion.processors.base_processor import BaseProcessor


class TimeSeriesProcessor(BaseProcessor):
    """
    Dedicated processor for time-series model inputs.
    Prepares, normalizes/standardizes, and shapes time-series samples into expected tensor format
    (e.g., [batch, channels, sequence_length] or [batch, sequence_length, channels]).
    """

    def __init__(
        self,
        processor_name_or_path: Optional[str] = None,
        processor: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
        target_format: str = "batch_channels_length",  # 'batch_channels_length' (NCL) or 'batch_length_channels' (NLC)
        target_shape: Optional[Tuple[int, ...]] = None,
        sequence_length: Optional[int] = None,
        num_channels: Optional[int] = None,
        normalize: bool = False,
        standardize: bool = False,
        scaling: Optional[str] = None,  # 'standardize', 'minmax', 'robust'
        mean: Optional[Union[float, List[float], np.ndarray, torch.Tensor]] = None,
        std: Optional[Union[float, List[float], np.ndarray, torch.Tensor]] = None,
        min_val: Optional[Union[float, np.ndarray, torch.Tensor]] = None,
        max_val: Optional[Union[float, np.ndarray, torch.Tensor]] = None,
        pad_mode: str = "constant",
        pad_value: float = 0.0,
    ):
        """
        Initialize TimeSeriesProcessor.

        Args:
            processor_name_or_path: Optional path or identifier for custom preprocessor/scaler.
            processor: Optional existing processor or scaler instance.
            device: Target execution device.
            target_format: Expected tensor layout ('batch_channels_length' or 'batch_length_channels').
            target_shape: Explicit target shape tuple (e.g., (1, 1, 100)).
            sequence_length: Expected sequence length (pads or truncates if specified).
            num_channels: Expected number of channels/features.
            normalize: Whether to apply normalization (min-max or z-score).
            standardize: Whether to apply z-score standardization (zero mean, unit variance).
            scaling: Scaling method ('standardize', 'minmax', etc.).
            mean: Precomputed mean for standardization.
            std: Precomputed std for standardization.
            min_val: Precomputed min for min-max scaling.
            max_val: Precomputed max for min-max scaling.
            pad_mode: Padding mode when input length is less than expected sequence_length.
            pad_value: Value to pad with.
        """
        self.processor_name = processor_name_or_path
        self._processor = processor
        self._device = torch.device(device) if device is not None else None
        self.target_format = target_format
        self.target_shape = target_shape
        self.sequence_length = sequence_length
        self.num_channels = num_channels
        self.normalize = normalize
        self.standardize = standardize or (scaling == "standardize")
        self.scaling = scaling
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val
        self.pad_mode = pad_mode
        self.pad_value = pad_value

    @property
    def data_domain(self) -> str:
        return "time-series"

    def get_processor(self) -> Any:
        return self._processor

    def load_sample(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Load a time-series sample from a file on disk (CSV, TSV, NPY, NPZ, PT, Parquet).

        Args:
            file_path: Path to the time-series file.

        Returns:
            np.ndarray: Loaded time-series numerical data.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Time-series sample file not found: {path}")

        suffix = path.suffix.lower()
        if suffix in (".csv", ".tsv"):
            import pandas as pd
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
            # Drop non-numeric columns (like date/time strings or ID tags)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                return df[numeric_cols].values.astype(np.float32)
            return df.values.astype(np.float32)
        elif suffix == ".npy":
            return np.load(path).astype(np.float32)
        elif suffix == ".npz":
            data = np.load(path)
            first_key = list(data.keys())[0]
            return data[first_key].astype(np.float32)
        elif suffix in (".pt", ".pth"):
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(loaded, torch.Tensor):
                return loaded.detach().cpu().numpy().astype(np.float32)
            elif isinstance(loaded, np.ndarray):
                return loaded.astype(np.float32)
            elif isinstance(loaded, dict):
                for k in ["data", "values", "x", "sample", "past_values"]:
                    if k in loaded:
                        val = loaded[k]
                        return val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else np.array(val, dtype=np.float32)
            raise ValueError(f"Could not extract numerical time-series tensor from {path}")
        elif suffix == ".parquet":
            import pandas as pd
            df = pd.read_parquet(path)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            return df[numeric_cols].values.astype(np.float32) if len(numeric_cols) > 0 else df.values.astype(np.float32)
        else:
            raise ValueError(f"Unsupported time-series file extension: {suffix}")

    def _to_tensor(self, inputs: Any) -> torch.Tensor:
        """Convert input data to a float32 torch.Tensor."""
        if isinstance(inputs, (str, Path)):
            p = Path(inputs)
            if p.is_file():
                arr = self.load_sample(p)
                return torch.from_numpy(arr).float()

        # Handle Pandas DataFrame or Series
        if hasattr(inputs, "values") and not isinstance(inputs, (torch.Tensor, np.ndarray, Mapping)):
            arr = getattr(inputs, "values")
            if isinstance(arr, np.ndarray):
                return torch.from_numpy(arr.astype(np.float32)).float()

        # Handle Mapping/Dictionary
        if isinstance(inputs, Mapping) or hasattr(inputs, "items"):
            for k in ["past_values", "values", "series", "data", "x", "input", "inputs"]:
                if k in inputs:
                    return self._to_tensor(inputs[k])
            raise ValueError(f"Dictionary input missing recognizable time-series key: {list(inputs.keys())}")

        # Handle PyTorch Tensor
        if isinstance(inputs, torch.Tensor):
            return inputs.float()

        # Handle NumPy Array
        if isinstance(inputs, np.ndarray):
            return torch.from_numpy(inputs.astype(np.float32)).float()

        # Handle Python list / tuple
        if isinstance(inputs, (list, tuple)):
            return torch.tensor(inputs, dtype=torch.float32)

        raise TypeError(f"Unsupported time-series input type: {type(inputs)}")

    def _apply_scaling(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply standardization or normalization to tensor along sequence axis."""
        eps = 1e-8
        if self.standardize or self.scaling == "standardize":
            if self.mean is not None and self.std is not None:
                mean_t = torch.as_tensor(self.mean, dtype=tensor.dtype, device=tensor.device)
                std_t = torch.as_tensor(self.std, dtype=tensor.dtype, device=tensor.device)
                return (tensor - mean_t) / (std_t + eps)
            else:
                # Standardize per sample along time axis (last dimension for NCL, or dim 1 for NLC)
                time_dim = 2 if self.target_format == "batch_channels_length" and tensor.ndim == 3 else (
                    1 if tensor.ndim == 3 else -1
                )
                mean = tensor.mean(dim=time_dim, keepdim=True)
                std = tensor.std(dim=time_dim, keepdim=True)
                return (tensor - mean) / (std + eps)

        elif self.normalize or self.scaling == "minmax":
            if self.min_val is not None and self.max_val is not None:
                min_t = torch.as_tensor(self.min_val, dtype=tensor.dtype, device=tensor.device)
                max_t = torch.as_tensor(self.max_val, dtype=tensor.dtype, device=tensor.device)
                return (tensor - min_t) / (max_t - min_t + eps)
            else:
                time_dim = 2 if self.target_format == "batch_channels_length" and tensor.ndim == 3 else (
                    1 if tensor.ndim == 3 else -1
                )
                min_v = tensor.amin(dim=time_dim, keepdim=True)
                max_v = tensor.amax(dim=time_dim, keepdim=True)
                return (tensor - min_v) / (max_v - min_v + eps)

        return tensor

    def _adjust_sequence_length(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pad or truncate sequence dimension if sequence_length is specified."""
        if self.sequence_length is None:
            return tensor

        target_len = self.sequence_length
        # Determine sequence dimension:
        # For 3D: NCL -> dim 2; NLC -> dim 1
        if tensor.ndim == 3:
            time_dim = 2 if self.target_format == "batch_channels_length" else 1
            curr_len = tensor.shape[time_dim]
            if curr_len == target_len:
                return tensor
            elif curr_len > target_len:
                slices = [slice(None)] * 3
                slices[time_dim] = slice(0, target_len)
                return tensor[tuple(slices)]
            else:
                # Pad
                pad_size = target_len - curr_len
                if time_dim == 2:
                    # Pad last dimension (left, right)
                    pad_tuple = (0, pad_size)
                    return torch.nn.functional.pad(tensor, pad_tuple, mode=self.pad_mode, value=self.pad_value)
                else:
                    # NLC -> transpose to NCL, pad, transpose back
                    transposed = tensor.transpose(1, 2)
                    padded = torch.nn.functional.pad(transposed, (0, pad_size), mode=self.pad_mode, value=self.pad_value)
                    return padded.transpose(1, 2)
        elif tensor.ndim == 2:
            curr_len = tensor.shape[1]
            if curr_len == target_len:
                return tensor
            elif curr_len > target_len:
                return tensor[:, :target_len]
            else:
                return torch.nn.functional.pad(tensor, (0, target_len - curr_len), mode=self.pad_mode, value=self.pad_value)

        return tensor

    def process(self, inputs: Any, **kwargs: Any) -> torch.Tensor:
        """
        Process raw time-series inputs into standard PyTorch tensor format.

        Args:
            inputs: Raw time-series data (array, tensor, dataframe, list, or file path).
            **kwargs: Extra processing options.

        Returns:
            torch.Tensor: Formatted tensor (default [batch, channels, sequence_length]).
        """
        tensor = self._to_tensor(inputs)

        # Reshape to expected layout
        if tensor.ndim == 1:
            # 1D single sequence [L]
            if self.target_format == "batch_channels_length":
                tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, L]
            else:
                tensor = tensor.unsqueeze(0).unsqueeze(-1)  # [1, L, 1]
        elif tensor.ndim == 2:
            # 2D [batch, L] or [channels, L] or [L, channels]
            if self.target_format == "batch_channels_length":
                tensor = tensor.unsqueeze(1)  # [batch, 1, L]
            else:
                tensor = tensor.unsqueeze(-1)  # [batch, L, 1]
        elif tensor.ndim == 3:
            # 3D [batch, dim1, dim2]
            if self.target_format == "batch_channels_length":
                # If num_channels is specified and dim2 == num_channels and dim1 != num_channels, transpose
                if self.num_channels is not None and tensor.shape[2] == self.num_channels and tensor.shape[1] != self.num_channels:
                    tensor = tensor.transpose(1, 2)
            elif self.target_format == "batch_length_channels":
                if self.num_channels is not None and tensor.shape[1] == self.num_channels and tensor.shape[2] != self.num_channels:
                    tensor = tensor.transpose(1, 2)

        # Pad or truncate to desired sequence length
        tensor = self._adjust_sequence_length(tensor)

        # Apply scaling / standardization
        tensor = self._apply_scaling(tensor)

        # Move to target device
        if self._device is not None:
            tensor = tensor.to(self._device)

        return tensor
