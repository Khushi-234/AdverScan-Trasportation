"""
Dataset loading and batch processing for generalized model baseline evaluation in AdverScan.
"""

from abc import ABC, abstractmethod
import io
import os
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional, Tuple, Union
import torch
from PIL import Image
from datasets import Image as HFImage, load_dataset
from transformers import AutoImageProcessor


class BaseDatasetLoader(ABC):
    """
    Abstract base dataset loader for AdverScan evaluation modules.
    Allows generic domain support (ITS, Financial, Medical, Vision, NLP, Time-Series).
    """

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Get dataset identifier."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get total number of samples in dataset split."""
        pass

    @abstractmethod
    def iterate_batches(
        self,
    ) -> Generator[Tuple[Any, Any, List[int]], None, None]:
        """Yield (batch_inputs, batch_targets, target_labels_list)."""
        pass


class HFVisionDatasetLoader(BaseDatasetLoader):
    """
    Generalized dataset loader for Hugging Face vision datasets (e.g., GTSRB, CIFAR-10, Intel Image Classification).
    Dynamically discovers image and label columns, decodes image payloads, applies image processor,
    and yields mini-batches.
    """

    DATASET_ALIASES = {
        "puneet6060/intel-image-classification": "miladfa7/Intel-Image-Classification",
        "intel-image-classification": "miladfa7/Intel-Image-Classification",
    }
    LABEL_CANDIDATES = ["label", "labels", "ClassId", "target", "class_id", "category", "fine_label"]
    IMAGE_CANDIDATES = ["image", "img", "Path", "bytes", "file"]

    def __init__(
        self,
        dataset_name: str,
        processor_name: Optional[str] = None,
        split: str = "test",
        batch_size: int = 32,
    ):
        """
        Initialize HFVisionDatasetLoader.

        Args:
            dataset_name: Hugging Face dataset identifier.
            processor_name: Hugging Face image processor model identifier.
            split: Dataset split to evaluate ('test', 'train', 'validation', or split string).
            batch_size: Evaluation batch size.
        """
        resolved_name = self.DATASET_ALIASES.get(dataset_name, dataset_name)
        self._dataset_name = dataset_name
        self.processor_name = processor_name
        self.split = split
        self.batch_size = batch_size

        # Load Hugging Face dataset split with fallback if requested split does not exist
        try:
            self._dataset = load_dataset(resolved_name, split=split)
        except Exception:
            # Fallback to 'train' if 'test' split is unavailable in this dataset repository
            # Preserve sample slice (e.g., [:100]) if present in split specification
            slice_suffix = ""
            if "[" in split and "]" in split:
                slice_suffix = split[split.index("[") : split.index("]") + 1]
            fallback_split = f"train{slice_suffix}"
            try:
                self._dataset = load_dataset(resolved_name, split=fallback_split)
                self.split = fallback_split
            except Exception:
                self._dataset = load_dataset(resolved_name, split="train")
                self.split = "train"

        # Prevent automatic HF image decoding when necessary to avoid fsspec/zipfile decoding issues
        if hasattr(self._dataset, "features") and self._dataset.features:
            try:
                for col_name, feat in list(self._dataset.features.items()):
                    if isinstance(feat, HFImage):
                        self._dataset = self._dataset.cast_column(col_name, HFImage(decode=False))
            except Exception:
                pass

        self._processor = None
        if processor_name:
            try:
                self._processor = AutoImageProcessor.from_pretrained(processor_name)
            except Exception:
                try:
                    from transformers import ViTImageProcessor
                    self._processor = ViTImageProcessor.from_pretrained(processor_name)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load AutoImageProcessor for '{processor_name}': {e}"
                    ) from e

        # Inspect dataset schema to discover label and image column names
        self.label_col = self._find_column(self.LABEL_CANDIDATES, default=None)
        self.image_col = self._find_column(self.IMAGE_CANDIDATES, default="image")

        # Fallback for datasets like miladfa7/Intel-Image-Classification whose HF repository lacks a label column
        if self.label_col is None and "intel" in resolved_name.lower():
            try:
                import re
                import zipfile
                from huggingface_hub import hf_hub_download
                from datasets import Dataset
                archive_path = hf_hub_download(resolved_name, "archive.zip", repo_type="dataset")
                label_map = {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}
                with zipfile.ZipFile(archive_path) as z:
                    test_files = [f for f in z.namelist() if f.startswith('seg_test/seg_test/') and f.endswith('.jpg')]
                    max_samples = len(test_files)
                    if "[" in split and "]" in split:
                        m = re.search(r':(\d+)', split)
                        if m:
                            max_samples = min(int(m.group(1)), len(test_files))
                    sub_files = test_files[:max_samples]
                    self._dataset = Dataset.from_dict({
                        "image": [z.read(f) for f in sub_files],
                        "label": [label_map.get(f.split('/')[2].lower(), 0) for f in sub_files],
                    })
                    self.label_col = "label"
                    self.image_col = "image"
            except Exception:
                pass

        if self.label_col is None:
            raise KeyError(
                f"Dataset '{self.dataset_name}' does not contain any recognized label column. "
                f"Candidates checked: {self.LABEL_CANDIDATES}. "
                f"Available columns: {getattr(self._dataset, 'column_names', [])}"
            )

    def _find_column(self, candidates: List[str], default: Optional[str] = None) -> Optional[str]:
        """Discover existing column name from candidate list."""
        if hasattr(self._dataset, "column_names") and self._dataset.column_names:
            cols = self._dataset.column_names
            for c in candidates:
                if c in cols:
                    return c
        # Fallback to inspecting first sample
        if len(self._dataset) > 0:
            try:
                sample = self._dataset[0]
                if isinstance(sample, dict):
                    for c in candidates:
                        if c in sample:
                            return c
            except Exception:
                pass
        return default

    @property
    def dataset_name(self) -> str:
        """Get dataset identifier."""
        return self._dataset_name

    @property
    def processor(self) -> Any:
        """Get image processor instance."""
        return self._processor

    def __len__(self) -> int:
        """Total number of samples in split."""
        return len(self._dataset)

    def _decode_image(self, sample: dict) -> Image.Image:
        """Decode PIL RGB image from raw bytes payload, file path, or PIL object."""
        # Handle uninitialized instances created via __new__ in unit tests
        image_col = getattr(self, "image_col", "image")

        def _to_rgb_image(val: Any) -> Optional[Image.Image]:
            if val is None:
                return None
            if isinstance(val, Image.Image):
                return val.convert("RGB")
            if isinstance(val, dict):
                raw_bytes = val.get("bytes")
                if raw_bytes is not None:
                    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                raw_path = val.get("path")
                if raw_path is not None and os.path.exists(str(raw_path)):
                    return Image.open(str(raw_path)).convert("RGB")
            if isinstance(val, (bytes, bytearray)):
                return Image.open(io.BytesIO(val)).convert("RGB")
            if isinstance(val, (str, Path)) and os.path.exists(str(val)):
                return Image.open(str(val)).convert("RGB")
            return None

        # 1. Direct check of image_col
        if image_col in sample and sample[image_col] is not None:
            img = _to_rgb_image(sample[image_col])
            if img is not None:
                return img

        # 2. Check "Path" bytes structure (common in GTSRB dataset)
        if "Path" in sample and sample["Path"] is not None:
            img = _to_rgb_image(sample["Path"])
            if img is not None:
                return img

        # 3. Check common image key candidates
        for key in ("image", "img", "file", "bytes"):
            if key in sample and sample[key] is not None:
                img = _to_rgb_image(sample[key])
                if img is not None:
                    return img

        # 4. Fallback key scan
        for k, v in sample.items():
            if v is not None:
                img = _to_rgb_image(v)
                if img is not None:
                    return img

        raise KeyError("Dataset sample does not contain a valid image payload or path bytes.")

    def iterate_batches(
        self,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, List[int]], None, None]:
        """
        Yield mini-batches of preprocessed tensors and ground-truth targets.

        Yields:
            Tuple of (pixel_values_tensor, target_labels_tensor, class_ids_list)
        """
        total_samples = len(self._dataset)

        for i in range(0, total_samples, self.batch_size):
            try:
                batch_samples = self._dataset[i : i + self.batch_size]
            except Exception:
                # Fallback to item-by-item access if batch slicing fails inside fsspec/zipfile
                batch_samples = [self._dataset[j] for j in range(i, min(i + self.batch_size, total_samples))]

            # Reconstruct list of dicts if Hugging Face dataset returns dict of lists
            if isinstance(batch_samples, dict):
                first_key = next(iter(batch_samples))
                num_items = len(batch_samples[first_key])
                items = [
                    {key: batch_samples[key][j] for key in batch_samples}
                    for j in range(num_items)
                ]
            else:
                items = batch_samples

            images: List[Image.Image] = []
            targets: List[int] = []

            for item in items:
                img = self._decode_image(item)
                if not getattr(self, "label_col", None) or self.label_col not in item:
                    raise KeyError(
                        f"Dataset '{self.dataset_name}' sample is missing required label column '{getattr(self, 'label_col', None)}'. "
                        f"Available columns: {list(item.keys())}"
                    )
                class_id = item[self.label_col]
                images.append(img)
                targets.append(int(class_id))

            if self._processor is not None:
                processed = self._processor(images=images, return_tensors="pt")
                pixel_values = processed["pixel_values"]
            else:
                # Default torchvision transform fallback if no HuggingFace processor is available
                from torchvision import transforms
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                pixel_values = torch.stack([preprocess(img) for img in images])

            targets_tensor = torch.tensor(targets, dtype=torch.long)

            yield pixel_values, targets_tensor, targets


class GTSRBDatasetLoader(HFVisionDatasetLoader):
    """
    Dataset loader for Hugging Face GTSRB dataset (bazyl/GTSRB).
    Inherits from generalized HFVisionDatasetLoader for backward compatibility.
    """

    def __init__(
        self,
        dataset_name: str = "bazyl/GTSRB",
        processor_name: str = "bazyl/gtsrb-model",
        split: str = "test",
        batch_size: int = 32,
    ):
        super().__init__(
            dataset_name=dataset_name,
            processor_name=processor_name,
            split=split,
            batch_size=batch_size,
        )


class HFTextDatasetLoader(BaseDatasetLoader):
    """
    Generalized dataset loader for Hugging Face text / NLP classification datasets (e.g., IMDb, SST-2, AG News).
    Discovers text and label columns, tokenizes inputs using Hugging Face AutoTokenizer, and yields mini-batches.
    """

    TEXT_CANDIDATES = ["text", "sentence", "content", "premise", "input", "document", "review"]
    LABEL_CANDIDATES = ["label", "labels", "target", "class", "category"]

    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: Optional[str] = None,
        split: str = "test",
        batch_size: int = 32,
        max_length: int = 512,
    ):
        self._dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.split = split
        self.batch_size = batch_size
        self.max_length = max_length

        self._dataset = load_dataset(dataset_name, split=split)

        self._tokenizer = None
        if tokenizer_name:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load AutoTokenizer for '{tokenizer_name}': {e}"
                ) from e

        self.text_col = self._find_column(self.TEXT_CANDIDATES, default="text")
        self.label_col = self._find_column(self.LABEL_CANDIDATES, default=None)
        if self.label_col is None:
            raise KeyError(
                f"Dataset '{self.dataset_name}' does not contain any recognized label column. "
                f"Candidates checked: {self.LABEL_CANDIDATES}. "
                f"Available columns: {getattr(self._dataset, 'column_names', [])}"
            )

    def _find_column(self, candidates: List[str], default: Optional[str] = None) -> Optional[str]:
        if hasattr(self._dataset, "column_names") and self._dataset.column_names:
            cols = self._dataset.column_names
            for c in candidates:
                if c in cols:
                    return c
        if len(self._dataset) > 0:
            sample = self._dataset[0]
            if isinstance(sample, dict):
                for c in candidates:
                    if c in sample:
                        return c
        return default

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    def __len__(self) -> int:
        return len(self._dataset)

    def iterate_batches(
        self,
    ) -> Generator[Tuple[Any, torch.Tensor, List[int]], None, None]:
        total_samples = len(self._dataset)

        for i in range(0, total_samples, self.batch_size):
            batch_samples = self._dataset[i : i + self.batch_size]

            if isinstance(batch_samples, dict):
                first_key = next(iter(batch_samples))
                num_items = len(batch_samples[first_key])
                items = [
                    {key: batch_samples[key][j] for key in batch_samples}
                    for j in range(num_items)
                ]
            else:
                items = batch_samples

            texts: List[str] = []
            targets: List[int] = []
            for item in items:
                if not getattr(self, "label_col", None) or self.label_col not in item:
                    raise KeyError(
                        f"Dataset '{self.dataset_name}' sample is missing required label column '{getattr(self, 'label_col', None)}'. "
                        f"Available columns: {list(item.keys())}"
                    )
                texts.append(str(item.get(self.text_col, "")))
                targets.append(int(item[self.label_col]))

            if self._tokenizer is not None:
                encoded = self._tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                batch_inputs = encoded
            else:
                batch_inputs = texts

            targets_tensor = torch.tensor(targets, dtype=torch.long)
            yield batch_inputs, targets_tensor, targets


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


def get_dataset_loader(
    dataset_name: str,
    data_domain: str = "image",
    processor_name: Optional[str] = None,
    split: str = "test",
    batch_size: int = 32,
    **kwargs: Any,
) -> BaseDatasetLoader:
    """
    Factory function to construct domain-specific dataset loaders for evaluation.
    Supports data_domain: 'image'/'vision', 'text'/'nlp', 'time_series'/'timeseries'/'sequence',
    'tabular'/'csv'/'table', 'generic'/'custom'.
    """
    domain_clean = str(data_domain).strip().lower().replace("-", "_")

    if domain_clean in ("image", "vision"):
        return HFVisionDatasetLoader(
            dataset_name=dataset_name,
            processor_name=processor_name,
            split=split,
            batch_size=batch_size,
            **kwargs,
        )
    elif domain_clean in ("text", "nlp"):
        return HFTextDatasetLoader(
            dataset_name=dataset_name,
            tokenizer_name=processor_name,
            split=split,
            batch_size=batch_size,
            **kwargs,
        )
    elif domain_clean in ("time_series", "timeseries", "sequence"):
        if "inputs" not in kwargs or "targets" not in kwargs:
            raise ValueError(
                "TimeSeriesDatasetLoader requires both 'inputs' and 'targets' keyword arguments."
            )
        return TimeSeriesDatasetLoader(
            inputs=kwargs["inputs"],
            targets=kwargs["targets"],
            dataset_name=dataset_name,
            batch_size=batch_size,
        )
    elif domain_clean in ("tabular", "csv", "table"):
        if "data" not in kwargs:
            raise ValueError(
                "TabularDatasetLoader requires 'data' keyword argument (file path or (X, y) tuple)."
            )
        return TabularDatasetLoader(
            data=kwargs["data"],
            target_col=kwargs.get("target_col", "target"),
            dataset_name=dataset_name,
            batch_size=batch_size,
        )
    elif domain_clean in ("generic", "custom"):
        if "inputs" not in kwargs or "targets" not in kwargs:
            raise ValueError(
                "GenericDatasetLoader requires both 'inputs' and 'targets' keyword arguments."
            )
        return GenericDatasetLoader(
            inputs=kwargs["inputs"],
            targets=kwargs["targets"],
            dataset_name=dataset_name,
            batch_size=batch_size,
        )
    else:
        raise ValueError(
            f"Unsupported data domain: '{data_domain}'. "
            f"Expected one of: 'image', 'vision', 'text', 'nlp', 'time_series', 'timeseries', 'sequence', 'tabular', 'csv', 'table', 'generic', 'custom'."
        )


