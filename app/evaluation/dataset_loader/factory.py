"""
Factory function to construct domain-specific dataset loaders for evaluation.
"""

from typing import Any, Optional

from app.evaluation.dataset_loader.base_dataset_loader import BaseDatasetLoader
from app.evaluation.dataset_loader.vision_loader import HFVisionDatasetLoader
from app.evaluation.dataset_loader.text_loader import HFTextDatasetLoader
from app.evaluation.dataset_loader.timeseries_loader import TimeSeriesDatasetLoader
from app.evaluation.dataset_loader.tabular_loader import TabularDatasetLoader
from app.evaluation.dataset_loader.generic_loader import GenericDatasetLoader


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
