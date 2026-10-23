"""
Dataset loading and batch processing package for generalized model evaluation.
"""

from datasets import load_dataset
from app.evaluation.dataset_loader.base_dataset_loader import BaseDatasetLoader
from app.evaluation.dataset_loader.vision_loader import HFVisionDatasetLoader
from app.evaluation.dataset_loader.text_loader import HFTextDatasetLoader
from app.evaluation.dataset_loader.timeseries_loader import TimeSeriesDatasetLoader
from app.evaluation.dataset_loader.tabular_loader import TabularDatasetLoader
from app.evaluation.dataset_loader.generic_loader import GenericDatasetLoader
from app.evaluation.dataset_loader.factory import get_dataset_loader

# Backward-compatibility alias
GTSRBDatasetLoader = HFVisionDatasetLoader

__all__ = [
    "BaseDatasetLoader",
    "HFVisionDatasetLoader",
    "GTSRBDatasetLoader",
    "HFTextDatasetLoader",
    "TimeSeriesDatasetLoader",
    "TabularDatasetLoader",
    "GenericDatasetLoader",
    "get_dataset_loader",
    "load_dataset",
]
