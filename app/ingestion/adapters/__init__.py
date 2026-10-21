"""
Model adapters for AdverScan framework.
"""

from app.ingestion.adapters.base_adapter import AdapterCapabilities, BaseModelAdapter
from app.ingestion.adapters.huggingface_adapter import HuggingFaceAdapter
from app.ingestion.adapters.pytorch_adapter import PyTorchAdapter
from app.ingestion.adapters.sklearn_adapter import SklearnAdapter

__all__ = ["AdapterCapabilities", "BaseModelAdapter", "PyTorchAdapter", "HuggingFaceAdapter", "SklearnAdapter"]

