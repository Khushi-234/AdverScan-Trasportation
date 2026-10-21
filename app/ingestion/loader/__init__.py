"""
Model loader implementations for AdverScan.
"""

from app.ingestion.loader.base_loader import BaseModelLoader, LoadResult
from app.ingestion.loader.huggingface_loader import HuggingFaceLoader
from app.ingestion.loader.pytorch_loader import PyTorchLoader
from app.ingestion.loader.sklearn_loader import SklearnLoader

__all__ = ["BaseModelLoader", "LoadResult", "PyTorchLoader", "HuggingFaceLoader", "SklearnLoader"]

