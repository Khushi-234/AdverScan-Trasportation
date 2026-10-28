"""
Model loader implementations for AdverScan.
"""

from app.ingestion.loader.base_loader import BaseModelLoader
from app.ingestion.loader.pytorch_loader import PyTorchLoader

__all__ = ["BaseModelLoader", "PyTorchLoader"]
