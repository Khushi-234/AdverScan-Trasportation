"""
Model adapters for AdverScan framework.
"""

from app.ingestion.adapters.base_adapter import BaseModelAdapter
from app.ingestion.adapters.pytorch_adapter import PyTorchAdapter

__all__ = ["BaseModelAdapter", "PyTorchAdapter"]
