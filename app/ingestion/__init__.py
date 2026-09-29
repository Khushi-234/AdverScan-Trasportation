"""
Model Ingestion & Standardization Module for AdverScan.
"""

from app.ingestion.adapters import BaseModelAdapter, PyTorchAdapter
from app.ingestion.exceptions import (
    ModelIngestionError,
    ModelLoadError,
    ModelValidationError,
    UnsupportedModelError,
)
from app.ingestion.loader import BaseModelLoader, PyTorchLoader
from app.ingestion.metadata import ModelMetadata
from app.ingestion.pipeline import ingest_model
from app.ingestion.runtime import DeviceManager
from app.ingestion.validation import ModelValidator
from app.ingestion.utils import log_ingestion_event

__all__ = [
    "BaseModelLoader",
    "PyTorchLoader",
    "BaseModelAdapter",
    "PyTorchAdapter",
    "DeviceManager",
    "ModelValidator",
    "ModelMetadata",
    "ModelIngestionError",
    "ModelLoadError",
    "ModelValidationError",
    "UnsupportedModelError",
    "ingest_model",
    "log_ingestion_event",
]
