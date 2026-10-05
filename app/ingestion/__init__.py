"""
Model Ingestion & Standardization Module for AdverScan.
"""

from app.ingestion.adapters import (
    BaseModelAdapter,
    HuggingFaceAdapter,
    PyTorchAdapter,
)
from app.ingestion.exceptions import (
    ConfigurationError,
    ModelIngestionError,
    ModelLoadError,
    ModelValidationError,
    UnsupportedDomainError,
    UnsupportedModelError,
)
from app.ingestion.loader import (
    BaseModelLoader,
    HuggingFaceLoader,
    PyTorchLoader,
)
from app.ingestion.metadata import ModelMetadata
from app.ingestion.model_registry import ModelPresetRegistry
from app.ingestion.pipeline import IngestionResult, ingest_model
from app.ingestion.presets import DATASET_TO_RECOMMENDED_MODEL, MODEL_PRESETS
from app.ingestion.processors import (
    BaseProcessor,
    ImageProcessor,
    TextProcessor,
)
from app.ingestion.utils import log_ingestion_event
from app.ingestion.utils.device_manager import DeviceManager
from app.ingestion.validation import ModelValidator

__all__ = [
    "BaseModelLoader",
    "PyTorchLoader",
    "HuggingFaceLoader",
    "BaseModelAdapter",
    "PyTorchAdapter",
    "HuggingFaceAdapter",
    "BaseProcessor",
    "ImageProcessor",
    "TextProcessor",
    "DeviceManager",
    "ModelValidator",
    "ModelMetadata",
    "ModelPresetRegistry",
    "MODEL_PRESETS",
    "DATASET_TO_RECOMMENDED_MODEL",
    "IngestionResult",
    "ModelIngestionError",
    "ModelLoadError",
    "ModelValidationError",
    "ConfigurationError",
    "UnsupportedDomainError",
    "UnsupportedModelError",
    "ingest_model",
    "log_ingestion_event",
]

