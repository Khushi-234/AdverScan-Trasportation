"""
Model Ingestion & Standardization Module for AdverScan.
"""

from app.ingestion.adapters import (
    AdapterCapabilities,
    BaseModelAdapter,
    HuggingFaceAdapter,
    PyTorchAdapter,
    SklearnAdapter,
)
from app.ingestion.exceptions import (
    ConfigurationError,
    ModelIngestionError,
    ModelLoadError,
    ModelValidationError,
    UnsafeModelError,
    UnsupportedDomainError,
    UnsupportedModelError,
    UnsupportedTaskError,
)
from app.ingestion.loader import (
    BaseModelLoader,
    HuggingFaceLoader,
    LoadResult,
    PyTorchLoader,
    SklearnLoader,
)
from app.ingestion.metadata import ModelMetadata, TabularMetadata
from app.ingestion.model_registry import (
    ModelPresetRegistry,
    detect_model_domain,
    detect_model_framework,
    detect_model_source,
    inspect_model_evidence,
    is_forecasting_or_regression,
    is_sklearn_model,
)
from app.ingestion.pipeline import IngestionResult, ingest_model
from app.ingestion.presets import DATASET_TO_RECOMMENDED_MODEL, MODEL_PRESETS
from app.ingestion.processors import (
    AttackCompatibility,
    BaseProcessor,
    ImageProcessor,
    TabularProcessor,
    TextProcessor,
    TimeSeriesProcessor,
)
from app.ingestion.utils import log_ingestion_event
from app.ingestion.utils.device_manager import DeviceManager
from app.ingestion.validation import ModelValidator

__all__ = [
    "AdapterCapabilities",
    "AttackCompatibility",
    "BaseModelLoader",
    "LoadResult",
    "PyTorchLoader",
    "HuggingFaceLoader",
    "BaseModelAdapter",
    "PyTorchAdapter",
    "HuggingFaceAdapter",
    "BaseProcessor",
    "ImageProcessor",
    "TabularProcessor",
    "TextProcessor",
    "TimeSeriesProcessor",
    "DeviceManager",
    "ModelValidator",
    "ModelMetadata",
    "TabularMetadata",
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
    "UnsupportedTaskError",
    "SklearnAdapter",
    "SklearnLoader",
    "UnsafeModelError",
    "is_sklearn_model",
    "detect_model_framework",
    "detect_model_source",
    "inspect_model_evidence",
    "detect_model_domain",
    "is_forecasting_or_regression",
    "ingest_model",
    "log_ingestion_event",
]

