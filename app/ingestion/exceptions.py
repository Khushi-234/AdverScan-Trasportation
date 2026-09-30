"""
Custom exceptions for the Model Ingestion module.
"""

class ModelIngestionError(Exception):
    """Base exception for model ingestion errors."""
    pass


class ModelLoadError(ModelIngestionError):
    """Raised when loading a model file or model instance fails."""
    pass


class ModelValidationError(ModelIngestionError):
    """Raised when model validation or sanity checking fails."""
    pass


class UnsupportedModelError(ModelIngestionError):
    """Raised when an unsupported model framework or format is provided."""
    pass
