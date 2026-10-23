"""
Custom exceptions for the Model Ingestion module.
"""

class ModelIngestionError(Exception):
    """Base exception for model ingestion errors."""
    pass


class ModelLoadError(ModelIngestionError):
    """Raised when loading a model file or model instance fails."""
    pass


class UnsafeModelError(ModelLoadError):
    """Raised when an unsafe deserialization operation is attempted without explicit user consent."""
    pass


class ModelValidationError(ModelIngestionError):
    """Raised when model validation or sanity checking fails."""
    pass


class ConfigurationError(ModelValidationError):
    """Raised when model configuration is missing required fields or invalid."""
    pass


class UnsupportedDomainError(ModelValidationError):
    """Raised when an unsupported data domain is provided."""
    pass


class UnsupportedModelError(ModelIngestionError):
    """Raised when an unsupported model framework or format is provided."""
    pass


class UnsupportedTaskError(ModelIngestionError):
    """Raised when an unsupported task (e.g. forecasting or regression) is provided."""
    pass


class MissingDependencyError(ModelIngestionError):
    """Raised when an optional third-party runtime dependency is required but not installed."""
    pass


class DatasetFormatError(ModelValidationError):
    """Raised when a dataset format or structure is invalid or unsupported."""
    pass

