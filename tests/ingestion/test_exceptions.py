"""
Unit tests for ingestion custom exception hierarchy.
"""

from app.ingestion.exceptions import (
    ModelIngestionError,
    ModelLoadError,
    ModelValidationError,
    UnsupportedModelError,
)


def test_exception_inheritance():
    """Verify that all custom exceptions inherit from ModelIngestionError."""
    assert issubclass(ModelLoadError, ModelIngestionError)
    assert issubclass(ModelValidationError, ModelIngestionError)
    assert issubclass(UnsupportedModelError, ModelIngestionError)


def test_exception_raising():
    """Verify catching specific custom exceptions under base exception handler."""
    try:
        raise ModelLoadError("Failed to load model file.")
    except ModelIngestionError as e:
        assert "Failed to load model file." in str(e)
