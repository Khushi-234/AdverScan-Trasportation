"""
Unit tests for model validation module.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from app.ingestion.adapters.pytorch_adapter import PyTorchAdapter
from app.ingestion.exceptions import ModelValidationError
from app.ingestion.validation.validator import ModelValidator


class ValidNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


class NanNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return torch.tensor([[float("nan"), 1.0]])


def test_validator_success():
    """Test validator success with valid model adapter."""
    model = ValidNet()
    adapter = PyTorchAdapter(model, device="cpu")

    # Without sample input
    assert ModelValidator.validate(adapter) is True

    # With sample input
    sample_input = torch.randn(2, 4)
    assert ModelValidator.validate(adapter, sample_input=sample_input) is True


def test_validator_invalid_adapter_instance():
    """Test error when adapter is not a BaseModelAdapter."""
    with pytest.raises(ModelValidationError, match="Expected adapter instance"):
        ModelValidator.validate("not_an_adapter")


def test_validator_nan_output():
    """Test error when sample inference outputs NaN values."""
    model = NanNet()
    adapter = PyTorchAdapter(model, device="cpu")
    sample_input = torch.randn(1, 4)

    with pytest.raises(ModelValidationError, match="NaN or Inf"):
        ModelValidator.validate(adapter, sample_input=sample_input)


def test_validator_inference_error():
    """Test error when sample inference raises a runtime exception due to dimension mismatch."""
    model = ValidNet()
    adapter = PyTorchAdapter(model, device="cpu")
    bad_input = torch.randn(1, 100)  # Dimension mismatch (expects 4)

    with pytest.raises(ModelValidationError, match="Sample inference failed"):
        ModelValidator.validate(adapter, sample_input=bad_input)


def test_validator_configuration_success():
    """Test valid configuration passes validation."""
    valid_cfg = {
        "model_id": "custom-org/custom-model",
        "framework": "huggingface",
        "data_domain": "image",
        "task_type": "classification",
    }
    assert ModelValidator.validate_configuration(valid_cfg) is True


def test_validator_configuration_missing_model_id():
    """Test missing model_id raises ModelValidationError."""
    with pytest.raises(ModelValidationError, match="'model_id' is required"):
        ModelValidator.validate_configuration({"framework": "pytorch", "data_domain": "image", "task_type": "classification"})


def test_validator_configuration_invalid_framework():
    """Test invalid/missing framework raises error."""
    from app.ingestion.exceptions import UnsupportedModelError

    with pytest.raises(ModelValidationError, match="'framework' is required"):
        ModelValidator.validate_configuration({"model_id": "m1", "data_domain": "image", "task_type": "classification"})

    with pytest.raises(UnsupportedModelError, match="Unsupported framework"):
        ModelValidator.validate_configuration({"model_id": "m1", "framework": "tensorflow", "data_domain": "image", "task_type": "classification"})


def test_validator_configuration_unsupported_domain():
    """Test invalid/unsupported domain raises UnsupportedDomainError."""
    from app.ingestion.exceptions import UnsupportedDomainError

    with pytest.raises(ModelValidationError, match="'data_domain' is required"):
        ModelValidator.validate_configuration({"model_id": "m1", "framework": "pytorch", "task_type": "classification"})

    with pytest.raises(UnsupportedDomainError, match="Unsupported data domain"):
        ModelValidator.validate_configuration({"model_id": "m1", "framework": "pytorch", "data_domain": "audio", "task_type": "classification"})


def test_validator_configuration_missing_task_type():
    """Test missing task_type raises ModelValidationError."""
    with pytest.raises(ModelValidationError, match="'task_type' is required"):
        ModelValidator.validate_configuration({"model_id": "m1", "framework": "pytorch", "data_domain": "image"})


def test_validator_configuration_incompatible_processor():
    """Test incompatible processor and domain raises ModelValidationError."""
    # Image domain with tokenizer
    with pytest.raises(ModelValidationError, match="Incompatible processor"):
        ModelValidator.validate_configuration({
            "model_id": "vit-model",
            "framework": "huggingface",
            "data_domain": "image",
            "task_type": "classification",
            "processor_name": "bert-base-uncased-tokenizer",
        })

    # Text domain with vision processor
    with pytest.raises(ModelValidationError, match="Incompatible processor"):
        ModelValidator.validate_configuration({
            "model_id": "bert-model",
            "framework": "huggingface",
            "data_domain": "text",
            "task_type": "classification",
            "processor_name": "google/vit-base-patch16-224-imageprocessor",
        })

