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
