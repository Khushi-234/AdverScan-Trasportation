"""
Unit tests for model adapter implementations.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from app.ingestion.adapters.base_adapter import BaseModelAdapter
from app.ingestion.adapters.pytorch_adapter import PyTorchAdapter


class LinearNet(nn.Module):
    """Simple linear network for testing adapter."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 3)

    def forward(self, x):
        return self.fc(x)


def test_base_adapter_abstract():
    """Ensure BaseModelAdapter cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseModelAdapter()  # type: ignore[abstract]


def test_pytorch_adapter_init_invalid_type():
    """Ensure PyTorchAdapter raises TypeError when initialized with invalid model object."""
    with pytest.raises(TypeError, match="Expected torch.nn.Module"):
        PyTorchAdapter("not_a_module")


def test_pytorch_adapter_predict_tensor():
    """Test predict with torch.Tensor input."""
    model = LinearNet()
    adapter = PyTorchAdapter(model, device="cpu")

    inputs = torch.randn(4, 8)
    outputs = adapter.predict(inputs)

    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (4, 3)


def test_pytorch_adapter_predict_numpy():
    """Test predict with numpy.ndarray input."""
    model = LinearNet()
    adapter = PyTorchAdapter(model, device="cpu")

    inputs = np.random.randn(4, 8).astype(np.float32)

    # Return torch tensor
    out_tensor = adapter.predict(inputs)
    assert isinstance(out_tensor, torch.Tensor)

    # Return numpy array
    out_np = adapter.predict(inputs, return_numpy=True)
    assert isinstance(out_np, np.ndarray)
    assert out_np.shape == (4, 3)


def test_pytorch_adapter_predict_list():
    """Test predict with list/tuple input."""
    model = LinearNet()
    adapter = PyTorchAdapter(model, device="cpu")

    inputs = [[0.5] * 8]
    outputs = adapter.predict(inputs)

    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (1, 3)


def test_pytorch_adapter_unsupported_input_type():
    """Test error when predict is called with unsupported input type."""
    model = LinearNet()
    adapter = PyTorchAdapter(model, device="cpu")

    with pytest.raises(TypeError, match="Unsupported input type"):
        adapter.predict({"invalid": "dict"})


def test_pytorch_adapter_device_and_mode():
    """Test get_model, device tracking, to(), eval(), and train()."""
    model = LinearNet()
    adapter = PyTorchAdapter(model, device="cpu")

    assert adapter.get_model() is model
    assert str(adapter.device) == "cpu"

    adapter.to("cpu")
    assert str(adapter.device) == "cpu"

    adapter.train(True)
    assert model.training is True

    adapter.eval()
    assert model.training is False


def test_pytorch_adapter_callable():
    """Test calling adapter instance directly."""
    model = LinearNet()
    adapter = PyTorchAdapter(model, device="cpu")

    inputs = torch.randn(2, 8)
    outputs = adapter(inputs)

    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (2, 3)
