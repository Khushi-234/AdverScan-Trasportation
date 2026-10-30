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


def test_pytorch_adapter_predict_batch_and_get_logits():
    """Test PyTorchAdapter predict_batch and get_logits methods."""
    model = LinearNet()
    adapter = PyTorchAdapter(model, device="cpu")

    inputs = torch.randn(4, 8)
    # predict_batch
    batch_out = adapter.predict_batch(inputs)
    assert isinstance(batch_out, torch.Tensor)
    assert batch_out.shape == (4, 3)

    # get_logits
    logits = adapter.get_logits(inputs)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (4, 3)

    # get_logits return numpy
    logits_np = adapter.get_logits(inputs, return_numpy=True)
    assert isinstance(logits_np, np.ndarray)
    assert logits_np.shape == (4, 3)


class DummyHFModel(nn.Module):
    """Mock Hugging Face model returning an object with .logits attribute."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 3)

    def forward(self, input_ids=None, pixel_values=None, **kwargs):
        x = pixel_values if pixel_values is not None else input_ids
        if x is None and "inputs" in kwargs:
            x = kwargs["inputs"]
        logits = self.fc(x.float())
        
        class Output:
            def __init__(self, l):
                self.logits = l
        return Output(logits)


def test_huggingface_adapter_interface():
    """Test HuggingFaceAdapter standard interface: predict, predict_batch, get_logits, eval, to, __call__."""
    from app.ingestion.adapters.huggingface_adapter import HuggingFaceAdapter

    raw_model = DummyHFModel()
    adapter = HuggingFaceAdapter(raw_model, device="cpu", data_domain="image")

    assert adapter.get_model() is raw_model
    assert str(adapter.device) == "cpu"

    inputs = torch.randn(3, 8)

    # predict
    out = adapter.predict(inputs)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 3)

    # predict with return_numpy
    out_np = adapter.predict(inputs, return_numpy=True)
    assert isinstance(out_np, np.ndarray)
    assert out_np.shape == (3, 3)

    # predict_batch
    batch_out = adapter.predict_batch(inputs)
    assert isinstance(batch_out, torch.Tensor)
    assert batch_out.shape == (3, 3)

    # get_logits
    logits = adapter.get_logits(inputs)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (3, 3)

    # get_logits return_numpy
    logits_np = adapter.get_logits(inputs, return_numpy=True)
    assert isinstance(logits_np, np.ndarray)

    # eval / train / to
    adapter.eval()
    assert not raw_model.training
    adapter.train(True)
    assert raw_model.training
    adapter.eval()

    adapter.to("cpu")
    assert str(adapter.device) == "cpu"

    # callable / __call__
    call_out = adapter(pixel_values=inputs)
    assert hasattr(call_out, "logits")


def test_huggingface_adapter_userdict_mapping():
    """Test HuggingFaceAdapter handling collections.UserDict / BatchEncoding inputs."""
    from collections import UserDict
    from app.ingestion.adapters.huggingface_adapter import HuggingFaceAdapter

    raw_model = DummyHFModel()
    adapter = HuggingFaceAdapter(raw_model, device="cpu")

    mapping_inputs = UserDict({
        "input_ids": torch.randint(0, 100, (4, 8)),
        "attention_mask": torch.ones(4, 8),
    })
    preds = adapter.predict(mapping_inputs)
    assert isinstance(preds, torch.Tensor)
    assert preds.shape[0] == 4


