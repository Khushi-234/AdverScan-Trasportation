"""
Unit tests for model loader implementations.
"""

import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from app.ingestion.exceptions import ModelLoadError
from app.ingestion.loader.base_loader import BaseModelLoader
from app.ingestion.loader.pytorch_loader import PyTorchLoader


class DummyConvNet(nn.Module):
    """Dummy convolutional neural network for testing."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.fc = nn.Linear(4 * 8 * 8, 2)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


def test_base_loader_abstract():
    """Ensure BaseModelLoader cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseModelLoader()  # type: ignore[abstract]


def test_pytorch_loader_from_instance():
    """Test loading directly from an existing nn.Module instance."""
    loader = PyTorchLoader()
    model = DummyConvNet()
    model.train()

    loaded = loader.load(model)
    assert isinstance(loaded, nn.Module)
    assert not loaded.training  # PyTorchLoader should set eval mode


def test_pytorch_loader_from_file():
    """Test loading full serialized PyTorch model from file."""
    loader = PyTorchLoader()
    model = DummyConvNet()

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "convnet.pt"
        torch.save(model, model_path)

        loaded = loader.load(model_path)
        assert isinstance(loaded, nn.Module)
        assert not loaded.training


def test_pytorch_loader_from_state_dict():
    """Test loading weights state_dict with model_class provided."""
    loader = PyTorchLoader()
    model = DummyConvNet()

    with tempfile.TemporaryDirectory() as tmpdir:
        weights_path = Path(tmpdir) / "weights.pth"
        torch.save(model.state_dict(), weights_path)

        # Test passing class type
        loaded = loader.load(weights_path, model_class=DummyConvNet)
        assert isinstance(loaded, DummyConvNet)
        assert not loaded.training

        # Test passing model instance as factory structure
        loaded_inst = loader.load(weights_path, model_class=DummyConvNet())
        assert isinstance(loaded_inst, DummyConvNet)


def test_pytorch_loader_from_nested_state_dict():
    """Test loading state_dict wrapped in a checkpoint dictionary."""
    loader = PyTorchLoader()
    model = DummyConvNet()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "checkpoint.ckpt"
        torch.save({"state_dict": model.state_dict(), "epoch": 10}, ckpt_path)

        loaded = loader.load(ckpt_path, model_class=DummyConvNet)
        assert isinstance(loaded, DummyConvNet)


def test_pytorch_loader_state_dict_missing_class():
    """Test error when loading state_dict without providing model_class."""
    loader = PyTorchLoader()
    model = DummyConvNet()

    with tempfile.TemporaryDirectory() as tmpdir:
        weights_path = Path(tmpdir) / "weights.pth"
        torch.save(model.state_dict(), weights_path)

        with pytest.raises(ModelLoadError, match="no model_class or model structure was provided"):
            loader.load(weights_path)


def test_pytorch_loader_nonexistent_file():
    """Test error when file path does not exist."""
    loader = PyTorchLoader()
    with pytest.raises(ModelLoadError, match="Model file not found"):
        loader.load("/invalid/path/to/model.pt")


def test_pytorch_loader_corrupted_file():
    """Test error when attempting to load a corrupted file."""
    loader = PyTorchLoader()
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_path = Path(tmpdir) / "bad.pt"
        bad_path.write_bytes(b"not a valid torch file content")

        with pytest.raises(ModelLoadError, match="Failed to load PyTorch model"):
            loader.load(bad_path)
