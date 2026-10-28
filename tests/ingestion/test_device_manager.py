"""
Unit tests for DeviceManager module.
"""

import torch
import torch.nn as nn

from app.ingestion.runtime.device_manager import DeviceManager


def test_get_device_cpu():
    """Test explicit CPU device selection."""
    dev = DeviceManager.get_device("cpu")
    assert dev == torch.device("cpu")


def test_get_device_none():
    """Test auto-detection device selection."""
    dev = DeviceManager.get_device(None)
    assert isinstance(dev, torch.device)


def test_to_device_tensor():
    """Test moving tensor to specified device."""
    tensor = torch.randn(3, 3)
    moved = DeviceManager.to_device(tensor, "cpu")
    assert moved.device == torch.device("cpu")


def test_to_device_module():
    """Test moving nn.Module to specified device."""
    model = nn.Linear(5, 2)
    moved = DeviceManager.to_device(model, "cpu")
    assert next(moved.parameters()).device == torch.device("cpu")
