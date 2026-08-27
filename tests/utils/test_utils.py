"""
Unit tests for app.utils device and model utility functions.
"""

import torch
import pytest
from app.utils import (
    resolve_device,
    get_device,
    get_execution_device_info,
    to_device,
    DeviceManager,
    patch_hf_config,
)


def test_resolve_device_auto():
    device_str = resolve_device("auto")
    assert device_str in ["cuda", "mps", "cpu"]
    if torch.cuda.is_available():
        assert device_str == "cuda"
    else:
        assert device_str in ["mps", "cpu"]


def test_resolve_device_explicit_cpu():
    assert resolve_device("cpu") == "cpu"


def test_get_device():
    dev = get_device("cpu")
    assert isinstance(dev, torch.device)
    assert dev.type == "cpu"


def test_get_execution_device_info():
    info = get_execution_device_info("auto")
    assert "gpu_available" in info
    assert "gpu_model" in info
    assert "device_str" in info
    assert "device" in info
    assert isinstance(info["device"], torch.device)


def test_to_device():
    tensor = torch.tensor([1.0, 2.0, 3.0])
    moved = to_device(tensor, "cpu")
    assert moved.device.type == "cpu"


def test_device_manager_backward_compatibility():
    dev = DeviceManager.get_device("cpu")
    assert dev.type == "cpu"
    dev_str = DeviceManager.resolve_device_string("cpu")
    assert dev_str == "cpu"


def test_patch_hf_config():
    raw_config = {
        "id2label": {
            "0": "Label_0",
            "1": "Label_1",
            "43": None,
        }
    }
    patched = patch_hf_config(raw_config)
    assert patched["id2label"]["43"] == "Unused"
    assert patched["label2id"]["Unused"] == 43
    assert patched["num_labels"] == 3
