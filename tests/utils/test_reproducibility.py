"""
Unit tests for ReproducibilityManager in app.utils.reproducibility.
"""

import random
import numpy as np
import pytest
import torch
from app.utils.reproducibility import ReproducibilityManager
from app.orchestration.pipeline_config import PipelineConfig


def test_python_seed_reproducibility():
    ReproducibilityManager.set_seed(42)
    val1 = [random.randint(0, 1000) for _ in range(5)]

    ReproducibilityManager.set_seed(42)
    val2 = [random.randint(0, 1000) for _ in range(5)]

    assert val1 == val2


def test_numpy_seed_reproducibility():
    ReproducibilityManager.set_seed(123)
    arr1 = np.random.randn(10)

    ReproducibilityManager.set_seed(123)
    arr2 = np.random.randn(10)

    np.testing.assert_allclose(arr1, arr2)


def test_pytorch_cpu_seed_reproducibility():
    ReproducibilityManager.set_seed(999)
    t1 = torch.randn(5, 5)

    ReproducibilityManager.set_seed(999)
    t2 = torch.randn(5, 5)

    assert torch.equal(t1, t2)


def test_cuda_seed_behavior_cpu_safe():
    # Calling set_seed should work seamlessly on CPU-only or CUDA machines
    ReproducibilityManager.set_seed(777, deterministic=False)
    t1 = torch.randn(3, 3)

    ReproducibilityManager.set_seed(777, deterministic=False)
    t2 = torch.randn(3, 3)

    assert torch.equal(t1, t2)


def test_deterministic_mode_configuration():
    ReproducibilityManager.set_deterministic(True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

    ReproducibilityManager.set_deterministic(False)
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True


def test_invalid_seed():
    with pytest.raises(ValueError, match="Invalid seed"):
        ReproducibilityManager.set_seed(-1)


def test_collect_environment_metadata():
    meta = ReproducibilityManager.collect_environment_metadata(seed=42, deterministic=True)
    assert "python_version" in meta
    assert "operating_system" in meta
    assert "pytorch_version" in meta
    assert "cuda_available" in meta
    assert "cuda_version" in meta
    assert "gpu_name" in meta
    assert "cpu_info" in meta
    assert meta["reproducibility_settings"]["seed"] == 42
    assert meta["reproducibility_settings"]["deterministic"] is True


def test_collect_identity_metadata():
    config = PipelineConfig(
        model_path="test_model.pt",
        model_name="MyModel",
        dataset_name="bazyl/GTSRB",
        split="test",
    )
    identity = ReproducibilityManager.collect_identity_metadata(config=config)
    assert identity["model_identity"]["model_name"] == "MyModel"
    assert identity["model_identity"]["model_path"] == "test_model.pt"
    assert identity["dataset_identity"]["dataset_name"] == "bazyl/GTSRB"
    assert identity["dataset_identity"]["split"] == "test"
    assert "model_hash" in identity["model_identity"]
    assert "dataset_checksum" in identity["dataset_identity"]
