"""
Unit tests for Module 7 defenses (preprocessing, smoothing, adversarial training).
"""

import pytest
import torch
import torch.nn as nn

from app.hardening.defenses import (
    SpatialSmoothingDefense,
    FeatureSqueezingDefense,
    JPEGCompressionDefense,
    PreprocessingDefense,
    RandomizedSmoothingDefense,
    AdversarialTrainingDefense,
)
from app.hardening.utils import (
    add_gaussian_noise,
    apply_spatial_smoothing,
    reduce_bit_depth,
    simulate_jpeg_compression,
)
from app.hardening.exceptions import DefenseExecutionError, HardeningConfigurationError


class DummyClassifier(nn.Module):
    """Simple linear PyTorch classifier for testing."""

    def __init__(self, in_features: int = 12, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        return self.fc(x)


class DummyConvClassifier(nn.Module):
    """Simple conv PyTorch classifier for image testing."""

    def __init__(self, channels: int = 3, num_classes: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 4, kernel_size=3, padding=1)
        self.fc = nn.Linear(4 * 8 * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


def test_utils_tensor_operations():
    inputs = torch.rand(4, 3, 8, 8)
    noisy = add_gaussian_noise(inputs, sigma=0.1)
    assert noisy.shape == inputs.shape
    assert torch.all(noisy >= 0.0) and torch.all(noisy <= 1.0)

    smooth = apply_spatial_smoothing(inputs, kernel_size=3, sigma=1.0)
    assert smooth.shape == inputs.shape

    squeezed = reduce_bit_depth(inputs, bit_depth=3)
    assert squeezed.shape == inputs.shape

    jpeg = simulate_jpeg_compression(inputs, quality=50)
    assert jpeg.shape == inputs.shape


def test_spatial_smoothing_defense():
    model = DummyConvClassifier()
    inputs = torch.rand(2, 3, 8, 8)
    defense = SpatialSmoothingDefense(kernel_size=3, sigma=1.0)

    result = defense.apply(model=model, inputs=inputs)
    assert result.success is True
    assert result.hardened_inputs is not None
    assert result.hardened_inputs.shape == inputs.shape

    output = result.hardened_model(inputs)
    assert output.shape == (2, 2)


def test_bit_depth_reduction_defense():
    model = DummyClassifier(in_features=12, num_classes=2)
    inputs = torch.rand(3, 12)
    defense = FeatureSqueezingDefense(bit_depth=4)

    result = defense.apply(model=model, inputs=inputs)
    assert result.success is True
    assert result.hardened_inputs is not None

    output = result.hardened_model(inputs)
    assert output.shape == (3, 2)


def test_jpeg_compression_defense():
    model = DummyConvClassifier()
    inputs = torch.rand(2, 3, 8, 8)
    defense = JPEGCompressionDefense(quality=70)

    result = defense.apply(model=model, inputs=inputs)
    assert result.success is True
    output = result.hardened_model(inputs)
    assert output.shape == (2, 2)


def test_preprocessing_defense_pipeline():
    model = DummyConvClassifier()
    inputs = torch.rand(2, 3, 8, 8)
    defense = PreprocessingDefense(methods=["spatial_smoothing", "feature_squeezing"])

    result = defense.apply(model=model, inputs=inputs)
    assert result.success is True
    assert "spatial_smoothing" in result.metadata.parameters["methods"]


def test_randomized_smoothing_defense():
    model = DummyClassifier(in_features=12, num_classes=2)
    inputs = torch.rand(4, 12)
    defense = RandomizedSmoothingDefense(sigma=0.1, num_samples=5)

    result = defense.apply(model=model, inputs=inputs)
    assert result.success is True

    # Test forward pass through smoothed wrapper
    smoothed_model = result.hardened_model
    smoothed_model.eval()
    outputs = smoothed_model(inputs)
    assert outputs.shape == (4, 2)


def test_adversarial_training_defense():
    model = DummyClassifier(in_features=12, num_classes=2)
    inputs = torch.rand(6, 12)
    labels = torch.tensor([0, 1, 0, 1, 0, 1])

    defense = AdversarialTrainingDefense(epochs=1, lr=1e-3, epsilon=0.05, attack_type="fgsm")
    result = defense.apply(model=model, inputs=inputs, labels=labels)

    assert result.success is True
    assert result.hardened_model is not None
    assert len(result.metadata.parameters["losses"]) == 1

    # Verify model evaluation working
    result.hardened_model.eval()
    preds = result.hardened_model(inputs)
    assert preds.shape == (6, 2)


def test_adversarial_training_missing_inputs_error():
    model = DummyClassifier()
    defense = AdversarialTrainingDefense()
    with pytest.raises(HardeningConfigurationError):
        defense.apply(model=model, inputs=None, labels=None)


def test_reduce_bit_depth_supports_16_bit():
    inputs = torch.linspace(0.0, 1.0, 500)
    # 12-bit quantizes into 4095 levels; 500 distinct values in [0, 1] will retain their unique bins
    squeezed_12 = reduce_bit_depth(inputs, bit_depth=12)
    squeezed_8 = reduce_bit_depth(inputs, bit_depth=8)
    # Squeezed 12 has higher precision than squeezed 8 (not clamped to 8)
    assert not torch.allclose(squeezed_12, squeezed_8)

    defense = FeatureSqueezingDefense(bit_depth=12)
    res = defense.apply(model=DummyClassifier(in_features=500, num_classes=2), inputs=inputs.unsqueeze(0))
    assert res.success is True
    assert res.metadata.parameters["bit_depth"] == 12


def test_preprocessing_wrapper_prevents_double_application():
    from app.hardening.defenses.preprocessing import SpatialSmoothingDefense
    from app.hardening.defenses.wrapper import HardenedModelWrapper

    model = DummyConvClassifier()
    inputs = torch.rand(2, 3, 8, 8)
    defense = SpatialSmoothingDefense(kernel_size=3, sigma=1.0)
    result = defense.apply(model=model, inputs=inputs)
    assert isinstance(result.hardened_model, HardenedModelWrapper)

    # Output from passing raw inputs (filter applied once inside wrapped model)
    out_raw = result.hardened_model(inputs)

    # Output from passing hardened_inputs (guard prevents filter from running a second time)
    out_hardened = result.hardened_model(result.hardened_inputs)

    # Both must match exactly since filter was applied once in both paths
    assert torch.allclose(out_raw, out_hardened, atol=1e-6)

    # Explicit flag already_preprocessed=True also bypasses re-filtering
    out_flag = result.hardened_model(result.hardened_inputs, already_preprocessed=True)
    assert torch.allclose(out_raw, out_flag, atol=1e-6)

