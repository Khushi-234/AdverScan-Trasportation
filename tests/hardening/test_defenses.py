"""
Unit tests for Module 7 defenses (preprocessing, smoothing, adversarial training).
"""

import pytest
import torch
import torch.nn as nn

from app.hardening.defenses import (
    SpatialSmoothingDefense,
    BitDepthReductionDefense,
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
    defense = BitDepthReductionDefense(bit_depth=4)

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
    defense = PreprocessingDefense(methods=["spatial_smoothing", "bit_depth_reduction"])

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
