"""
Unit and integration tests for Data Augmentation defense.
"""

import pytest
import torch
import torch.nn as nn

from app.hardening.defenses.data_augmentation import DataAugmentationDefense
from app.hardening.defenses import get_defense_class
from app.hardening.hardening_engine import HardeningEngine
from app.hardening.hardening_result import HardeningResult


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        return self.fc(x)


def test_data_augmentation_tensor_transform():
    defense = DataAugmentationDefense(
        noise_std=0.02,
        flip_prob=0.5,
        brightness_jitter=0.1,
        contrast_jitter=0.1,
        crop_scale=(0.9, 1.0),
    )
    inputs = torch.ones(4, 3, 16, 16)
    out = defense.transform_tensor(inputs)

    assert out.shape == inputs.shape
    assert not torch.allclose(out, inputs)


def test_data_augmentation_apply_batch_mode():
    model = DummyModel()
    defense = DataAugmentationDefense(noise_std=0.02)
    inputs = torch.randn(4, 8)

    result = defense.apply(model=model, inputs=inputs)
    assert isinstance(result, HardeningResult)
    assert result.success is True
    assert result.hardened_inputs is not None
    assert result.hardened_inputs.shape == inputs.shape
    assert result.hardened_model == model
    assert "noise_std" in result.metadata.parameters
    assert result.metadata.execution_time_seconds >= 0.0


def test_data_augmentation_registry_lookup():
    def_cls = get_defense_class("data_augmentation")
    assert def_cls == DataAugmentationDefense


def test_hardening_engine_with_data_augmentation():
    engine = HardeningEngine()
    model = DummyModel()
    inputs = torch.randn(5, 8)

    result = engine.harden(
        model=model,
        defense="data_augmentation",
        inputs=inputs,
        defense_config={"noise_std": 0.02, "brightness_jitter": 0.1},
    )

    assert result.success is True
    assert result.metadata.defense_name == "data_augmentation"
    assert result.hardened_inputs is not None
    assert result.hardened_inputs.shape == inputs.shape
