"""
Unit tests for BaseAttack interface and AttackMetadata.
"""

import pytest
import torch
import torch.nn as nn

from app.attack_engine.attacks.base_attack import BaseAttack
from app.attack_engine.models import AttackMetadata
from app.attack_engine.config import AttackConfig
from app.ingestion.adapters.pytorch_adapter import PyTorchAdapter


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc(x)


class ConcreteAttack(BaseAttack):
    metadata = AttackMetadata(
        name="concrete",
        domain="image",
        category="gradient",
        attack_type="white_box",
    )

    def generate(self, inputs, labels, config=None):
        return inputs + 0.1


def test_base_attack_metadata_exposure():
    attack_cls = ConcreteAttack
    metadata = attack_cls.get_metadata()
    assert metadata is not None
    assert metadata.name == "concrete"
    assert metadata.domain == "image"
    assert metadata.category == "gradient"
    assert metadata.attack_type == "white_box"
    assert metadata.type == "white_box"
    assert metadata.modality == "image"
    assert metadata.attack_category == "gradient"


def test_base_attack_instance_metadata():
    model = DummyModel()
    attack = ConcreteAttack(model)
    assert attack.attack_metadata.name == "concrete"


def test_base_attack_model_unwrapping():
    model = DummyModel()
    adapter = PyTorchAdapter(model)
    attack = ConcreteAttack(adapter)

    # Test that _get_raw_model() returns the underlying PyTorch nn.Module
    raw = attack._get_raw_model()
    assert raw is model
    assert isinstance(raw, nn.Module)


def test_base_attack_validate_config_default():
    model = DummyModel()
    attack = ConcreteAttack(model)
    # Default validate_config should not raise for valid config
    attack.validate_config(AttackConfig(epsilon=0.1))


def test_base_attack_abstract_cannot_instantiate():
    class IncompleteAttack(BaseAttack):
        pass

    with pytest.raises(TypeError):
        IncompleteAttack(DummyModel())  # type: ignore[abstract]
