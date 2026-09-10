"""
Unit tests for Carlini & Wagner (C&W) L2 attack implementation.
"""

import pytest
import torch
import torch.nn as nn

from app.attack_engine.attacks.image.cw import CW
from app.attack_engine.base.base_attack import BaseAttack
from app.attack_engine.base.attack_metadata import AttackMetadata
from app.attack_engine.config import AttackConfig
from app.attack_engine.exceptions import AttackExecutionError, AttackConfigurationError
from app.attack_engine.registry import get_attack, list_attacks
from app.attack_engine.discovery import discover_attacks


class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)
        nn.init.constant_(self.fc.weight[0], 1.0)
        nn.init.constant_(self.fc.weight[1], -0.5)
        nn.init.constant_(self.fc.weight[2], -0.5)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        return self.fc(x)


def test_cw_registration():
    discover_attacks()
    attacks = list_attacks()
    assert "cw" in attacks
    assert "carlini_wagner" in attacks
    cls = get_attack("cw")
    assert cls.__name__ == "CW"
    assert issubclass(cls, BaseAttack)


def test_cw_metadata():
    metadata = CW.get_metadata()
    assert isinstance(metadata, AttackMetadata)
    assert metadata.name == "cw"
    assert metadata.domain == "image"
    assert metadata.category == "optimization"
    assert metadata.attack_type == "white_box"
    assert metadata.requires_gradient is True
    assert metadata.requires_loss is True
    assert metadata.iterative is True
    assert metadata.targeted_supported is True
    assert metadata.untargeted_supported is True
    assert metadata.computational_cost == "high"
    assert "L2" in metadata.supported_norms


def test_cw_generation_basic():
    model = SimpleClassifier()
    attack = CW(model)
    config = AttackConfig(
        clip_min=0.0,
        clip_max=1.0,
        params={"max_iterations": 15, "learning_rate": 0.05, "binary_search_steps": 1},
    )

    inputs = torch.full((2, 4), 0.5)
    labels = torch.tensor([0, 0])

    adv_inputs = attack.generate(inputs, labels, config)

    assert isinstance(adv_inputs, torch.Tensor)
    assert adv_inputs.shape == inputs.shape
    assert (adv_inputs >= 0.0).all()
    assert (adv_inputs <= 1.0).all()


def test_cw_targeted():
    model = SimpleClassifier()
    attack = CW(model)
    config = AttackConfig(
        clip_min=0.0,
        clip_max=1.0,
        params={"max_iterations": 10, "targeted": True},
    )

    inputs = torch.full((2, 4), 0.5)
    target_labels = torch.tensor([1, 2])

    adv_inputs = attack.generate(inputs, target_labels, config)
    assert isinstance(adv_inputs, torch.Tensor)
    assert adv_inputs.shape == inputs.shape


def test_cw_invalid_input_type():
    model = SimpleClassifier()
    attack = CW(model)

    with pytest.raises(AttackExecutionError):
        attack.generate(inputs="not_a_tensor", labels=[0])


def test_cw_invalid_config():
    model = SimpleClassifier()
    attack = CW(model)
    with pytest.raises(AttackConfigurationError):
        config = AttackConfig(clip_min=1.0, clip_max=0.0)
        attack.validate_config(config)


def test_cw_model_state_restoration():
    model = SimpleClassifier()
    attack = CW(model)
    inputs = torch.full((2, 4), 0.5)
    labels = torch.tensor([0, 0])
    config = AttackConfig(params={"max_iterations": 2, "binary_search_steps": 1})

    model.train()
    assert model.training is True
    attack.generate(inputs, labels, config)
    assert model.training is True

    model.eval()
    assert model.training is False
    attack.generate(inputs, labels, config)
    assert model.training is False


def test_cw_no_parameter_gradients_accumulated():
    model = SimpleClassifier()
    attack = CW(model)
    inputs = torch.full((2, 4), 0.5)
    labels = torch.tensor([0, 0])
    config = AttackConfig(params={"max_iterations": 2, "binary_search_steps": 1})

    attack.generate(inputs, labels, config)
    for param in model.parameters():
        assert param.grad is None


def test_cw_binary_search_steps():
    model = SimpleClassifier()
    attack = CW(model)
    inputs = torch.full((2, 4), 0.5)
    labels = torch.tensor([0, 0])
    config = AttackConfig(
        clip_min=0.0,
        clip_max=1.0,
        params={"max_iterations": 10, "binary_search_steps": 3, "initial_const": 1.0},
    )
    adv_inputs = attack.generate(inputs, labels, config)
    assert isinstance(adv_inputs, torch.Tensor)
    assert adv_inputs.shape == inputs.shape
    assert (adv_inputs >= 0.0).all()
    assert (adv_inputs <= 1.0).all()

