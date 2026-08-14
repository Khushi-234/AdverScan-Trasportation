"""
Unit tests for DeepFool attack implementation.
"""

import pytest
import torch
import torch.nn as nn

from app.attack_engine.attacks.deepfool import DeepFool
from app.attack_engine.base.base_attack import BaseAttack
from app.attack_engine.config import AttackConfig
from app.attack_engine.exceptions import AttackExecutionError
from app.attack_engine.attack_registry import get_attack, list_attacks
from app.attack_engine.attack_discovery import discover_attacks


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


def test_deepfool_registration():
    discover_attacks()
    attacks = list_attacks()
    assert "deepfool" in attacks
    cls = get_attack("deepfool")
    assert cls.__name__ == "DeepFool"
    assert issubclass(cls, BaseAttack)


def test_deepfool_generation_basic():
    model = SimpleClassifier()
    attack = DeepFool(model)
    config = AttackConfig(
        epsilon=0.2,
        clip_min=0.0,
        clip_max=1.0,
        params={"max_iter": 10, "overshoot": 0.02},
    )

    inputs = torch.full((2, 4), 0.5)
    labels = torch.tensor([0, 0])

    adv_inputs = attack.generate(inputs, labels, config)

    assert isinstance(adv_inputs, torch.Tensor)
    assert adv_inputs.shape == inputs.shape


def test_deepfool_invalid_input_type():
    model = SimpleClassifier()
    attack = DeepFool(model)

    with pytest.raises(AttackExecutionError):
        attack.generate(inputs="not_a_tensor", labels=[0])
