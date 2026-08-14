"""
Unit tests for attack executor.
"""

import pytest
import torch
import torch.nn as nn

from app.attack_engine.attack_executor import execute_attack
from app.attack_engine.attacks.fgsm import FGSM
from app.attack_engine.config import AttackConfig
from app.attack_engine.exceptions import AttackExecutionError
from app.ingestion.adapters.pytorch_adapter import PyTorchAdapter


class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 2)
        nn.init.constant_(self.fc.weight[0], 0.3)
        nn.init.constant_(self.fc.weight[1], -0.3)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        return self.fc(x)


def test_execute_attack_raw_model():
    model = DummyNet()
    inputs = torch.ones((2, 5))
    labels = torch.tensor([0, 1])
    config = AttackConfig(epsilon=0.05)

    adv_inputs = execute_attack(model, FGSM, inputs, labels, config)
    assert isinstance(adv_inputs, torch.Tensor)
    assert adv_inputs.shape == inputs.shape
    assert not torch.equal(adv_inputs, inputs)


def test_execute_attack_with_pytorch_adapter():
    model = DummyNet()
    adapter = PyTorchAdapter(model)
    inputs = torch.ones((2, 5))
    labels = torch.tensor([0, 1])
    config = AttackConfig(epsilon=0.05)

    adv_inputs = execute_attack(adapter, FGSM, inputs, labels, config)
    assert isinstance(adv_inputs, torch.Tensor)
    assert adv_inputs.shape == inputs.shape


def test_execute_attack_invalid_class():
    model = DummyNet()
    inputs = torch.ones((2, 5))
    labels = torch.tensor([0, 1])

    class NotAnAttackClass:
        pass

    with pytest.raises(AttackExecutionError):
        execute_attack(model, NotAnAttackClass, inputs, labels)
