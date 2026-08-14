"""
Unit tests for PGD attack implementation.
"""

import torch
import torch.nn as nn
import pytest

from app.attack_engine.attacks.pgd import PGD
from app.attack_engine.config import AttackConfig
from app.attack_engine.exceptions import AttackExecutionError
from app.attack_engine.attack_registry import get_attack, list_attacks
from app.attack_engine.attack_discovery import discover_attacks


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        nn.init.constant_(self.fc.weight[0], 0.5)
        nn.init.constant_(self.fc.weight[1], -0.5)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        return self.fc(x)


def test_pgd_registration():
    discover_attacks()
    attacks = list_attacks()
    assert "pgd" in attacks
    assert "pgm" in attacks
    cls = get_attack("pgd")
    assert cls.__name__ == "PGD"


def test_pgd_generation_basic():
    model = SimpleModel()
    pgd = PGD(model)
    config = AttackConfig(
        epsilon=0.1,
        clip_min=0.0,
        clip_max=1.0,
        params={"num_steps": 5, "alpha": 0.02, "random_start": True},
    )

    inputs = torch.full((4, 10), 0.5)
    labels = torch.tensor([0, 1, 0, 1])

    adv_inputs = pgd.generate(inputs, labels, config)

    assert isinstance(adv_inputs, torch.Tensor)
    assert adv_inputs.shape == inputs.shape
    assert not torch.equal(adv_inputs, inputs)
    diff = (adv_inputs - inputs).abs()
    assert (diff <= 0.1001).all()


def test_pgd_clipping():
    model = SimpleModel()
    pgd = PGD(model)
    config = AttackConfig(
        epsilon=0.5,
        clip_min=0.2,
        clip_max=0.7,
        params={"num_steps": 10, "alpha": 0.1, "random_start": False},
    )

    inputs = torch.full((2, 10), 0.5)
    labels = torch.tensor([0, 1])

    adv_inputs = pgd.generate(inputs, labels, config)

    assert (adv_inputs >= 0.2).all()
    assert (adv_inputs <= 0.7).all()


def test_pgd_invalid_input_type():
    model = SimpleModel()
    pgd = PGD(model)
    config = AttackConfig()

    with pytest.raises(AttackExecutionError):
        pgd.generate(inputs="invalid_inputs", labels=[0], config=config)
