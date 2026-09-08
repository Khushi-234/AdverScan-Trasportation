"""
Unit tests for PGD attack implementation.
"""

import torch
import torch.nn as nn
import pytest

from app.attack_engine.attacks.image.pgd import PGD
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


def test_pgd_num_steps_validation():
    model = SimpleModel()
    pgd = PGD(model)
    inputs = torch.full((2, 10), 0.5)
    labels = torch.tensor([0, 1])

    with pytest.raises(AttackExecutionError, match="PGD num_steps must be greater than 0"):
        pgd.generate(inputs, labels, AttackConfig(params={"num_steps": 0}))

    with pytest.raises(AttackExecutionError, match="PGD num_steps must be greater than 0"):
        pgd.generate(inputs, labels, AttackConfig(params={"num_steps": -5}))


def test_pgd_l2_random_start():
    model = SimpleModel()
    pgd = PGD(model)
    inputs = torch.full((3, 10), 0.5)
    labels = torch.tensor([0, 1, 0])
    epsilon = 0.5
    config = AttackConfig(
        epsilon=epsilon,
        clip_min=0.0,
        clip_max=1.0,
        params={"norm": "L2", "num_steps": 5, "random_start": True},
    )

    adv_inputs = pgd.generate(inputs, labels, config)
    assert isinstance(adv_inputs, torch.Tensor)
    assert adv_inputs.shape == inputs.shape
    delta = adv_inputs - inputs
    l2_norms = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1)
    assert (l2_norms <= epsilon + 1e-4).all()


def test_pgd_model_state_restoration():
    model = SimpleModel()
    pgd = PGD(model)
    inputs = torch.full((2, 10), 0.5)
    labels = torch.tensor([0, 1])
    config = AttackConfig(epsilon=0.1, params={"num_steps": 2})

    # When model is training
    model.train()
    assert model.training is True
    pgd.generate(inputs, labels, config)
    assert model.training is True

    # When model is eval
    model.eval()
    assert model.training is False
    pgd.generate(inputs, labels, config)
    assert model.training is False


def test_pgd_no_parameter_gradients_accumulated():
    model = SimpleModel()
    pgd = PGD(model)
    inputs = torch.full((2, 10), 0.5)
    labels = torch.tensor([0, 1])
    config = AttackConfig(epsilon=0.1, params={"num_steps": 2})

    pgd.generate(inputs, labels, config)
    for param in model.parameters():
        assert param.grad is None
