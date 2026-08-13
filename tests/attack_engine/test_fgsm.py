"""
Unit tests for FGSM attack implementation.
"""

import torch
import torch.nn as nn
import pytest

from app.attack_engine.attacks.fgsm import FGSM
from app.attack_engine.config.attack_config import AttackConfig
from app.attack_engine.exceptions import AttackExecutionError


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        nn.init.constant_(self.fc.weight[0], 0.5)
        nn.init.constant_(self.fc.weight[1], -0.5)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        return self.fc(x)


def test_fgsm_generation_basic():
    model = SimpleModel()
    fgsm = FGSM(model)
    config = AttackConfig(epsilon=0.1, clip_min=0.0, clip_max=1.0)

    inputs = torch.full((4, 10), 0.5)
    labels = torch.tensor([0, 1, 0, 1])

    adv_inputs = fgsm.generate(inputs, labels, config)

    assert isinstance(adv_inputs, torch.Tensor)
    assert adv_inputs.shape == inputs.shape
    assert not torch.equal(adv_inputs, inputs)
    # Check max perturbation magnitude is <= epsilon + floating point tolerance
    diff = (adv_inputs - inputs).abs()
    assert (diff <= 0.1001).all()


def test_fgsm_clipping():
    model = SimpleModel()
    fgsm = FGSM(model)
    config = AttackConfig(epsilon=0.5, clip_min=0.2, clip_max=0.7)

    inputs = torch.full((2, 10), 0.5)
    labels = torch.tensor([0, 1])

    adv_inputs = fgsm.generate(inputs, labels, config)

    assert (adv_inputs >= 0.2).all()
    assert (adv_inputs <= 0.7).all()


def test_fgsm_invalid_input_type():
    model = SimpleModel()
    fgsm = FGSM(model)
    config = AttackConfig()

    with pytest.raises(AttackExecutionError):
        fgsm.generate(inputs="invalid_inputs", labels=[0], config=config)


class DataclassOutputModel(nn.Module):
    """Model returning an object/dataclass containing a .logits attribute (e.g. HF SequenceClassifierOutput)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        nn.init.constant_(self.fc.weight[0], 0.5)
        nn.init.constant_(self.fc.weight[1], -0.5)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        from types import SimpleNamespace
        return SimpleNamespace(logits=self.fc(x))


def test_fgsm_dataclass_logits_output():
    """Verify FGSM works when model returns an object with a .logits attribute."""
    model = DataclassOutputModel()
    fgsm = FGSM(model)
    config = AttackConfig(epsilon=0.1, clip_min=0.0, clip_max=1.0)

    inputs = torch.full((4, 10), 0.5)
    labels = torch.tensor([0, 1, 0, 1])

    adv_inputs = fgsm.generate(inputs, labels, config)

    assert isinstance(adv_inputs, torch.Tensor)
    assert adv_inputs.shape == inputs.shape
    assert torch.isfinite(adv_inputs).all()
    assert not torch.equal(adv_inputs, inputs)


def test_attack_config_negative_epsilon():
    """Verify AttackConfig rejects negative epsilon values."""
    from app.attack_engine.exceptions import AttackConfigurationError

    with pytest.raises(AttackConfigurationError):
        AttackConfig(epsilon=-0.1)


def test_attack_config_invalid_clip_bounds():
    """Verify AttackConfig rejects clip_min > clip_max."""
    from app.attack_engine.exceptions import AttackConfigurationError

    with pytest.raises(AttackConfigurationError):
        AttackConfig(clip_min=1.0, clip_max=0.0)

