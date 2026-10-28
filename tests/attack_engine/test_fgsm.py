"""
Unit tests for FGSM attack implementation.
"""

import torch
import torch.nn as nn
import pytest

from app.attack_engine.attacks.image.fgsm import FGSM
from app.attack_engine.config import AttackConfig
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


def test_fgsm_targeted_and_untargeted():
    model = SimpleModel()
    fgsm = FGSM(model)
    inputs = torch.full((1, 10), 0.5)
    label = torch.tensor([0])

    config_untargeted = AttackConfig(epsilon=0.1, params={"targeted": False})
    config_targeted = AttackConfig(epsilon=0.1, params={"targeted": True})

    # Untargeted on label 0 moves away (x + eps * sign(grad))
    adv_untargeted = fgsm.generate(inputs, label, config_untargeted)
    # Targeted on label 0 moves towards label 0 (x - eps * sign(grad))
    adv_targeted = fgsm.generate(inputs, label, config_targeted)

    assert adv_untargeted.requires_grad is False
    assert adv_targeted.requires_grad is False
    # Targeted and untargeted perturbations for the same label have opposite signs
    assert not torch.equal(adv_untargeted, adv_targeted)
    assert torch.allclose(adv_untargeted, torch.full((1, 10), 0.4))
    assert torch.allclose(adv_targeted, torch.full((1, 10), 0.6))


def test_fgsm_model_state_restoration():
    model = SimpleModel()
    fgsm = FGSM(model)

    # Test when model is set to training mode
    model.train()
    assert model.training is True
    inputs = torch.full((2, 10), 0.5)
    labels = torch.tensor([0, 1])
    config = AttackConfig(epsilon=0.1)

    fgsm.generate(inputs, labels, config)
    assert model.training is True  # State must be restored

    # Test when model is set to eval mode
    model.eval()
    assert model.training is False
    fgsm.generate(inputs, labels, config)
    assert model.training is False  # State must remain eval


def test_fgsm_no_parameter_gradients_accumulated():
    model = SimpleModel()
    fgsm = FGSM(model)
    inputs = torch.full((2, 10), 0.5)
    labels = torch.tensor([0, 1])
    config = AttackConfig(epsilon=0.1)

    fgsm.generate(inputs, labels, config)

    # Using torch.autograd.grad should not accumulate gradients in model parameters
    for param in model.parameters():
        assert param.grad is None


def test_fgsm_non_tensor_labels():
    model = SimpleModel()
    fgsm = FGSM(model)
    inputs = torch.full((2, 10), 0.5)
    labels_list = [0, 1]
    config = AttackConfig(epsilon=0.1)

    adv_inputs = fgsm.generate(inputs, labels_list, config)
    assert isinstance(adv_inputs, torch.Tensor)
    assert adv_inputs.shape == inputs.shape


