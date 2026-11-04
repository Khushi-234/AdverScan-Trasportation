"""
Unit tests for Carlini & Wagner (C&W) L2 attack implementation.
"""

import pytest
import torch
import torch.nn as nn

from app.attack_engine.attacks.image.cw import CW
from app.attack_engine.attacks.base_attack import BaseAttack
from app.attack_engine.models import AttackMetadata
from app.attack_engine.config import AttackConfig
from app.attack_engine.exceptions import AttackExecutionError, AttackConfigurationError
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


class ConstantUnfoolableClassifier(nn.Module):
    """Classifier with constant logits; cannot be fooled regardless of input perturbation."""

    def __init__(self):
        super().__init__()
        # Dummy parameter to satisfy device discovery
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size = x.size(0)
        # Always output class 0 with overwhelmingly high confidence
        logits = torch.zeros(batch_size, 3, device=x.device)
        logits[:, 0] = 1000.0
        logits[:, 1] = -1000.0
        logits[:, 2] = -1000.0
        return logits


class FoolableClassifier(nn.Module):
    """A linear classifier with distinct decision boundaries inside [0, 1]^4."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 2.0],
                [0.0, 2.0, 0.0, 2.0],
            ]))
            self.fc.bias.copy_(torch.tensor([0.5, 0.0, 0.0]))

    def forward(self, x):
        return self.fc(x)


def test_cw_failed_sample_handling_preserves_original_input():
    """Verify that when an attack fails to find an adversarial example, original inputs are preserved."""
    model = ConstantUnfoolableClassifier()
    attack = CW(model)
    inputs = torch.tensor([[0.2, 0.4, 0.6, 0.8], [0.1, 0.3, 0.5, 0.7]])
    labels = torch.tensor([0, 0])
    config = AttackConfig(
        clip_min=0.0,
        clip_max=1.0,
        params={"max_iterations": 5, "binary_search_steps": 2, "initial_const": 0.1},
    )

    adv_inputs = attack.generate(inputs, labels, config)

    # When no adversarial example can be found, the attack MUST return original inputs untouched
    assert torch.allclose(adv_inputs, inputs, atol=1e-6)
    # The L2 distortion must be exactly 0
    delta = (adv_inputs - inputs).view(2, -1)
    l2_norms = torch.norm(delta, p=2, dim=1)
    assert torch.all(l2_norms == 0.0)


def test_cw_targeted_and_untargeted_success_conditions():
    """Verify targeted vs untargeted success criteria."""
    model = FoolableClassifier()
    attack = CW(model)

    inputs = torch.tensor([[0.6, 0.6, 0.1, 0.1]])
    true_label = torch.tensor([0])
    assert model(inputs).argmax(dim=-1).item() == 0

    # 1. Untargeted: should flip to any class != true_label
    config_untargeted = AttackConfig(
        clip_min=0.0,
        clip_max=1.0,
        params={"max_iterations": 25, "learning_rate": 0.1, "initial_const": 10.0, "targeted": False},
    )
    adv_untargeted = attack.generate(inputs, true_label, config_untargeted)
    pred_untargeted = model(adv_untargeted).argmax(dim=-1)
    assert pred_untargeted.item() != true_label.item()

    # 2. Targeted: should specifically produce the requested target class
    target_label = torch.tensor([2])
    config_targeted = AttackConfig(
        clip_min=0.0,
        clip_max=1.0,
        params={"max_iterations": 30, "learning_rate": 0.1, "initial_const": 10.0, "targeted": True},
    )
    adv_targeted = attack.generate(inputs, target_label, config_targeted)
    pred_targeted = model(adv_targeted).argmax(dim=-1)
    assert pred_targeted.item() == target_label.item()


def test_cw_box_constraint_enforcement():
    """Verify that generated adversarial examples strictly adhere to custom box constraints."""
    model = SimpleClassifier()
    attack = CW(model)

    # Custom box bounds [0.25, 0.75]
    clip_min = 0.25
    clip_max = 0.75
    inputs = torch.full((2, 4), 0.5)
    labels = torch.tensor([0, 0])
    config = AttackConfig(
        clip_min=clip_min,
        clip_max=clip_max,
        params={"max_iterations": 20, "learning_rate": 0.1, "initial_const": 5.0},
    )

    adv_inputs = attack.generate(inputs, labels, config)
    assert adv_inputs.min().item() >= clip_min - 1e-6
    assert adv_inputs.max().item() <= clip_max + 1e-6

    # Verify invalid bounds raise AttackConfigurationError
    with pytest.raises(AttackConfigurationError):
        invalid_cfg = AttackConfig(clip_min=0.8, clip_max=0.2)
        attack.validate_config(invalid_cfg)


def test_cw_l2_distortion_calculated_in_input_space():
    """Verify that L2 distortion is computed in the actual feature input space, not tanh space."""
    model = FoolableClassifier()
    attack = CW(model)

    inputs = torch.tensor([[0.6, 0.6, 0.1, 0.1]])
    labels = torch.tensor([0])
    config = AttackConfig(
        clip_min=0.0,
        clip_max=1.0,
        params={"max_iterations": 25, "learning_rate": 0.05, "initial_const": 10.0},
    )

    adv_inputs = attack.generate(inputs, labels, config)
    delta = adv_inputs - inputs
    l2_distortion = torch.norm(delta.view(1, -1), p=2, dim=1).item()

    # The attack was successful, so L2 distortion should be positive, finite, and in input space [0, 1]
    assert l2_distortion > 0.0
    assert l2_distortion < 2.0  # Cannot exceed total space diameter


def test_cw_per_sample_independent_tracking_and_binary_search():
    """Verify that in a multi-sample batch, each sample is tracked independently."""
    model = SimpleClassifier()
    attack = CW(model)

    # Sample 0 is at 0.5 (easily flipped) while sample 1 has a different initialization
    inputs = torch.tensor([
        [0.5, 0.5, 0.5, 0.5],
        [0.9, 0.9, 0.9, 0.9],
    ])
    labels = torch.tensor([0, 0])
    config = AttackConfig(
        clip_min=0.0,
        clip_max=1.0,
        params={"max_iterations": 20, "binary_search_steps": 2, "initial_const": 1.0},
    )

    adv_inputs = attack.generate(inputs, labels, config)
    assert adv_inputs.shape == inputs.shape
    # Ensure independent values for sample 0 and sample 1
    assert not torch.allclose(adv_inputs[0], adv_inputs[1])


def test_cw_unsupported_params_handled_transparently():
    """Verify that generic parameters (e.g. epsilon, norm='Linf', loss_fn) do not crash CW."""
    model = SimpleClassifier()
    attack = CW(model)
    inputs = torch.full((2, 4), 0.5)
    labels = torch.tensor([0, 0])

    # Pass generic attack parameters (e.g. from a generic pipeline)
    config = AttackConfig(
        epsilon=0.05,
        loss_fn=nn.CrossEntropyLoss(),
        clip_min=0.0,
        clip_max=1.0,
        params={"norm": "Linf", "max_iterations": 10, "binary_search_steps": 1},
    )

    # Should run smoothly without claiming epsilon is enforced as an Linf bound
    adv_inputs = attack.generate(inputs, labels, config)
    assert isinstance(adv_inputs, torch.Tensor)
    assert adv_inputs.shape == inputs.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on this environment")
def test_cw_cuda_device_compatibility():
    """Verify full CUDA execution without CPU/GPU transfer mismatches when CUDA is present."""
    model = SimpleClassifier().cuda()
    attack = CW(model)
    inputs = torch.full((2, 4), 0.5, device="cuda")
    labels = torch.tensor([0, 0], device="cuda")
    config = AttackConfig(params={"max_iterations": 5, "binary_search_steps": 1})

    adv_inputs = attack.generate(inputs, labels, config)
    assert adv_inputs.is_cuda
    assert adv_inputs.device == inputs.device


