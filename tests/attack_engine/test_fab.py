"""
Unit tests for Fast Adaptive Boundary (FAB) attack implementation.
"""

import pytest
import torch
import torch.nn as nn

from app.attack_engine.attacks.image.fab import FAB
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


def test_fab_registration():
    discover_attacks()
    attacks = list_attacks()
    assert "fab" in attacks
    cls = get_attack("fab")
    assert cls.__name__ == "FAB"
    assert issubclass(cls, BaseAttack)


def test_fab_metadata():
    metadata = FAB.get_metadata()
    assert isinstance(metadata, AttackMetadata)
    assert metadata.name == "fab"
    assert metadata.domain == "image"
    assert metadata.category == "boundary"
    assert metadata.attack_type == "white_box"
    assert metadata.requires_gradient is True
    assert metadata.requires_loss is False
    assert metadata.iterative is True
    assert metadata.targeted_supported is True
    assert metadata.untargeted_supported is True
    assert metadata.computational_cost == "medium"
    assert "Linf" in metadata.supported_norms


def test_fab_generation_basic():
    model = SimpleClassifier()
    attack = FAB(model)
    config = AttackConfig(
        epsilon=0.3,
        clip_min=0.0,
        clip_max=1.0,
        params={"max_iter": 10, "alpha_max": 0.1, "eta": 1.05},
    )

    inputs = torch.full((2, 4), 0.5)
    labels = torch.tensor([0, 0])

    adv_inputs = attack.generate(inputs, labels, config)

    assert isinstance(adv_inputs, torch.Tensor)
    assert adv_inputs.shape == inputs.shape
    assert (adv_inputs >= 0.0).all()
    assert (adv_inputs <= 1.0).all()


def test_fab_l2_norm():
    model = SimpleClassifier()
    attack = FAB(model)
    config = AttackConfig(
        epsilon=1.0,
        clip_min=0.0,
        clip_max=1.0,
        params={"max_iter": 5, "norm": "L2"},
    )

    inputs = torch.full((2, 4), 0.5)
    labels = torch.tensor([0, 0])

    adv_inputs = attack.generate(inputs, labels, config)
    assert isinstance(adv_inputs, torch.Tensor)
    assert adv_inputs.shape == inputs.shape


def test_fab_invalid_input_type():
    model = SimpleClassifier()
    attack = FAB(model)

    with pytest.raises(AttackExecutionError):
        attack.generate(inputs="not_a_tensor", labels=[0])


def test_fab_invalid_config():
    model = SimpleClassifier()
    attack = FAB(model)
    with pytest.raises(AttackConfigurationError):
        config = AttackConfig(clip_min=0.8, clip_max=0.2)
        attack.validate_config(config)


def test_fab_model_state_restoration():
    model = SimpleClassifier()
    attack = FAB(model)
    inputs = torch.full((2, 4), 0.5)
    labels = torch.tensor([0, 0])
    config = AttackConfig(params={"max_iter": 2})

    model.train()
    assert model.training is True
    attack.generate(inputs, labels, config)
    assert model.training is True

    model.eval()
    assert model.training is False
    attack.generate(inputs, labels, config)
    assert model.training is False


def test_fab_no_parameter_gradients_accumulated():
    model = SimpleClassifier()
    attack = FAB(model)
    inputs = torch.full((2, 4), 0.5)
    labels = torch.tensor([0, 0])
    config = AttackConfig(params={"max_iter": 2})

    attack.generate(inputs, labels, config)
    for param in model.parameters():
        assert param.grad is None


def test_fab_parameter_validation():
    model = SimpleClassifier()
    attack = FAB(model)
    inputs = torch.full((2, 4), 0.5)
    labels = torch.tensor([0, 0])

    # Test max_iter validation
    with pytest.raises(AttackExecutionError, match="FAB max_iter must be greater than 0"):
        attack.generate(inputs, labels, AttackConfig(params={"max_iter": 0}))

    with pytest.raises(AttackExecutionError, match="FAB max_iter must be greater than 0"):
        attack.generate(inputs, labels, AttackConfig(params={"max_iter": -1}))

    # Test top_k validation
    with pytest.raises(AttackExecutionError, match="FAB top_k must be greater than 0"):
        attack.generate(inputs, labels, AttackConfig(params={"top_k": 0}))

    with pytest.raises(AttackExecutionError, match="FAB top_k must be greater than 0"):
        attack.generate(inputs, labels, AttackConfig(params={"top_k": -2}))

    # Test norm validation
    with pytest.raises(AttackExecutionError, match="Unsupported norm 'L1' for FAB"):
        attack.generate(inputs, labels, AttackConfig(params={"norm": "L1"}))

    with pytest.raises(AttackExecutionError, match="Unsupported norm"):
        attack.generate(inputs, labels, AttackConfig(params={"norm": "L0"}))

    # Also test directly via validate_config
    with pytest.raises(AttackExecutionError, match="FAB max_iter must be greater than 0"):
        attack.validate_config(AttackConfig(params={"max_iter": 0}))

    with pytest.raises(AttackExecutionError, match="FAB top_k must be greater than 0"):
        attack.validate_config(AttackConfig(params={"top_k": 0}))

    with pytest.raises(AttackExecutionError, match="Unsupported norm 'L1' for FAB"):
        attack.validate_config(AttackConfig(params={"norm": "L1"}))

    # Test alpha_max and eta validation
    with pytest.raises(AttackExecutionError, match="FAB alpha_max must be a positive float"):
        attack.validate_config(AttackConfig(params={"alpha_max": -0.1}))

    with pytest.raises(AttackExecutionError, match="FAB eta must be a float >= 1.0"):
        attack.validate_config(AttackConfig(params={"eta": 0.9}))


class FoolableClassifier(nn.Module):
    """A linear classifier with distinct decision boundaries in [0, 1]^4."""

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


class ConstantUnfoolableClassifier(nn.Module):
    """Classifier with constant logits; cannot be fooled regardless of perturbation."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)
        with torch.no_grad():
            self.fc.weight.zero_()
            self.fc.bias.copy_(torch.tensor([1000.0, -1000.0, -1000.0]))

    def forward(self, x):
        return self.fc(x)


class LinearBoundaryModel(nn.Module):
    """2-class linear classifier with explicit decision boundary."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([
                [1.0, 0.0],
                [0.0, 2.0],
            ]))
            self.fc.bias.copy_(torch.tensor([0.5, 0.0]))

    def forward(self, x):
        return self.fc(x)


def test_fab_targeted_and_untargeted_success_conditions():
    """Verify targeted vs untargeted success criteria and target label validation."""
    model = FoolableClassifier()
    attack = FAB(model)

    inputs = torch.tensor([[0.6, 0.6, 0.1, 0.1]])
    true_label = torch.tensor([0])
    assert model(inputs).argmax(dim=-1).item() == 0

    # 1. Untargeted: prediction should change to any class != true_label
    config_untargeted = AttackConfig(
        epsilon=1.0,
        clip_min=0.0,
        clip_max=1.0,
        params={"max_iter": 20, "alpha_max": 0.2, "eta": 1.05, "targeted": False},
    )
    adv_untargeted = attack.generate(inputs, true_label, config_untargeted)
    pred_untargeted = model(adv_untargeted).argmax(dim=-1)
    assert pred_untargeted.item() != true_label.item()

    # 2. Targeted: prediction should specifically equal target class
    target_label = torch.tensor([2])
    config_targeted = AttackConfig(
        epsilon=1.0,
        clip_min=0.0,
        clip_max=1.0,
        params={"max_iter": 25, "alpha_max": 0.2, "eta": 1.05, "targeted": True, "target_labels": target_label},
    )
    adv_targeted = attack.generate(inputs, true_label, config_targeted)
    pred_targeted = model(adv_targeted).argmax(dim=-1)
    assert pred_targeted.item() == target_label.item()

    # 3. Targeted validation: target label identical to original label must raise error
    with pytest.raises(AttackExecutionError, match="Target labels must be supplied and differ from original labels"):
        invalid_config = AttackConfig(
            params={"targeted": True, "target_labels": true_label}
        )
        attack.generate(inputs, true_label, invalid_config)


def test_fab_boundary_projection_linf_and_l2():
    """Verify linear decision boundary projections for Linf (L1 dual) and L2 (L2 dual)."""
    model = LinearBoundaryModel()
    attack = FAB(model)

    inputs = torch.tensor([[0.5, 0.0]])
    labels = torch.tensor([0])
    assert model(inputs).argmax(dim=-1).item() == 0

    # L_infinity projection
    config_linf = AttackConfig(
        epsilon=1.0,
        params={"max_iter": 15, "alpha_max": 0.5, "eta": 1.05, "norm": "Linf"}
    )
    adv_linf = attack.generate(inputs, labels, config_linf)
    pred_linf = model(adv_linf).argmax(dim=-1).item()
    assert pred_linf == 1
    linf_pert = (adv_linf - inputs).abs().max().item()
    assert linf_pert > 0.0

    # L_2 projection
    config_l2 = AttackConfig(
        epsilon=1.0,
        params={"max_iter": 15, "alpha_max": 0.5, "eta": 1.05, "norm": "L2"}
    )
    adv_l2 = attack.generate(inputs, labels, config_l2)
    pred_l2 = model(adv_l2).argmax(dim=-1).item()
    assert pred_l2 == 1
    l2_pert = torch.norm((adv_l2 - inputs).view(-1), p=2).item()
    assert l2_pert > 0.0
    l2_pert = torch.norm((adv_l2 - inputs).view(-1), p=2).item()
    assert l2_pert > 0.0


def test_fab_failed_sample_preserves_original_input():
    """Verify that when an attack cannot find an adversarial example, original inputs are preserved."""
    model = ConstantUnfoolableClassifier()
    attack = FAB(model)
    inputs = torch.tensor([[0.2, 0.4, 0.6, 0.8], [0.1, 0.3, 0.5, 0.7]])
    labels = torch.tensor([0, 0])
    config = AttackConfig(
        epsilon=0.1,
        params={"max_iter": 5},
    )

    adv_inputs = attack.generate(inputs, labels, config)
    # Original input must be preserved untouched
    assert torch.allclose(adv_inputs, inputs, atol=1e-6)
    delta = (adv_inputs - inputs).view(2, -1)
    assert torch.all(torch.norm(delta, p=2, dim=1) == 0.0)


def test_fab_normalized_input_preservation():
    """Verify that normalized input tensors outside [0, 1] are not blindly clipped."""
    model = FoolableClassifier()
    attack = FAB(model)

    # Input with negative and >1.0 values (typical of normalized feature space)
    inputs = torch.tensor([[-1.5, -0.5, 1.2, 2.0]])
    labels = torch.tensor([0])
    # Default config has clip_min=0.0, clip_max=1.0
    config = AttackConfig(clip_min=0.0, clip_max=1.0, params={"max_iter": 5})

    adv_inputs = attack.generate(inputs, labels, config)
    # The negative values should NOT have been forced to 0.0 by blind clipping
    assert adv_inputs[0, 0].item() < -1.0


def test_fab_label_and_output_validation():
    """Verify label bounds and output shape validation."""
    model = SimpleClassifier()
    attack = FAB(model)
    inputs = torch.full((2, 4), 0.5)

    # Out of range labels
    with pytest.raises(AttackExecutionError, match="Labels must be in range"):
        attack.generate(inputs, labels=torch.tensor([0, 99]))

    with pytest.raises(AttackExecutionError, match="Labels must be in range"):
        attack.generate(inputs, labels=torch.tensor([-1, 0]))

    # Target labels out of range
    with pytest.raises(AttackExecutionError, match="Target labels must be in range"):
        attack.generate(
            inputs,
            labels=torch.tensor([0, 0]),
            config=AttackConfig(params={"targeted": True, "target_labels": torch.tensor([0, 99])})
        )


def test_fab_epsilon_budget_constraint():
    """Verify that perturbations strictly respect the specified epsilon constraint."""
    model = FoolableClassifier()
    attack = FAB(model)

    inputs = torch.tensor([[0.6, 0.6, 0.1, 0.1]])
    labels = torch.tensor([0])
    epsilon = 0.35

    config_linf = AttackConfig(
        epsilon=epsilon,
        clip_min=0.0,
        clip_max=1.0,
        params={"norm": "Linf", "max_iter": 15, "alpha_max": 0.3, "eta": 1.05},
    )
    adv_linf = attack.generate(inputs, labels, config_linf)
    linf_dist = (adv_linf - inputs).abs().max().item()
    assert linf_dist <= epsilon + 1e-5

    config_l2 = AttackConfig(
        epsilon=epsilon,
        clip_min=0.0,
        clip_max=1.0,
        params={"norm": "L2", "max_iter": 15, "alpha_max": 0.3, "eta": 1.05},
    )
    adv_l2 = attack.generate(inputs, labels, config_l2)
    l2_dist = torch.norm((adv_l2 - inputs).view(-1), p=2).item()
    assert l2_dist <= epsilon + 1e-5



