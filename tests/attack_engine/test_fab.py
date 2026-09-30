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

