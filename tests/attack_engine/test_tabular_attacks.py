"""
Unit and integration tests for white-box tabular adversarial attacks in AdverScan:
- CapPGD (Constraint-Aware PGD)
- LowProFool (Feature-Weighted Perturbation)
- TabularCW (Carlini & Wagner for Tabular)
- TabularJSMA (Jacobian-based Saliency Map Attack for Tabular)
- VAEAttack (On-Manifold VAE Latent Attack)
"""

from typing import Dict, Any, Tuple
import pytest
import torch
import torch.nn as nn

from app.attack_engine.attacks.tabular.cap_pgd import CapPGD
from app.attack_engine.attacks.tabular.lowprofool import LowProFool
from app.attack_engine.attacks.tabular.carlini_wagner import TabularCW
from app.attack_engine.attacks.tabular.jsma import TabularJSMA
from app.attack_engine.attacks.tabular.vae_attack import VAEAttack
from app.attack_engine.config import AttackConfig
from app.attack_engine.exceptions import (
    AttackConfigurationError,
    AttackExecutionError,
)
from app.attack_engine.attack_registry import get_attack, list_attacks
from app.attack_engine.attack_discovery import discover_attacks
from app.attack_engine.attack_executor import execute_attack
from app.attack_engine.attack_engine import AttackEngine


class SimpleTabularClassifier(nn.Module):
    """Simple 2-layer MLP classifier for tabular data."""

    def __init__(self, num_features: int = 5, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleTabularVAE(nn.Module):
    """Lightweight deterministic encoder-decoder VAE mock for tabular tests."""

    def __init__(self, num_features: int = 5, latent_dim: int = 2):
        super().__init__()
        self.encoder = nn.Linear(num_features, latent_dim)
        self.decoder = nn.Linear(latent_dim, num_features)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return z, z  # (mu, logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


@pytest.fixture
def feature_metadata() -> Dict[str, Any]:
    return {
        "f_cont": {"type": "continuous", "min": 0.0, "max": 10.0, "mutable": True},
        "f_int": {"type": "integer", "min": 18.0, "max": 65.0, "mutable": True},
        "f_bin": {"type": "binary", "mutable": True},
        "f_cat": {
            "type": "categorical",
            "categories": [10.0, 20.0, 30.0],
            "mutable": True,
        },
        "f_imm": {"type": "continuous", "mutable": False},
    }


@pytest.fixture
def model():
    torch.manual_seed(42)
    m = SimpleTabularClassifier(num_features=5, num_classes=2)
    m.eval()
    return m


@pytest.fixture
def vae():
    torch.manual_seed(42)
    v = SimpleTabularVAE(num_features=5, latent_dim=2)
    v.eval()
    return v


@pytest.fixture
def sample_data():
    # Shape: (4, 5)
    # f_cont, f_int, f_bin, f_cat, f_imm
    x = torch.tensor(
        [
            [5.0, 25.0, 1.0, 10.0, 100.0],
            [2.0, 40.0, 0.0, 20.0, 200.0],
            [8.0, 30.0, 1.0, 30.0, 300.0],
            [1.0, 50.0, 0.0, 10.0, 400.0],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    return x, y


# =========================================================================
# Discovery and Registry Tests
# =========================================================================


def test_tabular_attacks_registered():
    discover_attacks()
    attacks = list_attacks()
    expected_attacks = [
        "cap_pgd",
        "lowprofool",
        "carlini_wagner_tabular",
        "jsma_tabular",
        "vae_attack",
    ]
    for name in expected_attacks:
        assert name in attacks
        cls = get_attack(name)
        assert cls is not None
        assert cls.metadata.domain == "tabular"
        assert cls.metadata.attack_type == "white_box"


# =========================================================================
# 1. CapPGD Tests
# =========================================================================


def test_cap_pgd_untargeted_and_constraints(model, sample_data, feature_metadata):
    x, y = sample_data
    attack = CapPGD(model)
    config = AttackConfig(
        epsilon=2.0,
        params={
            "num_steps": 5,
            "alpha": 0.5,
            "norm": "Linf",
            "feature_metadata": feature_metadata,
        },
    )

    x_adv = attack.generate(x, y, config)
    assert x_adv.shape == x.shape

    # 1. Immutable feature f_imm (column 4) must NOT change
    assert torch.allclose(x_adv[:, 4], x[:, 4])

    # 2. Continuous feature bounds [0.0, 10.0]
    assert (x_adv[:, 0] >= 0.0).all() and (x_adv[:, 0] <= 10.0).all()

    # 3. Integer feature f_int (column 1) must remain exact integers and in [18, 65]
    int_diff = (x_adv[:, 1] - torch.round(x_adv[:, 1])).abs()
    assert (int_diff < 1e-4).all()
    assert (x_adv[:, 1] >= 18.0).all() and (x_adv[:, 1] <= 65.0).all()

    # 4. Binary feature f_bin (column 2) must be in {0.0, 1.0}
    is_binary = ((x_adv[:, 2] == 0.0) | (x_adv[:, 2] == 1.0)).all()
    assert is_binary

    # 5. Categorical feature f_cat (column 3) must be in {10.0, 20.0, 30.0}
    for val in x_adv[:, 3].tolist():
        assert val in [10.0, 20.0, 30.0]

    # 6. Epsilon constraint (Linf <= epsilon) for mutable continuous features
    delta = (x_adv - x).abs()
    assert (delta <= 2.0 + 1e-4).all()


def test_cap_pgd_targeted_and_l2(model, sample_data, feature_metadata):
    x, y = sample_data
    attack = CapPGD(model)
    config = AttackConfig(
        epsilon=3.0,
        params={
            "targeted": True,
            "target_label": 1,
            "norm": "L2",
            "num_steps": 5,
            "alpha": 0.5,
            "feature_metadata": feature_metadata,
        },
    )

    x_adv = attack.generate(x, y, config)
    assert x_adv.shape == x.shape
    # Immutable feature unchanged
    assert torch.allclose(x_adv[:, 4], x[:, 4])


def test_cap_pgd_invalid_config(model):
    attack = CapPGD(model)
    with pytest.raises(AttackConfigurationError):
        attack.validate_config(AttackConfig(epsilon=-1.0))
    with pytest.raises(AttackConfigurationError):
        attack.validate_config(AttackConfig(params={"step_size": -0.1}))
    with pytest.raises(AttackConfigurationError):
        attack.validate_config(AttackConfig(params={"norm": "L3"}))


# =========================================================================
# 2. LowProFool Tests
# =========================================================================


def test_lowprofool_untargeted_and_weights(model, sample_data, feature_metadata):
    x, y = sample_data
    attack = LowProFool(model)
    config = AttackConfig(
        params={
            "num_steps": 10,
            "alpha": 0.1,
            "lambda_weight": 0.05,
            "feature_metadata": feature_metadata,
            "feature_weights": [1.0, 5.0, 1.0, 1.0, 100.0],
        }
    )

    x_adv = attack.generate(x, y, config)
    assert x_adv.shape == x.shape
    # Immutable feature preserved
    assert torch.allclose(x_adv[:, 4], x[:, 4])

    # Integer and binary constraints respected
    int_diff = (x_adv[:, 1] - torch.round(x_adv[:, 1])).abs()
    assert (int_diff < 1e-4).all()
    assert ((x_adv[:, 2] == 0.0) | (x_adv[:, 2] == 1.0)).all()

    # Returns statistics
    assert "weighted_distance" in config.params
    assert config.params["weighted_distance"] >= 0.0


def test_lowprofool_targeted(model, sample_data, feature_metadata):
    x, y = sample_data
    attack = LowProFool(model)
    config = AttackConfig(
        params={
            "targeted": True,
            "target_label": 0,
            "num_steps": 5,
            "feature_metadata": feature_metadata,
        }
    )
    x_adv = attack.generate(x, y, config)
    assert x_adv.shape == x.shape


def test_lowprofool_invalid_config(model):
    attack = LowProFool(model)
    with pytest.raises(AttackConfigurationError):
        attack.validate_config(AttackConfig(params={"alpha": -0.05}))
    with pytest.raises(AttackConfigurationError):
        attack.validate_config(AttackConfig(params={"num_steps": 0}))


# =========================================================================
# 3. TabularCW Tests
# =========================================================================


def test_tabular_cw_untargeted(model, sample_data, feature_metadata):
    x, y = sample_data
    attack = TabularCW(model)
    config = AttackConfig(
        params={
            "max_iterations": 15,
            "learning_rate": 0.05,
            "c_penalty": 1.0,
            "confidence": 0.0,
            "feature_metadata": feature_metadata,
        }
    )

    x_adv = attack.generate(x, y, config)
    assert x_adv.shape == x.shape
    # Immutable feature preserved
    assert torch.allclose(x_adv[:, 4], x[:, 4])
    # Bounds and integer checks
    assert (x_adv[:, 0] >= 0.0).all() and (x_adv[:, 0] <= 10.0).all()
    int_diff = (x_adv[:, 1] - torch.round(x_adv[:, 1])).abs()
    assert (int_diff < 1e-4).all()


def test_tabular_cw_targeted(model, sample_data, feature_metadata):
    x, y = sample_data
    attack = TabularCW(model)
    config = AttackConfig(
        params={
            "targeted": True,
            "target_label": 1,
            "max_iterations": 10,
            "learning_rate": 0.05,
            "feature_metadata": feature_metadata,
        }
    )
    x_adv = attack.generate(x, y, config)
    assert x_adv.shape == x.shape


def test_tabular_cw_invalid_config(model):
    attack = TabularCW(model)
    with pytest.raises(AttackConfigurationError):
        attack.validate_config(AttackConfig(params={"learning_rate": -0.1}))
    with pytest.raises(AttackConfigurationError):
        attack.validate_config(AttackConfig(params={"c_penalty": 0.0}))
    with pytest.raises(AttackConfigurationError):
        attack.validate_config(AttackConfig(params={"confidence": -1.0}))


# =========================================================================
# 4. TabularJSMA Tests
# =========================================================================


def test_tabular_jsma_untargeted_sparsity(model, sample_data, feature_metadata):
    x, y = sample_data
    attack = TabularJSMA(model)
    config = AttackConfig(
        params={
            "max_features": 2,
            "step_size": 1.0,
            "feature_metadata": feature_metadata,
        }
    )

    x_adv = attack.generate(x, y, config)
    assert x_adv.shape == x.shape
    # Immutable feature preserved
    assert torch.allclose(x_adv[:, 4], x[:, 4])

    # Sparsity: at most 2 features modified per sample
    num_changed = (x_adv != x).sum(dim=1)
    assert (num_changed <= 2).all()

    # Discrete constraints
    int_diff = (x_adv[:, 1] - torch.round(x_adv[:, 1])).abs()
    assert (int_diff < 1e-4).all()
    assert ((x_adv[:, 2] == 0.0) | (x_adv[:, 2] == 1.0)).all()


def test_tabular_jsma_targeted(model, sample_data, feature_metadata):
    x, y = sample_data
    attack = TabularJSMA(model)
    config = AttackConfig(
        params={
            "targeted": True,
            "target_label": 1,
            "max_features": 1,
            "step_size": 0.5,
            "feature_metadata": feature_metadata,
        }
    )
    x_adv = attack.generate(x, y, config)
    assert x_adv.shape == x.shape
    # At most 1 feature modified
    num_changed = (x_adv != x).sum(dim=1)
    assert (num_changed <= 1).all()


def test_tabular_jsma_invalid_config(model):
    attack = TabularJSMA(model)
    with pytest.raises(AttackConfigurationError):
        attack.validate_config(AttackConfig(params={"max_features": 0}))
    with pytest.raises(AttackConfigurationError):
        attack.validate_config(AttackConfig(params={"step_size": -1.0}))


# =========================================================================
# 5. VAEAttack Tests
# =========================================================================


def test_vae_attack_with_trained_vae(model, vae, sample_data, feature_metadata):
    x, y = sample_data
    attack = VAEAttack(model)
    config = AttackConfig(
        epsilon=0.5,
        params={
            "vae": vae,
            "num_steps": 5,
            "alpha": 0.1,
            "feature_metadata": feature_metadata,
        },
    )

    x_adv = attack.generate(x, y, config)
    assert x_adv.shape == x.shape
    # Immutable feature strictly preserved
    assert torch.allclose(x_adv[:, 4], x[:, 4])
    # Tabular constraints
    assert (x_adv[:, 0] >= 0.0).all() and (x_adv[:, 0] <= 10.0).all()
    int_diff = (x_adv[:, 1] - torch.round(x_adv[:, 1])).abs()
    assert (int_diff < 1e-4).all()


def test_vae_attack_missing_vae_raises_error(model, sample_data):
    x, y = sample_data
    attack = VAEAttack(model)
    # No VAE provided in params or model
    with pytest.raises(AttackConfigurationError):
        attack.generate(x, y, AttackConfig())


def test_vae_attack_invalid_config(model, vae):
    attack = VAEAttack(model)
    with pytest.raises(AttackConfigurationError):
        attack.validate_config(AttackConfig(params={"vae": vae, "epsilon": -0.5}))
    with pytest.raises(AttackConfigurationError):
        attack.validate_config(AttackConfig(params={"vae": vae, "step_size": -0.1}))


# =========================================================================
# 6. Framework Pipeline and Executor Integration Tests
# =========================================================================


def test_execute_attack_integration(model, sample_data, feature_metadata):
    x, y = sample_data
    res = execute_attack(
        model=model,
        attack_cls=CapPGD,
        inputs=x,
        labels=y,
        config=AttackConfig(
            epsilon=1.0,
            params={"num_steps": 3, "feature_metadata": feature_metadata},
        ),
    )
    assert res is not None
    assert res.attack_name == "cap_pgd"
    assert res.adversarial_examples is not None
    assert res.adversarial_examples.shape == x.shape
    assert "l0" in res.perturbation_metrics


def test_attack_engine_orchestration(model, sample_data, feature_metadata):
    engine = AttackEngine(model)
    x, y = sample_data

    res_pgd = engine.run_attack(
        "cap_pgd",
        x,
        y,
        AttackConfig(
            epsilon=1.0, params={"num_steps": 2, "feature_metadata": feature_metadata}
        ),
    )
    assert res_pgd.attack_name == "cap_pgd"

    res_lpf = engine.run_attack(
        "lowprofool",
        x,
        y,
        AttackConfig(params={"num_steps": 3, "feature_metadata": feature_metadata}),
    )
    assert res_lpf.attack_name == "lowprofool"

    res_cw = engine.run_attack(
        "carlini_wagner_tabular",
        x,
        y,
        AttackConfig(
            params={"max_iterations": 3, "feature_metadata": feature_metadata}
        ),
    )
    assert res_cw.attack_name == "carlini_wagner_tabular"

    res_jsma = engine.run_attack(
        "jsma_tabular",
        x,
        y,
        AttackConfig(params={"max_features": 1, "feature_metadata": feature_metadata}),
    )
    assert res_jsma.attack_name == "jsma_tabular"


# =========================================================================
# 7. Common Input Validation Tests
# =========================================================================


def test_tabular_input_validation_errors(model):
    attack = CapPGD(model)
    with pytest.raises(AttackExecutionError):
        attack.generate(None, torch.tensor([0]))
    with pytest.raises(AttackExecutionError):
        attack.generate(torch.empty(0), torch.tensor([0]))
    with pytest.raises(AttackExecutionError):
        # 3D tensor is invalid for tabular
        attack.generate(torch.randn(2, 3, 4), torch.tensor([0, 1]))
    with pytest.raises(AttackExecutionError):
        # Mismatched batch size
        attack.generate(torch.randn(2, 5), torch.tensor([0, 1, 0]))


def test_feature_specific_perturbation_limits(model, sample_data, feature_metadata):
    x, y = sample_data
    attack = CapPGD(model)
    # Give feature 0 a strict perturbation limit of 0.2
    config = AttackConfig(
        epsilon=5.0,
        params={
            "num_steps": 5,
            "alpha": 0.5,
            "feature_metadata": feature_metadata,
            "feature_limits": {0: 0.2},
        },
    )
    x_adv = attack.generate(x, y, config)
    f0_diff = (x_adv[:, 0] - x[:, 0]).abs()
    assert (f0_diff <= 0.2001).all()


def test_lowprofool_budget_and_stats(model, sample_data, feature_metadata):
    x, y = sample_data
    attack = LowProFool(model)
    budget = 1.0
    config = AttackConfig(
        params={
            "num_steps": 10,
            "alpha": 0.2,
            "budget": budget,
            "feature_metadata": feature_metadata,
            "feature_weights": [1.0, 1.0, 1.0, 1.0, 10.0],
        }
    )
    x_adv = attack.generate(x, y, config)
    assert x_adv.shape == x.shape
    stats = attack.get_perturbation_stats()
    assert isinstance(stats, dict)
    assert "weighted_distance" in stats
    assert "success_rate" in stats
    assert stats["weighted_distance"] <= budget + 1e-3


def test_batch_sizes(model, feature_metadata):
    attack = CapPGD(model)
    # Batch size 1
    x1 = torch.tensor([[5.0, 25.0, 1.0, 10.0, 100.0]])
    y1 = torch.tensor([0])
    adv1 = attack.generate(
        x1, y1, AttackConfig(epsilon=0.5, params={"feature_metadata": feature_metadata})
    )
    assert adv1.shape == (1, 5)

    # Batch size 8
    x8 = torch.randn(8, 5)
    y8 = torch.randint(0, 2, (8,))
    adv8 = attack.generate(x8, y8, AttackConfig(epsilon=0.5))
    assert adv8.shape == (8, 5)
