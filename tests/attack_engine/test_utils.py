"""
Unit tests for attack engine utilities (tensor, validation, perturbation).
"""

import pytest
import torch
import torch.nn as nn

from app.attack_engine.utils.tensor_utils import (
    get_device,
    to_tensor,
    clip_tensor,
    project_lp,
)
from app.attack_engine.utils.validation import (
    validate_inputs,
    validate_bounds,
    validate_attack_config,
    check_model_compatibility,
)
from app.attack_engine.utils.perturbation import (
    compute_perturbation,
    compute_l0_norm,
    compute_l2_norm,
    compute_linf_norm,
    compute_perturbation_metrics,
)
from app.attack_engine.config import AttackConfig
from app.attack_engine.models import AttackMetadata
from app.attack_engine.exceptions import (
    AttackConfigurationError,
    AttackExecutionError,
    UnsupportedModelError,
)


def test_tensor_utils():
    # to_tensor
    t = to_tensor([1, 2, 3], dtype=torch.float32)
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.float32

    # clip_tensor
    x = torch.tensor([-1.0, 0.5, 2.0])
    clipped = clip_tensor(x, clip_min=0.0, clip_max=1.0)
    assert (clipped >= 0.0).all() and (clipped <= 1.0).all()

    # project_lp Linf
    x_orig = torch.zeros(2, 4)
    x_adv = torch.ones(2, 4) * 0.5
    projected = project_lp(x_adv, x_orig, epsilon=0.2, norm="Linf")
    diff = (projected - x_orig).abs()
    assert (diff <= 0.20001).all()

    # project_lp L2
    projected_l2 = project_lp(x_adv, x_orig, epsilon=0.3, norm="L2")
    l2 = torch.norm((projected_l2 - x_orig).view(2, -1), p=2, dim=1)
    assert (l2 <= 0.30001).all()


def test_validation_utils():
    # validate_inputs matching
    inputs = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 0, 1])
    validate_inputs(inputs, labels)

    # validate_inputs mismatched batch
    with pytest.raises(AttackExecutionError):
        validate_inputs(inputs, torch.tensor([0, 1]))

    # validate_bounds
    with pytest.raises(AttackConfigurationError):
        validate_bounds(clip_min=1.0, clip_max=0.0)

    # validate_attack_config negative epsilon
    with pytest.raises(AttackConfigurationError):
        validate_attack_config(AttackConfig(epsilon=-0.1))

    # check_model_compatibility
    class NonPyTorchModel:
        pass

    meta = AttackMetadata(name="test", requires_gradient=True)
    with pytest.raises(UnsupportedModelError):
        check_model_compatibility(NonPyTorchModel(), meta)


def test_perturbation_metrics():
    x_orig = torch.zeros(2, 4)
    x_adv = torch.tensor([[0.1, 0.2, 0.0, 0.0], [0.3, 0.0, 0.0, 0.4]])

    metrics = compute_perturbation_metrics(x_orig, x_adv)
    assert "l0" in metrics
    assert "l2" in metrics
    assert "linf" in metrics
    assert "mean_abs_error" in metrics
    assert metrics["linf"] > 0
    assert metrics["l2"] > 0
    assert metrics["l0"] > 0
