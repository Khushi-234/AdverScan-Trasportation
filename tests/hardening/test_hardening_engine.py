"""
Integration tests for HardeningEngine coordinator.
"""

import pytest
import torch
import torch.nn as nn

from app.hardening.hardening_engine import HardeningEngine
from app.hardening.hardening_result import HardeningResult
from app.hardening.exceptions import HardeningConfigurationError, DefenseNotFoundError


class DummyModel(nn.Module):

    def __init__(self, features: int = 8, classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(features, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        return self.fc(x)


def mock_eval_fn(model: nn.Module) -> dict:
    """Mock evaluation function returning dummy accuracy metric."""
    return {"accuracy": 0.85}


def test_hardening_engine_auto_selection():
    engine = HardeningEngine()
    model = DummyModel(features=8, classes=2)
    inputs = torch.rand(4, 8)
    labels = torch.tensor([0, 1, 0, 1])

    result = engine.harden(
        model=model,
        defense="auto",
        inputs=inputs,
        labels=labels,
        attack_name="pgd",
        risk_level="HIGH",
        vulnerability_score=80.0,
        eval_fn=mock_eval_fn,
    )

    assert isinstance(result, HardeningResult)
    assert result.success is True
    assert result.metrics_before.get("accuracy") == 0.85
    assert result.metrics_after.get("accuracy") == 0.85
    assert len(result.recommendations) > 0


def test_hardening_engine_explicit_defense():
    engine = HardeningEngine()
    model = DummyModel(features=8, classes=2)
    inputs = torch.rand(4, 8)

    result = engine.harden(
        model=model,
        defense="spatial_smoothing",
        inputs=inputs,
        defense_config={"kernel_size": 3, "sigma": 1.0},
    )

    assert result.success is True
    assert result.metadata.defense_name == "spatial_smoothing"


def test_hardening_engine_invalid_defense_raises():
    engine = HardeningEngine()
    model = DummyModel()

    with pytest.raises(DefenseNotFoundError):
        engine.harden(model=model, defense="non_existent_defense")


def test_hardening_engine_invalid_model_raises():
    engine = HardeningEngine()
    with pytest.raises(HardeningConfigurationError):
        engine.harden(model="not_a_model")
