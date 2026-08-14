"""
Unit tests for AttackEngine orchestrator and package entrypoint.
"""

import pytest
import torch
import torch.nn as nn

from app.attack_engine import (
    AttackEngine,
    run_attack_pipeline,
    AttackConfig,
    FGSM,
    AttackResult,
    AttackResults,
)
import app.attack_engine as attack_module


class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        nn.init.constant_(self.fc.weight[0], 0.1)
        nn.init.constant_(self.fc.weight[1], -0.1)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        return self.fc(x)


def test_attack_engine_run_single():
    model = DummyNet()
    engine = AttackEngine(model)
    inputs = torch.randn(3, 4)
    labels = torch.tensor([0, 1, 0])

    result = engine.run_attack("fgsm", inputs, labels)
    assert isinstance(result, AttackResult)
    assert isinstance(result.adversarial_examples, torch.Tensor)
    assert result.adv_inputs.shape == inputs.shape


def test_attack_engine_run_pipeline():
    model = DummyNet()
    engine = AttackEngine(model)
    inputs = torch.randn(3, 4)
    labels = torch.tensor([0, 1, 0])
    configs = {"fgsm": AttackConfig(epsilon=0.2)}

    results = engine.run_pipeline(["fgsm", "pgd", "deepfool"], inputs, labels, configs=configs)
    assert isinstance(results, AttackResults)
    assert "fgsm" in results
    assert "pgd" in results
    assert "deepfool" in results
    assert isinstance(results["fgsm"], AttackResult)
    assert isinstance(results["pgd"], AttackResult)
    assert isinstance(results["deepfool"], AttackResult)


def test_run_attack_pipeline_helper():
    model = DummyNet()
    inputs = torch.randn(2, 4)
    labels = torch.tensor([1, 0])

    results = run_attack_pipeline(model, ["fgsm", "deepfool"], inputs, labels)
    assert isinstance(results, AttackResults)
    assert "fgsm" in results
    assert "deepfool" in results


def test_package_exports():
    assert hasattr(attack_module, "AttackEngine")
    assert hasattr(attack_module, "FGSM")
    assert hasattr(attack_module, "PGD")
    assert hasattr(attack_module, "DeepFool")
    assert hasattr(attack_module, "AttackConfig")
    assert hasattr(attack_module, "AttackResult")
    assert hasattr(attack_module, "AttackResults")
    assert hasattr(attack_module, "execute_attack")
    assert hasattr(attack_module, "discover_attacks")
    assert hasattr(attack_module, "run_attack_pipeline")
