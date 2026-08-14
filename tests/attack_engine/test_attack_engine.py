"""
Unit tests for AttackEngine orchestrator and package entrypoint.
"""

import pytest
import torch
import torch.nn as nn

from app.attack_engine import AttackEngine, run_attack_pipeline, AttackConfig, FGSM
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

    adv_inputs = engine.run_attack("fgsm", inputs, labels)
    assert isinstance(adv_inputs, torch.Tensor)
    assert adv_inputs.shape == inputs.shape


def test_attack_engine_run_pipeline():
    model = DummyNet()
    engine = AttackEngine(model)
    inputs = torch.randn(3, 4)
    labels = torch.tensor([0, 1, 0])
    configs = {"fgsm": AttackConfig(epsilon=0.2)}

    results = engine.run_pipeline(["fgsm"], inputs, labels, configs=configs)
    assert "fgsm" in results
    assert isinstance(results["fgsm"], torch.Tensor)


def test_run_attack_pipeline_helper():
    model = DummyNet()
    inputs = torch.randn(2, 4)
    labels = torch.tensor([1, 0])

    results = run_attack_pipeline(model, ["fgsm"], inputs, labels)
    assert "fgsm" in results


def test_package_exports():
    assert hasattr(attack_module, "AttackEngine")
    assert hasattr(attack_module, "FGSM")
    assert hasattr(attack_module, "AttackConfig")
    assert hasattr(attack_module, "execute_attack")
    assert hasattr(attack_module, "discover_attacks")
    assert hasattr(attack_module, "run_attack_pipeline")
