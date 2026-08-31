"""
Unit and integration tests for Adversarial Detection and Confidence Rejection defenses.
"""

import pytest
import torch
import torch.nn as nn

from app.hardening.defenses.adversarial_detection import AdversarialDetectionDefense
from app.hardening.defenses.confidence_rejection import ConfidenceRejectionDefense
from app.hardening.hardening_engine import HardeningEngine
from app.hardening.hardening_result import HardeningResult
from app.hardening.defenses import get_defense_class


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_confidence_rejection_defense_execution():
    model = SimpleModel()
    defense = ConfidenceRejectionDefense(threshold=0.8)

    inputs = torch.randn(10, 4)
    result = defense.apply(model=model, inputs=inputs)

    assert isinstance(result, HardeningResult)
    assert result.success is True
    assert result.metadata.defense_name == "confidence_rejection"
    assert "total_samples" in result.metadata.extra_metadata
    assert result.metadata.extra_metadata["total_samples"] == 10
    assert "rejection_rate" in result.metadata.extra_metadata


def test_adversarial_detection_defense_execution():
    model = SimpleModel()
    detector = AdversarialDetectionDefense(threshold=0.5, noise_std=0.05, num_samples=3, method="sensitivity")

    inputs = torch.randn(8, 4)
    result = detector.apply(model=model, inputs=inputs)

    assert isinstance(result, HardeningResult)
    assert result.success is True
    assert result.metadata.defense_name == "adversarial_detection"
    assert result.metadata.extra_metadata["total_samples"] == 8
    assert "adversarial_detected_count" in result.metadata.extra_metadata
    assert "detection_mask" in result.metadata.extra_metadata
    assert "detection_scores" in result.metadata.extra_metadata
    assert len(result.metadata.extra_metadata["detection_scores"]) == 8


def test_adversarial_detection_methods():
    model = SimpleModel()
    inputs = torch.randn(6, 4)

    for method in ["sensitivity", "margin", "entropy"]:
        detector = AdversarialDetectionDefense(threshold=0.4, method=method)
        is_adv, scores = detector.detect(model, inputs)
        assert is_adv.shape[0] == 6
        assert scores.shape[0] == 6
        assert (scores >= 0.0).all() and (scores <= 1.5).all()


def test_detector_registry_lookup():
    det_cls = get_defense_class("adversarial_detection")
    assert det_cls == AdversarialDetectionDefense

    conf_cls = get_defense_class("confidence_rejection")
    assert conf_cls == ConfidenceRejectionDefense


def test_hardening_engine_with_adversarial_detection():
    engine = HardeningEngine()
    model = SimpleModel()
    inputs = torch.randn(5, 4)

    result = engine.harden(
        model=model,
        defense="adversarial_detection",
        inputs=inputs,
        defense_config={"threshold": 0.4, "method": "sensitivity"},
    )

    assert result.success is True
    assert result.metadata.defense_name == "adversarial_detection"
    assert "adversarial_detected_count" in result.metadata.extra_metadata
