"""
Unit and integration tests for Iterative Defense Selection with Configurable Acceptance Thresholds.
"""

import pytest
import torch
import torch.nn as nn

from app.hardening.hardening_engine import HardeningEngine
from app.hardening.hardening_result import DefenseAttemptResult, HardeningResult
from app.retest.threshold import RetestThresholds, Thresholds

# Alias for testing
HardeningThresholds = RetestThresholds


class SimpleDummyModel(nn.Module):
    def __init__(self, features: int = 4, classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(features, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        return self.fc(x)


def test_thresholds_evaluation_accepted():
    thresholds = HardeningThresholds(
        min_vuln_score_improvement=5.0,
        min_asr_reduction=0.10,
        max_clean_accuracy_drop=0.02,
        max_latency_overhead_ms=30.0,
    )

    before = {"vulnerability_score": 80.0, "asr": 0.85, "clean_accuracy": 0.95, "latency_ms": 10.0}
    after = {"vulnerability_score": 72.0, "asr": 0.70, "clean_accuracy": 0.94, "latency_ms": 25.0}

    eval_res = thresholds.evaluate(before_metrics=before, after_metrics=after, latency_sensitive=True)

    assert eval_res["status"] == "accepted"
    assert eval_res["is_improved"] is True
    assert eval_res["improvement_sufficient"] is True
    assert eval_res["vulnerability_score_improvement"] == 8.0
    assert eval_res["asr_reduction"] == 0.15
    assert eval_res["clean_accuracy_drop"] == 0.01
    assert eval_res["clean_accuracy_drop_pct_points"] == 1.0
    assert eval_res["latency_overhead_ms"] == 15.0


def test_thresholds_evaluation_insufficient():
    thresholds = HardeningThresholds(
        min_vuln_score_improvement=5.0,
        min_asr_reduction=0.10,
        max_clean_accuracy_drop=0.02,
    )

    # Improved, but only 2 points improvement (< 5 points threshold)
    before = {"vulnerability_score": 80.0, "asr": 0.85, "clean_accuracy": 0.95}
    after = {"vulnerability_score": 78.0, "asr": 0.82, "clean_accuracy": 0.95}

    eval_res = thresholds.evaluate(before_metrics=before, after_metrics=after)

    assert eval_res["status"] == "insufficient"
    assert eval_res["is_improved"] is True
    assert eval_res["improvement_sufficient"] is False
    assert eval_res["vulnerability_score_improvement"] == 2.0


def test_thresholds_evaluation_harmful():
    thresholds = HardeningThresholds(
        severe_clean_drop_threshold=0.10,
    )

    # Clean accuracy dropped from 95% to 80% (15 percentage points drop > 10% severe threshold)
    before = {"vulnerability_score": 80.0, "clean_accuracy": 0.95}
    after = {"vulnerability_score": 60.0, "clean_accuracy": 0.80}

    eval_res = thresholds.evaluate(before_metrics=before, after_metrics=after)

    assert eval_res["status"] == "harmful"
    assert eval_res["is_improved"] is True
    assert eval_res["improvement_sufficient"] is False


def test_thresholds_latency_sensitive_constraint():
    thresholds = HardeningThresholds(
        max_latency_overhead_ms=30.0,
        respect_latency_sensitive=True,
    )

    before = {"vulnerability_score": 80.0, "clean_accuracy": 0.95, "latency_ms": 10.0}
    after = {"vulnerability_score": 60.0, "clean_accuracy": 0.95, "latency_ms": 55.0}  # +45ms overhead

    # When latency_sensitive is True -> rejected due to latency overhead
    eval_sensitive = thresholds.evaluate(before_metrics=before, after_metrics=after, latency_sensitive=True)
    assert eval_sensitive["improvement_sufficient"] is False
    assert "Latency overhead" in eval_sensitive["reason"]

    # When latency_sensitive is False -> accepted
    eval_nonsensitive = thresholds.evaluate(before_metrics=before, after_metrics=after, latency_sensitive=False)
    assert eval_nonsensitive["improvement_sufficient"] is True


def test_iterative_hardening_first_candidate_accepted():
    """Rule 5: First defense satisfies acceptance criteria -> accept and stop immediately."""
    engine = HardeningEngine()
    model = SimpleDummyModel()

    def mock_eval_fn(tested_model):
        return {
            "vulnerability_score": 65.0,  # 80.0 -> 65.0 (15 pts improvement >= 5.0)
            "clean_accuracy": 0.94,       # 0.95 -> 0.94 (1% drop <= 2%)
            "asr": 0.60,                  # 0.80 -> 0.60 (20% reduction >= 10%)
            "latency_ms": 12.0,
        }

    baseline = {"vulnerability_score": 80.0, "clean_accuracy": 0.95, "asr": 0.80, "latency_ms": 10.0}

    result = engine.harden_iterative(
        model=model,
        attack_name="fgsm",
        risk_level="MEDIUM",
        vulnerability_score=80.0,
        candidate_defenses=["spatial_smoothing", "bit_depth_reduction", "jpeg_compression"],
        eval_fn=mock_eval_fn,
        baseline_metrics=baseline,
    )

    assert result.success is True
    assert result.status == "accepted"
    assert result.is_improved is True
    assert result.improvement_sufficient is True
    assert result.selected_defense == "spatial_smoothing"
    assert result.num_attempts == 1
    assert len(result.defense_attempts) == 1

    attempt0 = result.defense_attempts[0]
    assert attempt0.defense_name == "spatial_smoothing"
    assert attempt0.status == "accepted"
    assert attempt0.vuln_score_improvement == 15.0
    assert attempt0.asr_reduction == 0.20
    assert attempt0.clean_accuracy_drop_pct_points == 1.0


def test_iterative_hardening_fallback_to_second_candidate():
    """Rule 6, 7: Candidate 1 insufficient -> tries candidate 2 -> candidate 2 accepted."""
    engine = HardeningEngine()
    model = SimpleDummyModel()

    attempt_counter = 0

    def mock_eval_fn(tested_model):
        nonlocal attempt_counter
        attempt_counter += 1
        if attempt_counter == 1:
            # First candidate (spatial_smoothing): insufficient improvement (only 2 points vuln drop)
            return {
                "vulnerability_score": 78.0,
                "clean_accuracy": 0.95,
                "asr": 0.78,
                "latency_ms": 12.0,
            }
        else:
            # Second candidate (bit_depth_reduction): sufficient improvement (12 points drop, clean drop 1%)
            return {
                "vulnerability_score": 68.0,
                "clean_accuracy": 0.94,
                "asr": 0.65,
                "latency_ms": 14.0,
            }

    baseline = {"vulnerability_score": 80.0, "clean_accuracy": 0.95, "asr": 0.80, "latency_ms": 10.0}

    result = engine.harden_iterative(
        model=model,
        candidate_defenses=["spatial_smoothing", "bit_depth_reduction", "jpeg_compression"],
        eval_fn=mock_eval_fn,
        baseline_metrics=baseline,
    )

    assert result.success is True
    assert result.status == "accepted"
    assert result.selected_defense == "bit_depth_reduction"
    assert result.num_attempts == 2
    assert len(result.defense_attempts) == 2

    assert result.defense_attempts[0].defense_name == "spatial_smoothing"
    assert result.defense_attempts[0].status == "insufficient"
    assert result.defense_attempts[0].is_improved is True
    assert result.defense_attempts[0].improvement_sufficient is False

    assert result.defense_attempts[1].defense_name == "bit_depth_reduction"
    assert result.defense_attempts[1].status == "accepted"
    assert result.defense_attempts[1].improvement_sufficient is True


def test_iterative_hardening_no_candidate_satisfies():
    """Rule 9: If no candidate satisfies criteria, report that no tested defense provided sufficient improvement."""
    engine = HardeningEngine()
    model = SimpleDummyModel()

    # All candidates give insufficient or harmful results
    def mock_eval_fn(tested_model):
        return {
            "vulnerability_score": 79.0,  # only 1 pt improvement
            "clean_accuracy": 0.95,
            "asr": 0.79,                  # only 1% reduction
            "latency_ms": 12.0,
        }

    baseline = {"vulnerability_score": 80.0, "clean_accuracy": 0.95, "asr": 0.80, "latency_ms": 10.0}

    result = engine.harden_iterative(
        model=model,
        candidate_defenses=["spatial_smoothing", "bit_depth_reduction"],
        eval_fn=mock_eval_fn,
        baseline_metrics=baseline,
    )

    assert result.success is True
    assert result.selected_defense is None
    assert result.improvement_sufficient is False
    assert result.is_improved is True  # Measurable slight improvement, but insufficient
    assert result.status == "insufficient"
    assert result.num_attempts == 2
    assert len(result.defense_attempts) == 2
    assert "No tested defense" in result.recommendations[0]


def test_iterative_hardening_non_stacking():
    """Rule 8: Each candidate evaluated independently against baseline, not stacked."""
    engine = HardeningEngine()
    model = SimpleDummyModel()

    models_received = []

    def mock_eval_fn(tested_model):
        models_received.append(tested_model)
        return {
            "vulnerability_score": 78.0,
            "clean_accuracy": 0.95,
            "asr": 0.78,
        }

    engine.harden_iterative(
        model=model,
        candidate_defenses=["spatial_smoothing", "bit_depth_reduction"],
        eval_fn=mock_eval_fn,
        baseline_metrics={"vulnerability_score": 80.0, "clean_accuracy": 0.95, "asr": 0.80},
    )

    # 2 attempts were made
    assert len(models_received) == 2
    # Second tested model must wrap original model, not the first wrapped model
    assert models_received[0] != models_received[1]


def test_harden_method_delegates_to_iterative():
    engine = HardeningEngine()
    model = SimpleDummyModel()

    def mock_eval_fn(m):
        return {"vulnerability_score": 60.0, "clean_accuracy": 0.95, "asr": 0.50}

    result = engine.harden(
        model=model,
        defense="iterative",
        candidate_defenses=["spatial_smoothing"],
        eval_fn=mock_eval_fn,
        vulnerability_score=80.0,
        baseline_metrics={"vulnerability_score": 80.0, "clean_accuracy": 0.95, "asr": 0.80},
    )

    assert result.success is True
    assert result.selected_defense == "spatial_smoothing"
    assert result.improvement_sufficient is True


def test_retest_threshold_consistency():
    r_thresh = RetestThresholds(min_vuln_score_improvement=5.0, min_asr_reduction=0.10)
    before = {"vulnerability_score": 75.0, "asr": 0.80, "clean_accuracy": 0.90}
    after = {"vulnerability_score": 65.0, "asr": 0.60, "clean_accuracy": 0.89}

    res = r_thresh.evaluate(before, after)
    assert res["status"] == "accepted"
    assert res["improvement_sufficient"] is True
    assert res["vulnerability_score_improvement"] == 10.0
