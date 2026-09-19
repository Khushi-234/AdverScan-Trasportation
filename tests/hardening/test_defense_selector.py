"""
Comprehensive unit tests for modular Context-Aware DefenseSelector in Module 7.
"""

import pytest
from app.hardening.defense_selector import DefenseSelector
from app.hardening.hardening_context import HardeningContext
from app.hardening.exceptions import DefenseNotFoundError
from app.hardening.defenses import (
    AdversarialTrainingDefense,
    SpatialSmoothingDefense,
    RandomizedSmoothingDefense,
    ConfidenceRejectionDefense,
)


def test_selector_pgd_retraining_disabled_rejection_reason():
    """Test explicit rejection reason when retraining is disabled."""
    selector = DefenseSelector()
    rec = selector.recommend(
        attack_name="pgd",
        risk_level="HIGH",
        allow_retraining=False,
    )

    assert "adversarial_training" in rec["rejected_defenses"]
    reason = rec["rejected_defenses"]["adversarial_training"]
    assert reason == "Retraining is disabled by operational constraints"


def test_selector_tabular_domain_rejection_reason():
    """Test metadata-driven domain filtering rejection reason."""
    selector = DefenseSelector()
    rec = selector.recommend(
        attack_name="fgsm",
        input_domain="tabular",
    )

    assert "spatial_smoothing" in rec["rejected_defenses"]
    reason = rec["rejected_defenses"]["spatial_smoothing"]
    assert "Input domain 'tabular' not supported" in reason
    assert "jpeg_compression" in rec["rejected_defenses"]


def test_selector_high_latency_sensitivity():
    """Test high latency sensitivity penalization and rejection."""
    selector = DefenseSelector()
    rec = selector.recommend(
        attack_name="pgd",
        risk_level="CRITICAL",
        latency_sensitive=True,
    )

    # Randomized smoothing has high latency cost (40 ms), should be rejected under strict latency constraints
    assert "randomized_smoothing" in rec["rejected_defenses"]
    assert "High inference latency overhead" in rec["rejected_defenses"]["randomized_smoothing"]
    assert rec["primary_defense"] not in ("randomized_smoothing", None)


def test_selector_no_compatible_defense():
    """Test context where no candidate defense is compatible and no default is set."""
    selector = DefenseSelector(default_defense=None)
    # Retraining disabled, non-image domain, strict latency sensitivity
    rec = selector.recommend(
        input_domain="custom_text_domain",
        allow_retraining=False,
        latency_sensitive=True,
    )

    # All defenses should be rejected
    assert rec["primary_defense"] is None
    assert rec["secondary_defenses"] == []
    assert rec["candidate_scores"] == {}
    assert len(rec["rejected_defenses"]) > 0

    with pytest.raises(DefenseNotFoundError):
        selector.select(
            input_domain="custom_text_domain",
            allow_retraining=False,
            latency_sensitive=True,
        )


def test_selector_domain_agnostic_defenses():
    """Test that domain-agnostic defenses with supported_domains=['*'] work on any domain."""
    selector = DefenseSelector(default_defense=None)
    rec = selector.recommend(
        input_domain="nlp",
        allow_retraining=False,
        latency_sensitive=False,
    )

    # confidence_rejection and adversarial_detection support '*' domains
    assert rec["primary_defense"] in ("confidence_rejection", "adversarial_detection")
    assert "confidence_rejection" not in rec["rejected_defenses"]
    assert "adversarial_detection" not in rec["rejected_defenses"]


def test_hardening_context_creation():
    """Test creation and normalization of HardeningContext dataclass."""
    ctx = HardeningContext(
        attack_name=" PGD ",
        risk_level="high",
        input_domain="IMAGE",
        has_training_data=True,
        has_labels=True,
        latency_sensitive=False,
    )
    assert ctx.attack_name == "pgd"
    assert ctx.is_iterative is True
    assert ctx.risk_level == "HIGH"
    assert ctx.input_domain == "image"
    assert ctx.norm == "Linf"
    assert isinstance(ctx.resource_limits, dict)


def test_selector_missing_data_or_labels_rejection():
    """Test rejection reasons when training data or labels are missing."""
    selector = DefenseSelector()

    # Missing training data
    rec_no_data = selector.recommend(
        allow_retraining=True,
        has_training_data=False,
        has_labels=True,
    )
    assert "adversarial_training" in rec_no_data["rejected_defenses"]
    assert "Training dataset is unavailable" in rec_no_data["rejected_defenses"]["adversarial_training"]

    # Missing labels
    rec_no_labels = selector.recommend(
        allow_retraining=True,
        has_training_data=True,
        has_labels=False,
    )
    assert "adversarial_training" in rec_no_labels["rejected_defenses"]
    assert "Ground truth labels are unavailable" in rec_no_labels["rejected_defenses"]["adversarial_training"]


def test_selector_backward_compatible_select_instantiation():
    """Test that select() returns an instantiated BaseDefense object backward-compatibly."""
    selector = DefenseSelector()
    defense_inst = selector.select(
        attack_name="fgsm",
        input_domain="image",
        risk_level="MEDIUM",
    )
    assert defense_inst is not None
    assert hasattr(defense_inst, "apply")
    assert hasattr(defense_inst, "name")


def test_selector_weighted_scoring_and_ranking():
    """Test weighted candidate scoring and ranking order in output recommendation."""
    selector = DefenseSelector()
    rec = selector.recommend(
        attack_name="pgd",
        risk_level="CRITICAL",
        allow_retraining=True,
        has_training_data=True,
        has_labels=True,
        vulnerability_score=85.0,
    )

    scores = rec["candidate_scores"]
    primary = rec["primary_defense"]
    secondaries = rec["secondary_defenses"]

    assert primary is not None
    assert len(scores) > 0
    # Verify primary defense has top score
    for sec in secondaries:
        assert scores[primary] >= scores[sec]

    assert rec["suggested_params"] != {}
    assert isinstance(rec["rationale"], str)


def test_candidate_generation_count_and_params():
    """Test that candidate selection generates 3 to 5 defenses and provides all_candidate_params."""
    selector = DefenseSelector()
    rec = selector.recommend(
        attack_name="pgd",
        risk_level="HIGH",
        vulnerability_score=75.0,
    )

    candidates = rec["candidate_defenses"]
    assert 3 <= len(candidates) <= 5
    assert rec["primary_defense"] in candidates
    assert len(rec["all_candidate_params"]) == len(candidates)
    for cand in candidates:
        assert cand in rec["all_candidate_params"]
    assert "selection_basis" in rec
    assert rec["selection_basis"]["attack"] == "pgd"


def test_rule_ordering_attack_priority_over_epsilon():
    """
    Test Problem 1 fix: attack-specific conditions take priority over generic epsilon intervals.
    FGSM with epsilon 0.03 should produce preprocessing candidates, not smoothing or DeepFool rules.
    """
    selector = DefenseSelector()
    rec = selector.recommend(
        attack_name="fgsm",
        epsilon=0.03,
        risk_level="MEDIUM",
    )

    candidates = rec["candidate_defenses"]
    # Single-step FGSM should prioritize spatial smoothing / feature squeezing / filters
    assert any(d in candidates for d in ("spatial_smoothing", "feature_squeezing", "gaussian_filter"))
    assert rec["primary_defense"] in ("spatial_smoothing", "feature_squeezing", "gaussian_filter")


def test_adversarial_training_never_automatically_applied():
    """
    Test Problem 2 fix: adversarial training is never automatically selected or applied as primary defense,
    even under CRITICAL risk and high vulnerability with retraining allowed.
    """
    selector = DefenseSelector()
    rec = selector.recommend(
        attack_name="pgd",
        risk_level="CRITICAL",
        vulnerability_score=95.0,
        allow_retraining=True,
        has_training_data=True,
        has_labels=True,
    )

    assert rec["primary_defense"] != "adversarial_training"
    assert "adversarial_training" not in rec["candidate_defenses"]

    # select() should return a post-training defense instance, never AdversarialTrainingDefense
    defense_instance = selector.select(
        attack_name="pgd",
        risk_level="CRITICAL",
        vulnerability_score=95.0,
        allow_retraining=True,
        has_training_data=True,
        has_labels=True,
    )
    assert not isinstance(defense_instance, AdversarialTrainingDefense)


def test_latency_sensitive_penalty_not_blind_spatial_override():
    """
    Test Problem 3 fix: latency sensitivity penalizes high-latency defenses without blindly forcing spatial_smoothing.
    Under PGD + HIGH risk + latency sensitivity, robust low-latency defenses like feature_denoising or confidence_rejection
    are prioritized over low-robustness spatial smoothing.
    """
    selector = DefenseSelector()
    rec = selector.recommend(
        attack_name="pgd",
        risk_level="HIGH",
        vulnerability_score=80.0,
        latency_sensitive=True,
    )

    candidates = rec["candidate_defenses"]
    assert len(candidates) >= 3
    # High-latency randomized smoothing (40ms) should be rejected under strict latency
    assert "randomized_smoothing" not in candidates
    # Candidates should include lightweight yet robust defenses for iterative attacks
    assert any(d in candidates for d in ("feature_denoising", "confidence_rejection", "feature_squeezing"))


def test_empirical_evaluation_clean_accuracy_percentage_points_and_zero_baseline():
    """
    Test Stage 2: clean accuracy drop computed in explicit percentage points,
    penalizing excessive drops, with zero-baseline protection.
    """
    selector = DefenseSelector()

    baseline = {"clean_accuracy": 0.92, "attack_success_rate": 0.80, "latency_ms": 2.0}
    candidates_metrics = {
        "candidate_mild_drop": {
            "clean_accuracy": 0.90,  # drop = 0.02 (2 pts, <= 5 pts threshold)
            "attack_success_rate": 0.25,
            "latency_ms": 6.0,
        },
        "candidate_severe_drop": {
            "clean_accuracy": 0.70,  # drop = 0.22 (22 pts, > 10 pts threshold)
            "attack_success_rate": 0.15,
            "latency_ms": 6.0,
        },
    }

    evals = selector.evaluate_candidate_results(candidates_metrics, baseline_metrics=baseline)

    mild = evals["candidate_mild_drop"]
    severe = evals["candidate_severe_drop"]

    assert mild["clean_accuracy_drop"] == 0.02
    assert mild["clean_accuracy_drop_pct_points"] == 2.0
    assert severe["clean_accuracy_drop"] == 0.22
    assert severe["clean_accuracy_drop_pct_points"] == 22.0
    assert mild["clean_preservation_score"] > severe["clean_preservation_score"]
    # Severe drop incurs heavy penalty, leading to a lower composite score despite lower ASR
    assert mild["composite_score"] > severe["composite_score"]

    # Test protection against zero baseline clean accuracy
    zero_baseline = {"clean_accuracy": 0.0, "attack_success_rate": 0.50, "latency_ms": 1.0}
    zero_evals = selector.evaluate_candidate_results(
        {"cand": {"clean_accuracy": 0.0, "attack_success_rate": 0.20, "latency_ms": 3.0}},
        baseline_metrics=zero_baseline,
    )
    assert zero_evals["cand"]["clean_accuracy_drop"] == 0.0
    assert zero_evals["cand"]["clean_accuracy_drop_pct_points"] == 0.0


def test_empirical_evaluation_latency_overhead():
    """Test Stage 2: concrete latency overhead calculation and second-to-ms unit conversion."""
    selector = DefenseSelector()

    baseline = {"clean_accuracy": 0.90, "attack_success_rate": 0.70, "latency_ms": 3.0}
    candidates_metrics = {
        "fast_defense": {"clean_accuracy": 0.90, "attack_success_rate": 0.30, "latency_ms": 7.5},
        "slow_defense": {"clean_accuracy": 0.90, "attack_success_rate": 0.30, "latency_ms": 48.0},
        "seconds_defense": {"clean_accuracy": 0.90, "attack_success_rate": 0.30, "execution_time_seconds": 0.015},  # 15.0 ms
    }

    evals = selector.evaluate_candidate_results(candidates_metrics, baseline_metrics=baseline)
    assert evals["fast_defense"]["latency_overhead_ms"] == 4.5
    assert evals["slow_defense"]["latency_overhead_ms"] == 45.0
    assert evals["seconds_defense"]["defended_latency_ms"] == 15.0
    assert evals["seconds_defense"]["latency_overhead_ms"] == 12.0
    assert evals["fast_defense"]["latency_efficiency_score"] > evals["slow_defense"]["latency_efficiency_score"]


def test_empirical_evaluation_missing_asr_handling():
    """
    Test edge case: when baseline ASR is None, robustness is marked unavailable,
    composite score is dynamically computed from remaining available metrics,
    and selection_status explicitly reports partial evaluation.
    """
    selector = DefenseSelector()
    baseline = {"clean_accuracy": 0.95, "attack_success_rate": None, "latency_ms": 2.0}
    candidate = {"cand": {"clean_accuracy": 0.94, "attack_success_rate": 0.20, "latency_ms": 5.0}}

    evals = selector.evaluate_candidate_results(candidate, baseline_metrics=baseline)
    res = evals["cand"]
    assert res["robustness_available"] is False
    assert res["robustness_improvement"] is None
    assert res["is_effective"] is None
    assert res["status"] == "partial_evaluation_no_asr"
    assert res["composite_score"] > 0.0

    best = selector.select_best_evaluated(evals)
    assert best["selection_status"] == "partial_evaluation_no_asr"
    assert "baseline ASR was unavailable" in best["selection_rationale"]
    assert "robustness unverified" in best["selection_rationale"]


def test_empirical_evaluation_zero_metric_extraction_safety():
    """Test that metric extraction safely handles 0.0 values without falsy boolean 'or' bugs."""
    selector = DefenseSelector()
    # When ASR is 0.0 (perfect defense or attack failed)
    baseline = {"clean_accuracy": 0.90, "attack_success_rate": 0.60, "latency_ms": 0.0}
    candidate = {"zero_asr_defense": {"clean_accuracy": 0.90, "attack_success_rate": 0.0, "latency_ms": 0.0}}

    evals = selector.evaluate_candidate_results(candidate, baseline_metrics=baseline)
    res = evals["zero_asr_defense"]
    assert res["defended_asr"] == 0.0
    assert res["robustness_improvement"] == 0.60
    assert res["is_effective"] is True
    assert res["latency_overhead_ms"] == 0.0
    assert res["latency_efficiency_score"] == 1.0


def test_empirical_evaluation_negative_robustness_improvement():
    """
    Test edge case: defense makes ASR worse (negative robustness improvement),
    marked ineffective, status is ineffective_defense, and penalized.
    """
    selector = DefenseSelector()
    baseline = {"clean_accuracy": 0.90, "attack_success_rate": 0.40, "latency_ms": 2.0}
    candidate = {"worse_defense": {"clean_accuracy": 0.90, "attack_success_rate": 0.50, "latency_ms": 4.0}}

    evals = selector.evaluate_candidate_results(candidate, baseline_metrics=baseline)
    res = evals["worse_defense"]
    assert res["robustness_improvement"] == -0.10
    assert res["is_effective"] is False
    assert res["status"] == "ineffective_defense"
    assert res["composite_score"] < 0.40

    best = selector.select_best_evaluated(evals)
    assert best["selection_status"] == "ineffective_defense"
    assert "Warning: Defense failed to improve robustness" in best["selection_rationale"]


def test_select_best_evaluated_winner_and_supported_rationale():
    """Test final empirical selection and fully data-supported rationale output."""
    selector = DefenseSelector()
    baseline = {"clean_accuracy": 0.94, "attack_success_rate": 0.85, "latency_ms": 2.0}
    candidates = {
        "defense_a": {"clean_accuracy": 0.65, "attack_success_rate": 0.10, "latency_ms": 10.0},  # high ASR drop, ruins clean acc
        "defense_b": {"clean_accuracy": 0.93, "attack_success_rate": 0.25, "latency_ms": 5.0},   # good ASR drop, preserves clean acc
    }

    evals = selector.evaluate_candidate_results(candidates, baseline_metrics=baseline)
    best_result = selector.select_best_evaluated(evals)

    assert best_result["best_defense"] == "defense_b"
    assert best_result["selection_status"] == "robustness_verified"
    assert best_result["ranked_defenses"][0] == "defense_b"
    assert "Empirically selected 'defense_b'" in best_result["selection_rationale"]
    assert "reduced ASR by 60.0%" in best_result["selection_rationale"]
    assert "Defense successfully reinforced model robustness" in best_result["selection_rationale"]



