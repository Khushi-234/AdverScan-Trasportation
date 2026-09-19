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

