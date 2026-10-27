"""
Unit tests for explanation aggregation module in AdverScan XAI.
"""

from app.explainability.aggregate import aggregate_explanations
from app.explainability.explanation_result import ExplanationResult


def test_aggregate_explanations_empty():
    res = aggregate_explanations([])
    assert res["total_samples"] == 0
    assert res["prediction_change_rate"] == 0.0
    assert res["global_top_features"] == []


def test_aggregate_explanations_multiple_samples():
    r1 = ExplanationResult(
        attack_name="fgsm",
        technique="shap",
        clean_prediction=0,
        adversarial_prediction=1,
        clean_confidence=0.9,
        adversarial_confidence=0.4,
        prediction_changed=True,
        attack_caused_failure=True,
        modality="tabular",
        comparison={"confidence_difference": 0.5, "attribution_l1": 0.2, "attribution_cosine_similarity": 0.8},
        failure_analysis={"failure_mode": "clean_correct_to_adversarial_incorrect"},
        attribution={
            "adversarial": {
                "top_features": [
                    {"feature": "feature_0", "importance": 0.8, "rank": 1},
                    {"feature": "feature_1", "importance": 0.4, "rank": 2},
                ]
            }
        },
    )

    r2 = ExplanationResult(
        attack_name="pgd",
        technique="shap",
        clean_prediction=0,
        adversarial_prediction=0,
        clean_confidence=0.9,
        adversarial_confidence=0.8,
        prediction_changed=False,
        attack_caused_failure=False,
        modality="tabular",
        comparison={"confidence_difference": 0.1, "attribution_l1": 0.05, "attribution_cosine_similarity": 0.95},
        failure_analysis={"failure_mode": "clean_correct_to_adversarial_correct"},
        attribution={
            "adversarial": {
                "top_features": [
                    {"feature": "feature_0", "importance": 0.6, "rank": 1},
                    {"feature": "feature_2", "importance": 0.3, "rank": 2},
                ]
            }
        },
    )

    agg = aggregate_explanations([r1, r2], top_k_features=3)

    assert agg["total_samples"] == 2
    assert agg["prediction_change_rate"] == 0.5
    assert agg["attack_failure_rate"] == 0.5
    assert abs(agg["mean_confidence_drop"] - 0.3) < 1e-5
    assert "fgsm" in agg["attacks"]
    assert "pgd" in agg["attacks"]

    # feature_0 was present in both samples
    top_feats = agg["global_top_features"]
    assert len(top_feats) > 0
    assert top_feats[0]["feature"] == "feature_0"
    assert top_feats[0]["frequency"] == 2
    assert top_feats[0]["frequency_ratio"] == 1.0

    # Failure mode breakdown
    assert "clean_correct_to_adversarial_incorrect" in agg["failure_mode_breakdown"]
    assert agg["failure_mode_breakdown"]["clean_correct_to_adversarial_incorrect"] == 1
