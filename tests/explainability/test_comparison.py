"""
Unit tests for comparison module in M6.
"""

import numpy as np
from app.explainability.comparison import compare_attributions, compare_explanations


def test_compare_explanations_same_prediction():
    comp = compare_explanations(
        clean_prediction=5,
        adversarial_prediction=5,
        clean_confidence=0.95,
        adversarial_confidence=0.90,
    )
    assert comp["clean_prediction"] == 5
    assert comp["adversarial_prediction"] == 5
    assert comp["prediction_changed"] is False
    assert abs(comp["confidence_difference"] - 0.05) < 1e-6
    assert comp["attribution_comparison_status"] == "unavailable"


def test_compare_explanations_changed_prediction():
    comp = compare_explanations(
        clean_prediction=5,
        adversarial_prediction=8,
        clean_confidence=0.95,
        adversarial_confidence=0.40,
    )
    assert comp["prediction_changed"] is True
    assert abs(comp["confidence_difference"] - 0.55) < 1e-6


def test_compare_attributions_numeric():
    clean_attr = np.array([[1.0, 2.0], [3.0, 4.0]])
    adv_attr = np.array([[1.0, 0.0], [3.0, 2.0]])

    # diff = [[0, 2], [0, 2]]
    # abs diff sum = 4, mean = 1.0
    # l2 diff = sqrt(0^2 + 2^2 + 0^2 + 2^2) = sqrt(8) ~ 2.8284
    # mean diff = (0 + 2 + 0 + 2)/4 = 1.0
    res = compare_attributions(clean_attr, adv_attr)

    assert res["attribution_comparison_status"] == "success"
    assert abs(res["attribution_l1"] - 1.0) < 1e-5
    assert abs(res["attribution_l2"] - np.sqrt(8)) < 1e-5
    assert abs(res["attribution_mean_difference"] - 1.0) < 1e-5
    assert res["attribution_cosine_similarity"] is not None


def test_compare_attributions_unavailable():
    res = compare_attributions(None, None)
    assert res["attribution_comparison_status"] == "unavailable"
    assert res["attribution_l1"] is None
    assert res["attribution_l2"] is None
    assert res["attribution_cosine_similarity"] is None
    assert res["attribution_mean_difference"] is None
