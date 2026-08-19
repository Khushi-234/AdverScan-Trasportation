"""
Unit tests for ExplanationResult DTO.
"""

import json
from app.explainability.explanation_result import ExplanationResult


def test_explanation_result_initialization():
    res = ExplanationResult(
        attack_name="fgsm",
        technique="shap",
        true_label=14,
        clean_prediction=14,
        adversarial_prediction=17,
        clean_confidence=0.98,
        adversarial_confidence=0.61,
        prediction_changed=True,
        attack_caused_failure=True,
        attribution={"status": "unavailable"},
        comparison={"confidence_difference": 0.37},
        failure_analysis={"clean_correct": True, "adversarial_correct": False},
        metadata={"model": "test_net"},
    )

    assert res.attack_name == "fgsm"
    assert res.technique == "shap"
    assert res.true_label == 14
    assert res.clean_prediction == 14
    assert res.adversarial_prediction == 17
    assert res.clean_confidence == 0.98
    assert res.adversarial_confidence == 0.61
    assert res.prediction_changed is True
    assert res.attack_caused_failure is True
    assert res.attribution == {"status": "unavailable"}
    assert res.comparison["confidence_difference"] == 0.37


def test_explanation_result_to_dict():
    res = ExplanationResult(
        attack_name="pgd",
        technique="lime",
        clean_prediction=0,
        adversarial_prediction=1,
        clean_confidence=0.9,
        adversarial_confidence=0.4,
        prediction_changed=True,
    )
    d = res.to_dict()

    assert isinstance(d, dict)
    assert d["attack_name"] == "pgd"
    assert d["technique"] == "lime"
    assert d["clean_prediction"] == 0
    assert d["adversarial_prediction"] == 1
    assert d["prediction_changed"] is True


def test_explanation_result_save_json(tmp_path):
    res = ExplanationResult(
        attack_name="deepfool",
        technique="shap",
        clean_prediction=3,
        adversarial_prediction=3,
        clean_confidence=0.95,
        adversarial_confidence=0.92,
        prediction_changed=False,
        attack_caused_failure=False,
    )
    out_file = tmp_path / "subdir" / "explanation.json"
    res.save_json(out_file)

    assert out_file.exists()
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["attack_name"] == "deepfool"
    assert data["technique"] == "shap"
    assert data["prediction_changed"] is False
