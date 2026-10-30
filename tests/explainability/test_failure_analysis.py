"""
Unit tests for failure_analysis module in M6.
"""

from app.explainability.failure_analysis import analyze_failure, is_prediction_correct


def test_is_prediction_correct():
    assert is_prediction_correct(10, 10) is True
    assert is_prediction_correct(10, 20) is False
    assert is_prediction_correct(10, None) is None
    assert is_prediction_correct(10, [10, 20, 30]) is True
    assert is_prediction_correct(5, [10, 20, 30]) is False


def test_failure_analysis_clean_correct_adv_incorrect():
    # Case 1: Clean prediction correct, adversarial prediction incorrect
    res = analyze_failure(clean_prediction=14, adversarial_prediction=80, true_label=14)
    assert res["clean_correct"] is True
    assert res["adversarial_correct"] is False
    assert res["prediction_changed"] is True
    assert res["attack_caused_failure"] is True
    assert res["failure_mode"] == "clean_correct_to_adversarial_incorrect"


def test_failure_analysis_clean_incorrect_adv_incorrect():
    # Case 2: Clean prediction already incorrect, adversarial prediction incorrect
    res = analyze_failure(clean_prediction=12, adversarial_prediction=80, true_label=14)
    assert res["clean_correct"] is False
    assert res["adversarial_correct"] is False
    assert res["prediction_changed"] is True
    assert res["attack_caused_failure"] is False
    assert res["failure_mode"] == "clean_incorrect_to_adversarial_incorrect"


def test_failure_analysis_clean_correct_adv_correct():
    # Case 3: Both clean and adversarial predictions correct (unchanged)
    res = analyze_failure(clean_prediction=14, adversarial_prediction=14, true_label=14)
    assert res["clean_correct"] is True
    assert res["adversarial_correct"] is True
    assert res["prediction_changed"] is False
    assert res["attack_caused_failure"] is False
    assert res["failure_mode"] == "clean_correct_to_adversarial_correct"


def test_failure_analysis_prediction_changed_remains_correct():
    # Case 4: Prediction changed, but both remain correct (e.g. multi-label / multiple valid classes)
    res = analyze_failure(clean_prediction=14, adversarial_prediction=15, true_label=[14, 15])
    assert res["clean_correct"] is True
    assert res["adversarial_correct"] is True
    assert res["prediction_changed"] is True
    assert res["attack_caused_failure"] is False
    assert res["failure_mode"] == "prediction_changed_remains_correct"


def test_failure_analysis_missing_true_label():
    # Case 5: Missing ground truth label
    res = analyze_failure(clean_prediction=14, adversarial_prediction=80, true_label=None)
    assert res["clean_correct"] is None
    assert res["adversarial_correct"] is None
    assert res["prediction_changed"] is True
    assert res["attack_caused_failure"] is None
    assert res["failure_mode"] == "prediction_changed_unknown_correctness"
    assert "attack_reason" in res
    assert "recommended_defenses" in res
    assert len(res["recommended_defenses"]) > 0


def test_attack_reasoning_and_defenses_multi_modal():
    import numpy as np

    # Tabular
    tab_res = analyze_failure(
        clean_prediction=0,
        adversarial_prediction=1,
        true_label=0,
        clean_attribution=np.array([0.1, 0.2]),
        adv_attribution=np.array([0.1, 0.9]),
        feature_names=["age", "debt"],
        modality="tabular",
        attack_name="cap_pgd",
    )
    assert "debt" in tab_res["attack_reason"]
    assert any("Feature Scaling" in d["defense"] for d in tab_res["recommended_defenses"])

    # Image
    img_res = analyze_failure(
        clean_prediction=3,
        adversarial_prediction=8,
        true_label=3,
        clean_attribution=np.zeros((1, 3, 16, 16)),
        adv_attribution=np.ones((1, 3, 16, 16)),
        modality="image",
        attack_name="fgsm",
    )
    assert "spatial regions" in img_res["attack_reason"]
    assert any("Adversarial Training" in d["defense"] for d in img_res["recommended_defenses"])

    # Time series
    ts_res = analyze_failure(
        clean_prediction=0,
        adversarial_prediction=1,
        true_label=0,
        clean_attribution=np.zeros((10, 2)),
        adv_attribution=np.ones((10, 2)),
        sensor_names=["ecg_lead_1", "ecg_lead_2"],
        modality="timeseries",
        attack_name="time_gradient",
    )
    assert "sensor" in ts_res["attack_reason"]
    assert any("Wavelet" in d["defense"] for d in ts_res["recommended_defenses"])


def test_defense_hints_and_multi_stage_pipeline():
    import numpy as np
    from app.explainability.failure_analysis import build_defense_pipeline

    # Test 6-stage pipeline structure
    pipeline = build_defense_pipeline("image")
    expected_stages = [
        "1_preprocessing",
        "2_transformation",
        "3_detection",
        "4_smoothing",
        "5_feature_level",
        "6_ensemble",
    ]
    for stage_key in expected_stages:
        assert stage_key in pipeline
        assert "stage" in pipeline[stage_key]
        assert len(pipeline[stage_key]["techniques"]) >= 2

    # Test defense_hints generation
    res = analyze_failure(
        clean_prediction=0,
        adversarial_prediction=1,
        true_label=0,
        clean_attribution=np.array([0.1, 0.05, 0.05, 0.05]),
        adv_attribution=np.array([0.8, 0.6, 0.4, 0.05]),
        feature_names=["f0", "f1", "f2", "f3"],
        modality="tabular",
        attack_name="cap_pgd",
    )
    assert "defense_hints" in res
    assert len(res["defense_hints"]) >= 2
    
    # Check hint structure matching user requirement
    h1 = res["defense_hints"][0]
    assert "finding" in h1
    assert "evidence" in h1
    assert "suggested_family" in h1
    assert "stage" in h1
    assert "top3_attribution_share" in h1["evidence"]

