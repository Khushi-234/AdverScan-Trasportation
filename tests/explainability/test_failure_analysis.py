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
