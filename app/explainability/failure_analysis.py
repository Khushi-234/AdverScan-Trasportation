"""
Prediction-change and failure analysis for explainability module in AdverScan.
"""

from typing import Any, Dict, Optional


def is_prediction_correct(prediction: Any, true_label: Optional[Any]) -> Optional[bool]:
    """
    Check if a prediction matches the ground truth label.

    Args:
        prediction: Predicted class or label.
        true_label: Ground truth label or container of acceptable labels.

    Returns:
        True if correct, False if incorrect, None if true_label is unavailable.
    """
    if true_label is None:
        return None
    if isinstance(true_label, (list, tuple, set)):
        return prediction in true_label
    if isinstance(prediction, (list, tuple, set)):
        return true_label in prediction
    return bool(prediction == true_label)


def analyze_failure(
    clean_prediction: Any,
    adversarial_prediction: Any,
    true_label: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Perform deterministic analysis on prediction changes and failure modes.

    Args:
        clean_prediction: Model prediction on clean input.
        adversarial_prediction: Model prediction on adversarial input.
        true_label: Optional ground truth label.

    Returns:
        Dictionary containing:
            - clean_correct: Optional[bool]
            - adversarial_correct: Optional[bool]
            - prediction_changed: bool
            - attack_caused_failure: Optional[bool]
            - true_label: Optional[Any]
            - clean_prediction: Any
            - adversarial_prediction: Any
            - failure_mode: str
    """
    prediction_changed = bool(clean_prediction != adversarial_prediction)
    clean_correct = is_prediction_correct(clean_prediction, true_label)
    adversarial_correct = is_prediction_correct(adversarial_prediction, true_label)

    if true_label is None:
        attack_caused_failure = None
        if prediction_changed:
            failure_mode = "prediction_changed_unknown_correctness"
        else:
            failure_mode = "prediction_unchanged_unknown_correctness"
    else:
        attack_caused_failure = bool(clean_correct is True and adversarial_correct is False)

        if clean_correct is True and adversarial_correct is False:
            failure_mode = "clean_correct_to_adversarial_incorrect"
        elif clean_correct is False and adversarial_correct is False:
            failure_mode = "clean_incorrect_to_adversarial_incorrect"
        elif clean_correct is True and adversarial_correct is True:
            if prediction_changed:
                failure_mode = "prediction_changed_remains_correct"
            else:
                failure_mode = "clean_correct_to_adversarial_correct"
        else:  # clean_correct is False, adversarial_correct is True
            failure_mode = "clean_incorrect_to_adversarial_correct"

    return {
        "clean_correct": clean_correct,
        "adversarial_correct": adversarial_correct,
        "prediction_changed": prediction_changed,
        "attack_caused_failure": attack_caused_failure,
        "true_label": true_label,
        "clean_prediction": clean_prediction,
        "adversarial_prediction": adversarial_prediction,
        "failure_mode": failure_mode,
    }
