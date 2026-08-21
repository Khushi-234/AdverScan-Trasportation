"""
Prediction-change and failure analysis for explainability module in AdverScan.
"""

from typing import Any, Dict, Optional
import numpy as np
import torch


def _to_clean_value(val: Any) -> Any:
    """Helper to convert PyTorch tensors or NumPy arrays to Python primitives/lists."""
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy().tolist() if val.ndim > 0 else val.item()
    if isinstance(val, np.ndarray):
        return val.tolist() if val.ndim > 0 else val.item()
    return val


def _is_equal(a: Any, b: Any) -> bool:
    """Safely compare equality between scalars, lists, tensors, or numpy arrays."""
    clean_a = _to_clean_value(a)
    clean_b = _to_clean_value(b)
    if isinstance(clean_a, list) or isinstance(clean_b, list):
        return clean_a == clean_b
    return bool(clean_a == clean_b)


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
    pred_val = _to_clean_value(prediction)
    true_val = _to_clean_value(true_label)

    if isinstance(true_val, (list, tuple, set)):
        if isinstance(pred_val, (list, tuple, set)):
            return pred_val == list(true_val)
        return pred_val in true_val
    if isinstance(pred_val, (list, tuple, set)):
        return true_val in pred_val
    return bool(pred_val == true_val)


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
        Dictionary containing failure mode analysis metrics.
    """
    prediction_changed = not _is_equal(clean_prediction, adversarial_prediction)
    clean_correct = is_prediction_correct(clean_prediction, true_label)
    adversarial_correct = is_prediction_correct(adversarial_prediction, true_label)

    clean_pred_val = _to_clean_value(clean_prediction)
    adv_pred_val = _to_clean_value(adversarial_prediction)
    true_label_val = _to_clean_value(true_label)

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
        "true_label": true_label_val,
        "clean_prediction": clean_pred_val,
        "adversarial_prediction": adv_pred_val,
        "failure_mode": failure_mode,
    }
