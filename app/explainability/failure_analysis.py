"""
Prediction-change and failure analysis for explainability module in AdverScan.
Identifies feature contributions to adversarial misclassification.
"""

from typing import Any, Dict, List, Optional
import numpy as np


def _normalize_label_scalar(val: Any) -> Any:
    """Normalize array, tensor, scalar, or one-hot vector into a comparable scalar or set of scalars."""
    if val is None:
        return None
    # PyTorch tensor
    if hasattr(val, "detach"):
        val = val.detach().cpu().numpy()
    # Numpy array
    if isinstance(val, np.ndarray):
        # One-hot vector
        if val.ndim == 1 and len(val) > 1 and np.issubdtype(val.dtype, np.number):
            if np.all((val >= 0.0) & (val <= 1.0)) and np.isclose(np.sum(val), 1.0, atol=1e-3):
                return int(np.argmax(val))
        if val.size == 1:
            return val.item()
        return [int(x) if np.issubdtype(type(x), np.integer) else x for x in val.tolist()]
    if hasattr(val, "item") and callable(val.item):
        try:
            return val.item()
        except Exception:
            pass
    return val


def is_prediction_correct(prediction: Any, true_label: Optional[Any]) -> Optional[bool]:
    """
    Check if a prediction matches the ground truth label.

    Handles numpy scalars, 0-d arrays, 1-element arrays, and one-hot encoded labels safely.

    Args:
        prediction: Predicted class or label (scalar, tensor, or array).
        true_label: Ground truth label or container of acceptable labels (scalar, list, array, or one-hot).

    Returns:
        True if correct, False if incorrect, None if true_label is unavailable.
    """
    if true_label is None:
        return None

    norm_pred = _normalize_label_scalar(prediction)
    norm_true = _normalize_label_scalar(true_label)

    if isinstance(norm_true, (list, tuple, set)):
        return norm_pred in norm_true
    if isinstance(norm_pred, (list, tuple, set)):
        return norm_true in norm_pred

    try:
        return bool(norm_pred == norm_true)
    except Exception:
        return bool(np.array_equal(norm_pred, norm_true))


def analyze_failure(
    clean_prediction: Any,
    adversarial_prediction: Any,
    true_label: Optional[Any] = None,
    clean_attribution: Optional[Any] = None,
    adv_attribution: Optional[Any] = None,
    clean_top_features: Optional[List[Dict[str, Any]]] = None,
    adv_top_features: Optional[List[Dict[str, Any]]] = None,
    target_class: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Perform deterministic analysis on prediction changes, failure modes, and feature contributions.

    Args:
        clean_prediction: Model prediction on clean input.
        adversarial_prediction: Model prediction on adversarial input.
        true_label: Optional ground truth label.
        clean_attribution: Optional clean feature attributions.
        adv_attribution: Optional adversarial feature attributions.
        clean_top_features: Top contributing features in clean input.
        adv_top_features: Top contributing features in adversarial input.
        target_class: Optional target class index for targeted attacks.

    Returns:
        Dictionary containing failure mode and contributing feature analysis.
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

    # Analyze which features contributed to the failure
    contributing_features = []
    if adv_top_features:
        # Features that have high importance in the adversarial example
        clean_feat_names = {f.get("feature"): f.get("importance", 0.0) for f in (clean_top_features or [])}
        for f in adv_top_features:
            name = f.get("feature")
            adv_imp = f.get("importance", 0.0)
            clean_imp = clean_feat_names.get(name, 0.0)
            imp_shift = float(adv_imp - clean_imp)
            contributing_features.append({
                "feature": name,
                "adversarial_importance": adv_imp,
                "clean_importance": clean_imp,
                "importance_shift": imp_shift,
                "rank": f.get("rank"),
            })

    # Targeted attack analysis
    targeted_info = None
    if target_class is not None:
        targeted_success = bool(adversarial_prediction == target_class)
        targeted_info = {
            "target_class": target_class,
            "targeted_success": targeted_success,
            "explanation": (
                f"Adversarial input successfully steered model to target class {target_class}."
                if targeted_success
                else f"Adversarial input failed to steer model to target class {target_class} (predicted {adversarial_prediction})."
            ),
        }

    # Summary narrative explanation
    if attack_caused_failure:
        explanation = (
            f"Adversarial attack caused model failure: prediction flipped from true class {clean_prediction} "
            f"to misclassified class {adversarial_prediction}."
        )
    elif prediction_changed:
        explanation = (
            f"Prediction changed from {clean_prediction} to {adversarial_prediction}."
        )
    else:
        explanation = f"Model prediction remained robust at class {clean_prediction}."

    result = {
        "clean_correct": clean_correct,
        "adversarial_correct": adversarial_correct,
        "prediction_changed": prediction_changed,
        "attack_caused_failure": attack_caused_failure,
        "true_label": true_label,
        "clean_prediction": clean_prediction,
        "adversarial_prediction": adversarial_prediction,
        "failure_mode": failure_mode,
        "contributing_features": contributing_features,
        "explanation": explanation,
    }

    if targeted_info is not None:
        result["targeted_attack"] = targeted_info

    return result
