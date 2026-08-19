"""
Clean vs adversarial model behavior and attribution comparison for explainability module in AdverScan.
"""

from typing import Any, Dict, Optional
import numpy as np


def compare_attributions(
    clean_attribution: Optional[Any],
    adv_attribution: Optional[Any],
) -> Dict[str, Any]:
    """
    Compare clean vs adversarial feature attribution maps using deterministic metrics.

    Mathematical definitions:
    - attribution_l1: Mean absolute element-wise difference: mean(|A_clean - A_adv|)
    - attribution_l2: Frobenius/Euclidean norm of difference: norm(A_clean - A_adv)
    - attribution_cosine_similarity: Cosine similarity between flattened attribution vectors:
        (A_clean . A_adv) / (norm(A_clean) * norm(A_adv))
    - attribution_mean_difference: Mean signed difference: mean(A_clean - A_adv)

    Returns explicit unavailable status if attributions are missing or empty.
    """
    if clean_attribution is None or adv_attribution is None:
        return {
            "attribution_comparison_status": "unavailable",
            "attribution_l1": None,
            "attribution_l2": None,
            "attribution_cosine_similarity": None,
            "attribution_mean_difference": None,
        }

    try:
        clean_arr = np.asarray(clean_attribution, dtype=np.float64)
        adv_arr = np.asarray(adv_attribution, dtype=np.float64)

        if clean_arr.size == 0 or adv_arr.size == 0 or clean_arr.shape != adv_arr.shape:
            return {
                "attribution_comparison_status": "shape_mismatch_or_empty",
                "attribution_l1": None,
                "attribution_l2": None,
                "attribution_cosine_similarity": None,
                "attribution_mean_difference": None,
            }

        diff = clean_arr - adv_arr
        l1_diff = float(np.mean(np.abs(diff)))
        l2_diff = float(np.linalg.norm(diff))
        mean_diff = float(np.mean(diff))

        clean_flat = clean_arr.flatten()
        adv_flat = adv_arr.flatten()

        norm_clean = np.linalg.norm(clean_flat)
        norm_adv = np.linalg.norm(adv_flat)

        if norm_clean > 1e-12 and norm_adv > 1e-12:
            cosine_sim = float(np.dot(clean_flat, adv_flat) / (norm_clean * norm_adv))
        else:
            cosine_sim = 1.0 if (norm_clean <= 1e-12 and norm_adv <= 1e-12) else 0.0

        return {
            "attribution_comparison_status": "success",
            "attribution_l1": l1_diff,
            "attribution_l2": l2_diff,
            "attribution_cosine_similarity": cosine_sim,
            "attribution_mean_difference": mean_diff,
        }
    except Exception as e:
        return {
            "attribution_comparison_status": f"error: {str(e)}",
            "attribution_l1": None,
            "attribution_l2": None,
            "attribution_cosine_similarity": None,
            "attribution_mean_difference": None,
        }


def compare_explanations(
    clean_prediction: Any,
    adversarial_prediction: Any,
    clean_confidence: float,
    adversarial_confidence: float,
    clean_attribution: Optional[Any] = None,
    adv_attribution: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Compare clean vs. adversarial prediction, confidence, and attribution.

    Args:
        clean_prediction: Model prediction on clean input.
        adversarial_prediction: Model prediction on adversarial input.
        clean_confidence: Prediction confidence on clean input.
        adversarial_confidence: Prediction confidence on adversarial input.
        clean_attribution: Optional clean feature attribution map/array.
        adv_attribution: Optional adversarial feature attribution map/array.

    Returns:
        Structured comparison dictionary.
    """
    prediction_changed = bool(clean_prediction != adversarial_prediction)
    confidence_difference = float(clean_confidence - adversarial_confidence)

    attr_comp = compare_attributions(clean_attribution, adv_attribution)

    result = {
        "clean_prediction": clean_prediction,
        "adversarial_prediction": adversarial_prediction,
        "clean_confidence": float(clean_confidence),
        "adversarial_confidence": float(adversarial_confidence),
        "prediction_changed": prediction_changed,
        "confidence_difference": confidence_difference,
    }
    result.update(attr_comp)

    return result
