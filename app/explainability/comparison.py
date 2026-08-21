"""
Clean vs adversarial model behavior and attribution comparison for explainability module in AdverScan.
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


def compare_attributions(
    clean_attribution: Optional[Any],
    adv_attribution: Optional[Any],
) -> Dict[str, Any]:
    """
    Compare clean vs adversarial feature attribution maps using deterministic metrics.
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
    """
    prediction_changed = not _is_equal(clean_prediction, adversarial_prediction)
    confidence_difference = float(clean_confidence - adversarial_confidence)

    attr_comp = compare_attributions(clean_attribution, adv_attribution)

    result = {
        "clean_prediction": _to_clean_value(clean_prediction),
        "adversarial_prediction": _to_clean_value(adversarial_prediction),
        "clean_confidence": float(clean_confidence),
        "adversarial_confidence": float(adversarial_confidence),
        "prediction_changed": prediction_changed,
        "confidence_difference": confidence_difference,
    }
    result.update(attr_comp)

    return result
