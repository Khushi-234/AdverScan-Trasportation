"""
Clean vs adversarial model behavior and attribution comparison for explainability module in AdverScan.
Supports image, text, tabular, and time-series modalities.
"""

from typing import Any, Dict, List, Optional
import numpy as np


def _extract_array(attr: Any) -> Optional[np.ndarray]:
    """Helper to unwrap attribution if wrapped in a dict or list."""
    if attr is None:
        return None
    if isinstance(attr, dict):
        if "attribution" in attr:
            attr = attr["attribution"]
        elif "heatmap" in attr:
            attr = attr["heatmap"]
        elif "raw_attribution" in attr:
            attr = attr["raw_attribution"]

    if attr is None:
        return None
    try:
        arr = np.asarray(attr, dtype=np.float64)
        return arr
    except Exception:
        return None


def compare_attributions(
    clean_attribution: Optional[Any],
    adv_attribution: Optional[Any],
    clean_top_features: Optional[List[Dict[str, Any]]] = None,
    adv_top_features: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Compare clean vs adversarial feature attribution maps using deterministic metrics.

    Mathematical definitions:
    - attribution_l1: Mean absolute element-wise difference: mean(|A_clean - A_adv|)
    - attribution_l2: Frobenius/Euclidean norm of difference: norm(A_clean - A_adv)
    - attribution_cosine_similarity: Cosine similarity between flattened attribution vectors:
        (A_clean . A_adv) / (norm(A_clean) * norm(A_adv))
    - attribution_mean_difference: Mean signed difference: mean(A_clean - A_adv)
    - top_feature_overlap: Jaccard similarity index of top features between clean and adversarial.

    Returns explicit unavailable status if attributions are missing or empty.
    """
    if clean_attribution is None or adv_attribution is None:
        return {
            "attribution_comparison_status": "unavailable",
            "attribution_l1": None,
            "attribution_l2": None,
            "attribution_cosine_similarity": None,
            "attribution_mean_difference": None,
            "top_feature_overlap": None,
        }

    try:
        clean_arr = _extract_array(clean_attribution)
        adv_arr = _extract_array(adv_attribution)

        if clean_arr is None or adv_arr is None:
            return {
                "attribution_comparison_status": "unavailable",
                "attribution_l1": None,
                "attribution_l2": None,
                "attribution_cosine_similarity": None,
                "attribution_mean_difference": None,
                "top_feature_overlap": None,
            }

        if clean_arr.size == 0 or adv_arr.size == 0 or clean_arr.shape != adv_arr.shape:
            return {
                "attribution_comparison_status": "shape_mismatch_or_empty",
                "attribution_l1": None,
                "attribution_l2": None,
                "attribution_cosine_similarity": None,
                "attribution_mean_difference": None,
                "top_feature_overlap": None,
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

        # Compute top feature overlap (Jaccard index)
        top_overlap = None
        if clean_top_features and adv_top_features:
            clean_feats = {f.get("feature", str(f)) for f in clean_top_features}
            adv_feats = {f.get("feature", str(f)) for f in adv_top_features}
            union = clean_feats | adv_feats
            top_overlap = float(len(clean_feats & adv_feats) / len(union)) if union else 1.0

        return {
            "attribution_comparison_status": "success",
            "attribution_l1": l1_diff,
            "attribution_l2": l2_diff,
            "attribution_cosine_similarity": cosine_sim,
            "attribution_mean_difference": mean_diff,
            "top_feature_overlap": top_overlap,
        }
    except Exception as e:
        return {
            "attribution_comparison_status": f"error: {str(e)}",
            "attribution_l1": None,
            "attribution_l2": None,
            "attribution_cosine_similarity": None,
            "attribution_mean_difference": None,
            "top_feature_overlap": None,
        }


def compare_explanations(
    clean_prediction: Any,
    adversarial_prediction: Any,
    clean_confidence: float,
    adversarial_confidence: float,
    clean_attribution: Optional[Any] = None,
    adv_attribution: Optional[Any] = None,
    clean_top_features: Optional[List[Dict[str, Any]]] = None,
    adv_top_features: Optional[List[Dict[str, Any]]] = None,
    target_class: Optional[Any] = None,
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
        clean_top_features: Optional list of top features for clean input.
        adv_top_features: Optional list of top features for adversarial input.
        target_class: Optional adversarial attack target class (for targeted attacks).

    Returns:
        Structured comparison dictionary.
    """
    prediction_changed = bool(clean_prediction != adversarial_prediction)
    confidence_difference = float(clean_confidence - adversarial_confidence)

    attr_comp = compare_attributions(
        clean_attribution,
        adv_attribution,
        clean_top_features=clean_top_features,
        adv_top_features=adv_top_features,
    )

    result = {
        "clean_prediction": clean_prediction,
        "adversarial_prediction": adversarial_prediction,
        "clean_confidence": float(clean_confidence),
        "adversarial_confidence": float(adversarial_confidence),
        "prediction_changed": prediction_changed,
        "confidence_difference": confidence_difference,
    }

    if target_class is not None:
        result["target_class"] = target_class
        result["targeted_attack_success"] = bool(adversarial_prediction == target_class)

    result.update(attr_comp)
    return result
