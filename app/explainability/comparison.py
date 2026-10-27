"""
Clean vs adversarial model behavior and attribution comparison for explainability module in AdverScan.
Supports image, text, tabular, and time-series modalities.
"""

from typing import Any, Dict, List, Optional
import numpy as np

try:
    from scipy.stats import spearmanr
    _SCIPY_AVAILABLE = True
except ImportError:
    spearmanr = None
    _SCIPY_AVAILABLE = False


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
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Compare clean vs adversarial feature attribution maps using deterministic and scale-invariant metrics.

    Mathematical definitions:
    - attribution_l1: Raw mean absolute element-wise difference: mean(|A_clean - A_adv|)
    - attribution_l2: Raw Euclidean norm of difference: norm(A_clean - A_adv)
    - normalized_attribution_l1: Scale-invariant L1 difference over unit L1-normalized attributions:
        sum(| A_clean/sum(|A_clean|) - A_adv/sum(|A_adv|) |), bounded in [0, 2]
    - attribution_cosine_similarity: Cosine similarity between flattened attribution vectors:
        (A_clean . A_adv) / (norm(A_clean) * norm(A_adv))
    - attribution_spearman_correlation: Spearman rank correlation of feature importance ranks.
    - attribution_mean_difference: Mean signed difference: mean(A_clean - A_adv)
    - top_feature_overlap: Jaccard similarity index of top features between clean and adversarial.

    Returns:
        Structured attribution comparison dictionary.
    """
    if clean_attribution is None or adv_attribution is None:
        return {
            "attribution_comparison_status": "unavailable",
            "attribution_l1": None,
            "attribution_l2": None,
            "normalized_attribution_l1": None,
            "attribution_cosine_similarity": None,
            "attribution_spearman_correlation": None,
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
                "normalized_attribution_l1": None,
                "attribution_cosine_similarity": None,
                "attribution_spearman_correlation": None,
                "attribution_mean_difference": None,
                "top_feature_overlap": None,
            }

        if clean_arr.size == 0 or adv_arr.size == 0 or clean_arr.shape != adv_arr.shape:
            return {
                "attribution_comparison_status": "shape_mismatch_or_empty",
                "attribution_l1": None,
                "attribution_l2": None,
                "normalized_attribution_l1": None,
                "attribution_cosine_similarity": None,
                "attribution_spearman_correlation": None,
                "attribution_mean_difference": None,
                "top_feature_overlap": None,
            }

        diff = clean_arr - adv_arr
        l1_diff = float(np.mean(np.abs(diff)))
        l2_diff = float(np.linalg.norm(diff))
        mean_diff = float(np.mean(diff))

        clean_flat = clean_arr.flatten()
        adv_flat = adv_arr.flatten()

        # Scale-invariant L1 normalization
        clean_sum = np.sum(np.abs(clean_flat))
        adv_sum = np.sum(np.abs(adv_flat))

        if clean_sum > 1e-12 and adv_sum > 1e-12:
            clean_normed = clean_flat / clean_sum
            adv_normed = adv_flat / adv_sum
            norm_l1 = float(np.sum(np.abs(clean_normed - adv_normed)))
        else:
            norm_l1 = 0.0

        norm_clean = np.linalg.norm(clean_flat)
        norm_adv = np.linalg.norm(adv_flat)

        if norm_clean > 1e-12 and norm_adv > 1e-12:
            cosine_sim = float(np.dot(clean_flat, adv_flat) / (norm_clean * norm_adv))
        else:
            cosine_sim = 1.0 if (norm_clean <= 1e-12 and norm_adv <= 1e-12) else 0.0

        # Spearman rank correlation
        if _SCIPY_AVAILABLE and spearmanr is not None:
            sp_res = spearmanr(clean_flat, adv_flat)
            spearman_corr = float(sp_res.statistic) if not np.isnan(sp_res.statistic) else 0.0
        else:
            rank_c = np.argsort(np.argsort(clean_flat))
            rank_a = np.argsort(np.argsort(adv_flat))
            if np.std(rank_c) > 1e-12 and np.std(rank_a) > 1e-12:
                r_sp = np.corrcoef(rank_c, rank_a)[0, 1]
                spearman_corr = float(r_sp) if not np.isnan(r_sp) else 0.0
            else:
                spearman_corr = 0.0

        # Compute top feature overlap directly from full vectors or top_features
        k = min(top_k, len(clean_flat))
        if k > 0:
            top_c = set(np.argsort(np.abs(clean_flat))[::-1][:k])
            top_a = set(np.argsort(np.abs(adv_flat))[::-1][:k])
            union = top_c | top_a
            top_overlap = float(len(top_c & top_a) / len(union)) if union else 1.0
        elif clean_top_features and adv_top_features:
            clean_feats = {f.get("feature", str(f)) for f in clean_top_features}
            adv_feats = {f.get("feature", str(f)) for f in adv_top_features}
            union = clean_feats | adv_feats
            top_overlap = float(len(clean_feats & adv_feats) / len(union)) if union else 1.0
        else:
            top_overlap = None

        return {
            "attribution_comparison_status": "success",
            "attribution_l1": l1_diff,
            "attribution_l2": l2_diff,
            "normalized_attribution_l1": norm_l1,
            "attribution_cosine_similarity": cosine_sim,
            "attribution_spearman_correlation": spearman_corr,
            "attribution_mean_difference": mean_diff,
            "top_feature_overlap": top_overlap,
        }
    except Exception as e:
        return {
            "attribution_comparison_status": f"error: {str(e)}",
            "attribution_l1": None,
            "attribution_l2": None,
            "normalized_attribution_l1": None,
            "attribution_cosine_similarity": None,
            "attribution_spearman_correlation": None,
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
    clean_probs: Optional[np.ndarray] = None,
    adv_probs: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compare clean vs. adversarial prediction, confidence, probabilities, and attribution.

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
        clean_probs: Optional full probability vector for clean input.
        adv_probs: Optional full probability vector for adversarial input.

    Returns:
        Structured comparison dictionary.
    """
    prediction_changed = bool(clean_prediction != adversarial_prediction)
    confidence_difference = float(clean_confidence - adversarial_confidence)

    # Compute clean_class_confidence_drop: p(clean_class|clean) - p(clean_class|adv)
    clean_class_confidence_drop = None
    logit_margin = None

    if clean_probs is not None and adv_probs is not None:
        try:
            cp = np.asarray(clean_probs, dtype=np.float64).ravel()
            ap = np.asarray(adv_probs, dtype=np.float64).ravel()
            c_idx = int(clean_prediction) if isinstance(clean_prediction, (int, np.integer)) else 0
            if 0 <= c_idx < len(cp) and 0 <= c_idx < len(ap):
                p_clean_at_clean = float(cp[c_idx])
                p_clean_at_adv = float(ap[c_idx])
                clean_class_confidence_drop = float(p_clean_at_clean - p_clean_at_adv)

            # Logit / prob margin on clean input
            if len(cp) >= 2:
                sorted_cp = np.sort(cp)[::-1]
                logit_margin = float(sorted_cp[0] - sorted_cp[1])
        except Exception:
            pass

    if clean_class_confidence_drop is None:
        clean_class_confidence_drop = confidence_difference

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
        "clean_class_confidence_drop": clean_class_confidence_drop,
        "logit_margin": logit_margin,
    }

    if target_class is not None:
        result["target_class"] = target_class
        result["targeted_attack_success"] = bool(adversarial_prediction == target_class)

    result.update(attr_comp)
    return result
