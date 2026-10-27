"""
Perturbation alignment metrics for AdverScan XAI.
Measures whether important features identified by XAI align with adversarial perturbations.
"""

from typing import Any, Dict, Optional
import numpy as np

from app.explainability.input_utils import to_numpy


def compute_perturbation_alignment(
    clean_input: Any,
    adversarial_input: Any,
    attribution: Any,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Compute alignment between adversarial perturbation vector and feature attributions.

    Args:
        clean_input: Clean input tensor or array.
        adversarial_input: Adversarial input tensor or array.
        attribution: Attribution map or array (from clean or adversarial explanation).
        top_k: Number of top features to evaluate intersection-over-union.

    Returns:
        Dictionary with cosine similarity, Pearson correlation, and top-k overlap.
    """
    if clean_input is None or adversarial_input is None or attribution is None:
        return {
            "alignment_status": "unavailable",
            "cosine_alignment": None,
            "pearson_correlation": None,
            "top_k_overlap": None,
        }

    try:
        clean_arr = to_numpy(clean_input, dtype=np.float64).ravel()
        adv_arr = to_numpy(adversarial_input, dtype=np.float64).ravel()
        attr_arr = to_numpy(attribution, dtype=np.float64).ravel()

        min_len = min(len(clean_arr), len(adv_arr), len(attr_arr))
        if min_len == 0:
            return {
                "alignment_status": "empty_input",
                "cosine_alignment": None,
                "pearson_correlation": None,
                "top_k_overlap": None,
            }

        # Truncate to common length
        clean_vec = clean_arr[:min_len]
        adv_vec = adv_arr[:min_len]
        attr_vec = np.abs(attr_arr[:min_len])

        # Perturbation magnitude
        pert_vec = np.abs(adv_vec - clean_vec)

        norm_pert = np.linalg.norm(pert_vec)
        norm_attr = np.linalg.norm(attr_vec)

        if norm_pert > 1e-12 and norm_attr > 1e-12:
            cosine_align = float(np.dot(pert_vec, attr_vec) / (norm_pert * norm_attr))
        else:
            cosine_align = 0.0

        # Pearson correlation
        if np.std(pert_vec) > 1e-12 and np.std(attr_vec) > 1e-12:
            r = np.corrcoef(pert_vec, attr_vec)[0, 1]
            pearson_corr = float(r) if not np.isnan(r) else 0.0
        else:
            pearson_corr = 0.0

        # Top-K Intersection over Union
        k = min(top_k, min_len)
        if k > 0:
            top_pert_indices = set(np.argsort(pert_vec)[::-1][:k])
            top_attr_indices = set(np.argsort(attr_vec)[::-1][:k])
            intersection = len(top_pert_indices & top_attr_indices)
            union = len(top_pert_indices | top_attr_indices)
            iou = float(intersection / union) if union > 0 else 1.0
        else:
            iou = 0.0

        return {
            "alignment_status": "success",
            "cosine_alignment": cosine_align,
            "pearson_correlation": pearson_corr,
            "top_k_overlap": iou,
        }
    except Exception as e:
        return {
            "alignment_status": f"error: {str(e)}",
            "cosine_alignment": None,
            "pearson_correlation": None,
            "top_k_overlap": None,
        }
