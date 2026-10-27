"""
Perturbation alignment metrics for AdverScan XAI.
Measures whether important features identified by clean XAI align with adversarial perturbations.
"""

from typing import Any, Dict, Optional, Union
import numpy as np

try:
    from scipy.stats import spearmanr
    _SCIPY_AVAILABLE = True
except ImportError:
    spearmanr = None
    _SCIPY_AVAILABLE = False

from app.explainability.input_utils import to_numpy


def compute_perturbation_alignment(
    clean_input: Any,
    adversarial_input: Any,
    attribution: Any,
    top_k: Union[int, float] = 5,
    background_std: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Compute alignment between adversarial perturbation vector and clean feature attributions.

    Args:
        clean_input: Clean input tensor or array.
        adversarial_input: Adversarial input tensor or array.
        attribution: Clean attribution map or array.
        top_k: Number of top features (if int >= 1) or fraction of features (if float in (0, 1]).
        background_std: Optional per-feature standard deviation for standardizing perturbation.

    Returns:
        Dictionary with cosine alignment, Pearson correlation, Spearman rank correlation,
        top-k IoU, and perturbation budget fraction on top-k features.
    """
    if clean_input is None or adversarial_input is None or attribution is None:
        return {
            "alignment_status": "unavailable",
            "cosine_alignment": None,
            "pearson_correlation": None,
            "spearman_correlation": None,
            "top_k_overlap": None,
            "budget_fraction": None,
        }

    try:
        clean_arr = to_numpy(clean_input, dtype=np.float64)
        adv_arr = to_numpy(adversarial_input, dtype=np.float64)
        attr_arr = to_numpy(attribution, dtype=np.float64)

        if clean_arr.shape != adv_arr.shape:
            return {
                "alignment_status": "shape_mismatch",
                "message": f"Clean shape {clean_arr.shape} != adv shape {adv_arr.shape}",
                "cosine_alignment": None,
                "pearson_correlation": None,
                "spearman_correlation": None,
                "top_k_overlap": None,
                "budget_fraction": None,
            }

        # Align attribution shape with input shape
        # Handle cases like Grad-CAM (1, 1, H, W) vs Image (1, C, H, W) or (H, W) vs (C, H, W)
        pert_raw = np.abs(adv_arr - clean_arr)

        if attr_arr.shape != pert_raw.shape:
            # Squeeze single-batch dimensions
            attr_sq = np.squeeze(attr_arr)
            pert_sq = np.squeeze(pert_raw)

            if attr_sq.shape == pert_sq.shape:
                pert_eval = pert_sq
                attr_eval = attr_sq
            elif attr_sq.ndim == 2 and pert_sq.ndim == 3 and pert_sq.shape[0] in (1, 3, 4):
                # e.g., (H, W) vs (C, H, W) -> reduce input channels by taking mean
                pert_eval = np.mean(pert_sq, axis=0)
                attr_eval = attr_sq
            elif attr_sq.ndim == 2 and pert_sq.ndim == 3 and pert_sq.shape[-1] in (1, 3, 4):
                # e.g., (H, W) vs (H, W, C) -> reduce channels
                pert_eval = np.mean(pert_sq, axis=-1)
                attr_eval = attr_sq
            else:
                try:
                    attr_eval = np.broadcast_to(attr_arr, pert_raw.shape)
                    pert_eval = pert_raw
                except ValueError:
                    return {
                        "alignment_status": "shape_mismatch",
                        "message": f"Attribution shape {attr_arr.shape} cannot be aligned with input shape {pert_raw.shape}",
                        "cosine_alignment": None,
                        "pearson_correlation": None,
                        "spearman_correlation": None,
                        "top_k_overlap": None,
                        "budget_fraction": None,
                    }
        else:
            pert_eval = pert_raw
            attr_eval = attr_arr

        # Flatten aligned vectors
        pert_vec = pert_eval.ravel()
        attr_vec = np.abs(attr_eval.ravel())

        if len(pert_vec) != len(attr_vec):
            return {
                "alignment_status": "shape_mismatch",
                "message": f"Perturbation features ({len(pert_vec)}) != attribution features ({len(attr_vec)})",
                "cosine_alignment": None,
                "pearson_correlation": None,
                "spearman_correlation": None,
                "top_k_overlap": None,
                "budget_fraction": None,
            }

        if len(pert_vec) == 0:
            return {
                "alignment_status": "empty_input",
                "cosine_alignment": None,
                "pearson_correlation": None,
                "spearman_correlation": None,
                "top_k_overlap": None,
                "budget_fraction": None,
            }

        # Standardize |delta| per feature if background_std provided
        if background_std is not None:
            bg_std = to_numpy(background_std, dtype=np.float64).ravel()
            if len(bg_std) == len(pert_vec):
                pert_vec = pert_vec / (bg_std + 1e-8)
        elif np.std(pert_vec) > 1e-12:
            pert_vec = pert_vec / np.std(pert_vec)

        # Cosine alignment
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

        # Spearman rank correlation
        if _SCIPY_AVAILABLE and spearmanr is not None:
            res_sp = spearmanr(pert_vec, attr_vec)
            spearman_corr = float(res_sp.statistic) if not np.isnan(res_sp.statistic) else 0.0
        else:
            # Fallback rank correlation via numpy
            rank_pert = np.argsort(np.argsort(pert_vec))
            rank_attr = np.argsort(np.argsort(attr_vec))
            if np.std(rank_pert) > 1e-12 and np.std(rank_attr) > 1e-12:
                r_sp = np.corrcoef(rank_pert, rank_attr)[0, 1]
                spearman_corr = float(r_sp) if not np.isnan(r_sp) else 0.0
            else:
                spearman_corr = 0.0

        # Determine k (handling percentage for large feature spaces like images)
        total_feats = len(pert_vec)
        if isinstance(top_k, float) and 0.0 < top_k <= 1.0:
            k = max(1, int(top_k * total_feats))
        else:
            k = min(int(top_k), total_feats)

        # Top-K Intersection over Union
        if k > 0:
            top_pert_indices = set(np.argsort(pert_vec)[::-1][:k])
            top_attr_indices = set(np.argsort(attr_vec)[::-1][:k])
            intersection = len(top_pert_indices & top_attr_indices)
            union = len(top_pert_indices | top_attr_indices)
            iou = float(intersection / union) if union > 0 else 1.0

            # Perturbation budget fraction on top-k features
            top_attr_idx_arr = np.argsort(attr_vec)[::-1][:k]
            total_budget = np.sum(pert_vec)
            top_k_budget = np.sum(pert_vec[top_attr_idx_arr])
            budget_fraction = float(top_k_budget / (total_budget + 1e-12))
        else:
            iou = 0.0
            budget_fraction = 0.0

        return {
            "alignment_status": "success",
            "cosine_alignment": cosine_align,
            "pearson_correlation": pearson_corr,
            "spearman_correlation": spearman_corr,
            "top_k_overlap": iou,
            "budget_fraction": budget_fraction,
            "k_used": k,
        }
    except Exception as e:
        return {
            "alignment_status": f"error: {str(e)}",
            "cosine_alignment": None,
            "pearson_correlation": None,
            "spearman_correlation": None,
            "top_k_overlap": None,
            "budget_fraction": None,
        }
