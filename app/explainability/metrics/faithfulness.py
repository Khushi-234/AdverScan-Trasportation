"""
Explanation faithfulness metrics for AdverScan XAI.
Evaluates explanation faithfulness via feature ablation curves (deletion & insertion) and AUC.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch

from app.explainability.utils import predict_proba_np, safe_predict, to_numpy, to_tensor


def compute_faithfulness(
    model: Any,
    inputs: Any,
    attribution: Any,
    top_k: Union[int, float] = 5,
    background_inputs: Optional[Any] = None,
    baseline_value: Optional[float] = None,
    num_curve_steps: int = 5,
) -> Dict[str, Any]:
    """
    Evaluate explanation faithfulness by deletion/insertion feature ablation curves.

    Ablating the most important features should significantly reduce model confidence
    for the target class (comprehensiveness / deletion curve).
    Progressively inserting top features into a baseline should increase target confidence
    (sufficiency / insertion curve).

    Args:
        model: Target ML model or predict function.
        inputs: Input sample (tensor or array).
        attribution: Attribution values corresponding to inputs.
        top_k: Max features to mask for single-k evaluation (or fraction if float in (0, 1]).
        background_inputs: Optional background dataset to compute baseline mean.
        baseline_value: Optional scalar fallback fill value for masked features.
        num_curve_steps: Number of evaluation points along the ablation curve.

    Returns:
        Dictionary with deletion AUC, insertion AUC, single-k drop, and random baseline comparison.
    """
    if inputs is None or attribution is None or model is None:
        return {
            "faithfulness_status": "unavailable",
            "faithfulness_score": None,
            "deletion_auc": None,
            "insertion_auc": None,
            "original_confidence": None,
            "ablated_confidence": None,
        }

    try:
        t_input = to_tensor(inputs).clone()
        input_shape = t_input.shape
        flat_input = t_input.reshape(-1).clone()
        num_features = len(flat_input)

        attr_arr = to_numpy(attribution, dtype=np.float32).ravel()
        if len(attr_arr) != num_features:
            # Try to match or slice appropriately
            min_len = min(len(attr_arr), num_features)
            attr_arr = attr_arr[:min_len]

        # Determine baseline tensor
        if background_inputs is not None:
            bg_np = to_numpy(background_inputs, dtype=np.float32)
            bg_mean = np.mean(bg_np, axis=0)
            baseline_tensor = to_tensor(bg_mean, device=t_input.device).reshape(-1)
            if len(baseline_tensor) != num_features:
                baseline_tensor = torch.zeros(num_features, device=t_input.device)
        elif baseline_value is not None:
            baseline_tensor = torch.full((num_features,), float(baseline_value), device=t_input.device)
        else:
            baseline_tensor = torch.zeros(num_features, device=t_input.device)

        # Baseline evaluation on original input
        orig_probs = predict_proba_np(model, t_input.unsqueeze(0) if t_input.ndim == input_shape else t_input)
        if orig_probs.ndim == 2:
            pred_class = int(np.argmax(orig_probs[0]))
            orig_conf = float(orig_probs[0, pred_class])
        else:
            pred_class = 0
            orig_conf = float(orig_probs.ravel()[0])

        # Feature ranking by attribution magnitude (ensure contiguous array with positive strides)
        sorted_indices = np.ascontiguousarray(np.argsort(np.abs(attr_arr))[::-1].copy())

        # Evaluate deletion and insertion curves
        # Steps from 0% to 100% ablation of top features
        steps = np.linspace(0.0, 1.0, num_curve_steps + 1)
        deletion_confs = [orig_conf]
        insertion_confs = []

        # Baseline only confidence
        baseline_input = baseline_tensor.reshape(input_shape)
        base_probs = predict_proba_np(model, baseline_input.unsqueeze(0) if baseline_input.ndim == input_shape else baseline_input)
        base_conf = float(base_probs[0, pred_class]) if base_probs.ndim == 2 else float(base_probs.ravel()[0])
        insertion_confs.append(base_conf)

        for frac in steps[1:]:
            n_mask = max(1, int(frac * len(sorted_indices)))

            # Deletion: replace top n_mask features with baseline
            cur_del = flat_input.clone()
            cur_del[sorted_indices[:n_mask]] = baseline_tensor[sorted_indices[:n_mask]]
            del_out = predict_proba_np(model, cur_del.reshape(input_shape).unsqueeze(0) if cur_del.ndim == input_shape else cur_del.reshape(input_shape))
            del_conf = float(del_out[0, pred_class]) if del_out.ndim == 2 else float(del_out.ravel()[0])
            deletion_confs.append(del_conf)

            # Insertion: start with baseline and insert top n_mask features
            cur_ins = baseline_tensor.clone()
            cur_ins[sorted_indices[:n_mask]] = flat_input[sorted_indices[:n_mask]]
            ins_out = predict_proba_np(model, cur_ins.reshape(input_shape).unsqueeze(0) if cur_ins.ndim == input_shape else cur_ins.reshape(input_shape))
            ins_conf = float(ins_out[0, pred_class]) if ins_out.ndim == 2 else float(ins_out.ravel()[0])
            insertion_confs.append(ins_conf)

        # Compute AUC using trapezoid rule
        deletion_auc = float(np.trapezoid(deletion_confs, steps))
        insertion_auc = float(np.trapezoid(insertion_confs, steps))

        # Random ablation baseline curve for comparison
        rng = np.random.RandomState(42)
        random_indices = rng.permutation(num_features)
        random_del_confs = [orig_conf]
        for frac in steps[1:]:
            n_mask = max(1, int(frac * len(random_indices)))
            cur_rand = flat_input.clone()
            cur_rand[random_indices[:n_mask]] = baseline_tensor[random_indices[:n_mask]]
            r_out = predict_proba_np(model, cur_rand.reshape(input_shape).unsqueeze(0) if cur_rand.ndim == input_shape else cur_rand.reshape(input_shape))
            r_conf = float(r_out[0, pred_class]) if r_out.ndim == 2 else float(r_out.ravel()[0])
            random_del_confs.append(r_conf)
        random_deletion_auc = float(np.trapezoid(random_del_confs, steps))

        # Single-k evaluation
        if isinstance(top_k, float) and 0.0 < top_k <= 1.0:
            k = max(1, int(top_k * num_features))
        else:
            k = min(int(top_k), num_features)

        k_del = flat_input.clone()
        k_del[sorted_indices[:k]] = baseline_tensor[sorted_indices[:k]]
        k_out = predict_proba_np(model, k_del.reshape(input_shape).unsqueeze(0) if k_del.ndim == input_shape else k_del.reshape(input_shape))
        k_conf = float(k_out[0, pred_class]) if k_out.ndim == 2 else float(k_out.ravel()[0])

        # Do NOT clip negative drops (keep true drop)
        confidence_drop = float(orig_conf - k_conf)

        return {
            "faithfulness_status": "success",
            "faithfulness_score": confidence_drop,
            "original_confidence": orig_conf,
            "ablated_confidence": k_conf,
            "confidence_drop": confidence_drop,
            "deletion_auc": deletion_auc,
            "insertion_auc": insertion_auc,
            "random_deletion_auc": random_deletion_auc,
            "k_used": k,
        }
    except Exception as e:
        return {
            "faithfulness_status": f"error: {str(e)}",
            "faithfulness_score": None,
            "deletion_auc": None,
            "insertion_auc": None,
            "original_confidence": None,
            "ablated_confidence": None,
        }
