"""
Explanation faithfulness metric for AdverScan XAI.
Tests whether masking or removing important features meaningfully affects model predictions.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import torch

from app.explainability.input_utils import safe_predict, to_numpy, to_tensor


def compute_faithfulness(
    model: Any,
    inputs: Any,
    attribution: Any,
    top_k: int = 5,
    baseline_value: float = 0.0,
) -> Dict[str, Any]:
    """
    Evaluate explanation faithfulness by feature ablation (comprehensiveness).

    Ablating the most important features should significantly reduce model confidence
    for the predicted class if the explanation is faithful.

    Args:
        model: Target ML model.
        inputs: Input sample (tensor or array).
        attribution: Attribution values corresponding to inputs.
        top_k: Number of top features to mask out.
        baseline_value: Fill value for masked features.

    Returns:
        Dictionary with original confidence, ablated confidence, and faithfulness score.
    """
    if inputs is None or attribution is None or model is None:
        return {
            "faithfulness_status": "unavailable",
            "faithfulness_score": None,
            "original_confidence": None,
            "ablated_confidence": None,
        }

    try:
        t_input = to_tensor(inputs).clone()
        attr_arr = to_numpy(attribution, dtype=np.float32).ravel()

        orig_outputs = safe_predict(model, t_input)
        if orig_outputs.ndim >= 2:
            orig_probs = torch.softmax(orig_outputs, dim=-1)
            pred_class = int(torch.argmax(orig_probs, dim=-1)[0].item())
            orig_conf = float(orig_probs[0, pred_class].item())
        elif orig_outputs.ndim == 1:
            orig_probs = torch.softmax(orig_outputs, dim=0)
            pred_class = int(torch.argmax(orig_probs, dim=0).item())
            orig_conf = float(orig_probs[pred_class].item())
        else:
            pred_class = 0
            orig_conf = 1.0

        flat_input = t_input.view(-1).clone()
        k = min(top_k, len(flat_input), len(attr_arr))

        if k == 0:
            return {
                "faithfulness_status": "empty_input",
                "faithfulness_score": 0.0,
                "original_confidence": orig_conf,
                "ablated_confidence": orig_conf,
            }

        top_indices = np.argsort(np.abs(attr_arr[:len(flat_input)]))[::-1][:k]
        for idx in top_indices:
            flat_input[int(idx)] = baseline_value

        ablated_input = flat_input.view_as(t_input)
        ablated_outputs = safe_predict(model, ablated_input)

        if ablated_outputs.ndim >= 2:
            ablated_probs = torch.softmax(ablated_outputs, dim=-1)
            ablated_conf = float(ablated_probs[0, pred_class].item())
        elif ablated_outputs.ndim == 1:
            ablated_probs = torch.softmax(ablated_outputs, dim=0)
            ablated_conf = float(ablated_probs[pred_class].item())
        else:
            ablated_conf = 1.0

        # Faithfulness score: relative drop in confidence on target class
        faithfulness_score = float(max(0.0, orig_conf - ablated_conf))

        return {
            "faithfulness_status": "success",
            "faithfulness_score": faithfulness_score,
            "original_confidence": orig_conf,
            "ablated_confidence": ablated_conf,
            "confidence_drop": float(orig_conf - ablated_conf),
        }
    except Exception as e:
        return {
            "faithfulness_status": f"error: {str(e)}",
            "faithfulness_score": None,
            "original_confidence": None,
            "ablated_confidence": None,
        }
