"""
Explanation stability metric for AdverScan XAI.
Checks whether explanations remain reasonably consistent when inputs change slightly.
"""

from typing import Any, Dict, Optional
import numpy as np
import torch

from app.explainability.input_utils import to_numpy, to_tensor


def compute_stability(
    explainer: Any,
    model: Any,
    inputs: Any,
    noise_std: float = 0.01,
    num_trials: int = 3,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Measure local stability of explanation against minor input perturbations.

    Computes the average cosine similarity and relative distance between
    the baseline explanation and explanations under slight Gaussian noise.

    Args:
        explainer: BaseExplainer instance.
        model: Target ML model.
        inputs: Input sample (tensor or array).
        noise_std: Standard deviation of additive Gaussian noise.
        num_trials: Number of noisy trials to average over.
        **kwargs: Extra parameters passed to explainer.explain.

    Returns:
        Dictionary containing stability status, average similarity, and mean drift.
    """
    if explainer is None or model is None or inputs is None:
        return {
            "stability_status": "unavailable",
            "mean_cosine_stability": None,
            "mean_l2_drift": None,
        }

    try:
        base_exp = explainer.explain(model, inputs, **kwargs)
        base_attr = base_exp.get("attribution")
        if base_attr is None:
            return {
                "stability_status": "base_attribution_failed",
                "mean_cosine_stability": None,
                "mean_l2_drift": None,
            }

        base_vec = to_numpy(base_attr, dtype=np.float64).ravel()
        norm_base = np.linalg.norm(base_vec)

        t_input = to_tensor(inputs)
        sims = []
        drifts = []

        for _ in range(num_trials):
            noise = torch.randn_like(t_input) * noise_std
            noisy_input = t_input + noise

            noisy_exp = explainer.explain(model, noisy_input, **kwargs)
            noisy_attr = noisy_exp.get("attribution")
            if noisy_attr is None:
                continue

            noisy_vec = to_numpy(noisy_attr, dtype=np.float64).ravel()
            min_len = min(len(base_vec), len(noisy_vec))
            b_v = base_vec[:min_len]
            n_v = noisy_vec[:min_len]

            norm_noisy = np.linalg.norm(n_v)
            norm_b = np.linalg.norm(b_v)

            if norm_b > 1e-12 and norm_noisy > 1e-12:
                cos_sim = float(np.dot(b_v, n_v) / (norm_b * norm_noisy))
            else:
                cos_sim = 1.0 if (norm_b <= 1e-12 and norm_noisy <= 1e-12) else 0.0

            l2_diff = float(np.linalg.norm(b_v - n_v))

            sims.append(cos_sim)
            drifts.append(l2_diff)

        if not sims:
            return {
                "stability_status": "noisy_trials_failed",
                "mean_cosine_stability": None,
                "mean_l2_drift": None,
            }

        return {
            "stability_status": "success",
            "mean_cosine_stability": float(np.mean(sims)),
            "mean_l2_drift": float(np.mean(drifts)),
            "noise_std": noise_std,
            "num_trials": len(sims),
        }
    except Exception as e:
        return {
            "stability_status": f"error: {str(e)}",
            "mean_cosine_stability": None,
            "mean_l2_drift": None,
        }
