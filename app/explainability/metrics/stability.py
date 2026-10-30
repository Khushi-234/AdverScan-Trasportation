"""
Explanation stability metric for AdverScan XAI.
Checks whether explanations remain reasonably consistent when inputs change slightly,
and separates explainer sampling variance from model input sensitivity.
"""

from typing import Any, Dict, Optional
import numpy as np
import torch

from app.explainability.utils import predict_proba_np, to_numpy, to_tensor


def compute_stability(
    explainer: Any,
    model: Any,
    inputs: Any,
    target_class: Optional[int] = None,
    noise_std: float = 0.01,
    num_trials: int = 3,
    background_inputs: Optional[Any] = None,
    seed: int = 42,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Measure local stability of explanation against minor input perturbations.

    Separates:
    1. Model input sensitivity (response under input noise) with FIXED target class.
    2. Explainer sampling variance (repeated explanations on identical input without noise).

    Args:
        explainer: BaseExplainer instance.
        model: Target ML model.
        inputs: Input sample (tensor or array).
        target_class: Fixed target class to explain across all noisy trials.
        noise_std: Relative standard deviation of additive Gaussian noise.
        num_trials: Number of noisy trials to average over.
        background_inputs: Optional background dataset to compute per-feature std.
        seed: Random seed for reproducibility.
        **kwargs: Extra parameters passed to explainer.explain.

    Returns:
        Dictionary containing stability status, average similarity, drift, and sampling variance.
    """
    if explainer is None or model is None or inputs is None:
        return {
            "stability_status": "unavailable",
            "mean_cosine_stability": None,
            "mean_l2_drift": None,
            "sampling_variance": None,
        }

    try:
        rng = np.random.RandomState(seed)

        # 1. Infer fixed target_class if not provided
        fixed_target = target_class
        if fixed_target is None:
            probs = predict_proba_np(model, inputs)
            fixed_target = int(np.argmax(probs[0])) if probs.ndim == 2 else 0

        # 2. Base explanation with fixed target_class
        base_exp = explainer.explain(model, inputs, target_class=fixed_target, **kwargs)
        base_attr = base_exp.get("attribution")
        if base_attr is None:
            return {
                "stability_status": "base_attribution_failed",
                "mean_cosine_stability": None,
                "mean_l2_drift": None,
                "sampling_variance": None,
            }

        base_vec = to_numpy(base_attr, dtype=np.float64).ravel()
        norm_base = np.linalg.norm(base_vec)

        # 3. Explainer sampling variance: repeat explanation on identical input
        # to quantify stochasticity in LIME/KernelSHAP
        sample_repeat_exps = []
        for _ in range(min(num_trials, 3)):
            rep = explainer.explain(model, inputs, target_class=fixed_target, **kwargs)
            rep_attr = rep.get("attribution")
            if rep_attr is not None:
                sample_repeat_exps.append(to_numpy(rep_attr, dtype=np.float64).ravel())

        if len(sample_repeat_exps) > 1:
            sampling_variance = float(np.mean(np.var(np.vstack(sample_repeat_exps), axis=0)))
        else:
            sampling_variance = 0.0

        # 4. Input sensitivity: scale noise by per-feature std
        t_input = to_tensor(inputs)
        inp_arr = to_numpy(inputs, dtype=np.float64)

        if background_inputs is not None:
            bg_arr = to_numpy(background_inputs, dtype=np.float64)
            feature_std = np.std(bg_arr, axis=0)
            feature_std = np.maximum(feature_std, 1e-4)
        else:
            feature_std = np.maximum(np.abs(inp_arr) * 0.1, 0.05)

        noise_scale = to_tensor(feature_std * noise_std, device=t_input.device)

        sims = []
        drifts = []

        for _ in range(num_trials):
            # Generate reproducible noise
            noise_np = rng.normal(0, 1, size=inp_arr.shape)
            noise_t = to_tensor(noise_np, device=t_input.device) * noise_scale
            noisy_input = t_input + noise_t

            noisy_exp = explainer.explain(model, noisy_input, target_class=fixed_target, **kwargs)
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
                "sampling_variance": sampling_variance,
            }

        return {
            "stability_status": "success",
            "mean_cosine_stability": float(np.mean(sims)),
            "mean_l2_drift": float(np.mean(drifts)),
            "sampling_variance": sampling_variance,
            "target_class": fixed_target,
            "noise_std": noise_std,
            "num_trials": len(sims),
        }
    except Exception as e:
        return {
            "stability_status": f"error: {str(e)}",
            "mean_cosine_stability": None,
            "mean_l2_drift": None,
            "sampling_variance": None,
        }
