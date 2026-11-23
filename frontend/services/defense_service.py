"""
Defense and Hardening Service for AdverScan Frontend.

Bridges the Streamlit UI to Module 7 (Hardening Engine) and Module 8 (Re-Test),
providing domain-aware defense cataloging, iterative candidate evaluation,
robustness telemetry, and prediction restoration comparisons.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.hardening.hardening_engine import HardeningEngine, HardeningResult
from app.hardening.defenses import DEFENSE_REGISTRY, get_defense_class
from app.retest.threshold import RetestThresholds


def _extract_model_logits(outputs: Any) -> Any:
    """Extract classification logits tensor from model outputs across frameworks and output types."""
    if hasattr(outputs, "logits") and outputs.logits is not None:
        return outputs.logits
    if hasattr(outputs, "prediction_logits") and outputs.prediction_logits is not None:
        return outputs.prediction_logits
    if hasattr(outputs, "prediction_outputs") and outputs.prediction_outputs is not None:
        return outputs.prediction_outputs
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        return outputs[0]
    if hasattr(outputs, "get") and callable(getattr(outputs, "get")):
        for k in ("logits", "prediction_logits", "prediction_outputs"):
            val = outputs.get(k)
            if val is not None:
                return val
    return outputs


# Comprehensive defense catalog per domain with interactive metadata
DOMAIN_DEFENSE_CATALOG: Dict[str, List[Dict[str, Any]]] = {
    "image": [
        {
            "id": "auto",
            "name": "Auto (Iterative Defense Selection)",
            "family": "Iterative Auto-Tuning",
            "description": "Automatically evaluates ranked candidate defenses, testing each candidate against acceptance thresholds with fallback to the most effective defense.",
            "is_auto": True,
            "default_params": {},
        },
        {
            "id": "spatial_smoothing",
            "name": "Spatial Smoothing (Median Filtering)",
            "family": "Preprocessing",
            "description": "Applies a local spatial window filter across input pixels to suppress high-frequency adversarial perturbations while preserving macro image semantics.",
            "is_auto": False,
            "default_params": {"kernel_size": 3},
            "param_specs": [
                {
                    "name": "kernel_size",
                    "label": "Kernel Window Size",
                    "type": "select",
                    "options": [3, 5, 7],
                    "default": 3,
                    "help": "Dimension of median smoothing kernel window (odd integer).",
                }
            ],
        },
        {
            "id": "randomized_smoothing",
            "name": "Randomized Smoothing",
            "family": "Smoothing",
            "description": "Injects isotropic Gaussian noise into the input and aggregates stochastic predictions to certify provable robust classification radii.",
            "is_auto": False,
            "default_params": {"noise_std": 0.08, "num_samples": 10},
            "param_specs": [
                {
                    "name": "noise_std",
                    "label": "Gaussian Noise Std Dev (σ)",
                    "type": "slider",
                    "min_value": 0.01,
                    "max_value": 0.30,
                    "step": 0.01,
                    "default": 0.08,
                    "help": "Standard deviation of isotropic Gaussian noise added during inference.",
                },
                {
                    "name": "num_samples",
                    "label": "Monte Carlo Sample Count",
                    "type": "slider",
                    "min_value": 5,
                    "max_value": 30,
                    "step": 5,
                    "default": 10,
                    "help": "Number of stochastic evaluations per input to form the certified consensus.",
                },
            ],
        },
        {
            "id": "feature_squeezing",
            "name": "Feature Squeezing (Color Depth Reduction)",
            "family": "Preprocessing",
            "description": "Reduces the precision of pixel color representation (e.g., 8-bit to 4-bit) to eliminate subtle adversarial perturbations in low-order bits.",
            "is_auto": False,
            "default_params": {"bit_depth": 4},
            "param_specs": [
                {
                    "name": "bit_depth",
                    "label": "Quantization Bit Depth",
                    "type": "select",
                    "options": [2, 3, 4, 5, 6],
                    "default": 4,
                    "help": "Number of bits per channel used to represent pixel values.",
                }
            ],
        },
        {
            "id": "gaussian_filter",
            "name": "Gaussian Blur Filter",
            "family": "Preprocessing",
            "description": "Convolves input image with a 2D Gaussian blur kernel to attenuate high-frequency gradient noise.",
            "is_auto": False,
            "default_params": {"kernel_size": 3, "sigma": 1.0},
            "param_specs": [
                {
                    "name": "kernel_size",
                    "label": "Kernel Size",
                    "type": "select",
                    "options": [3, 5],
                    "default": 3,
                },
                {
                    "name": "sigma",
                    "label": "Gaussian Sigma (σ)",
                    "type": "slider",
                    "min_value": 0.5,
                    "max_value": 2.5,
                    "step": 0.25,
                    "default": 1.0,
                },
            ],
        },
        {
            "id": "image_denoising",
            "name": "Image Denoising",
            "family": "Preprocessing",
            "description": "Non-local or adaptive bilateral filtering designed to strip adversarial noise patterns.",
            "is_auto": False,
            "default_params": {"strength": 0.1},
        },
        {
            "id": "jpeg_compression",
            "name": "JPEG Compression Defense",
            "family": "Transformation",
            "description": "Applies lossy Discrete Cosine Transform compression to disrupt pixel-level adversarial gradient alignment.",
            "is_auto": False,
            "default_params": {"quality": 75},
            "param_specs": [
                {
                    "name": "quality",
                    "label": "Compression Quality Factor",
                    "type": "slider",
                    "min_value": 50,
                    "max_value": 90,
                    "step": 5,
                    "default": 75,
                }
            ],
        },
    ],
    "tabular": [
        {
            "id": "auto",
            "name": "Auto (Iterative Defense Selection)",
            "family": "Iterative Auto-Tuning",
            "description": "Automatically evaluates ranked candidate defenses for tabular data with fallback to best.",
            "is_auto": True,
            "default_params": {},
        },
        {
            "id": "feature_squeezing",
            "name": "Feature Squeezing (Quantization)",
            "family": "Preprocessing",
            "description": "Discretizes continuous tabular features into equal-width bins to dampen boundary shifts.",
            "is_auto": False,
            "default_params": {"bit_depth": 4},
        },
        {
            "id": "randomized_smoothing",
            "name": "Randomized Smoothing",
            "family": "Smoothing",
            "description": "Injects Gaussian noise into numerical feature columns to create robust neighborhood consensus.",
            "is_auto": False,
            "default_params": {"noise_std": 0.05, "num_samples": 10},
        },
        {
            "id": "normalization",
            "name": "Feature Normalization & Clipping",
            "family": "Preprocessing",
            "description": "Standardizes numerical columns and clips out-of-distribution values to realistic boundaries.",
            "is_auto": False,
            "default_params": {},
        },
    ],
    "text": [
        {
            "id": "auto",
            "name": "Auto (Iterative Defense Selection)",
            "family": "Iterative Auto-Tuning",
            "description": "Automatically evaluates candidate defenses for NLP classification.",
            "is_auto": True,
            "default_params": {},
        },
        {
            "id": "confidence_rejection",
            "name": "Confidence Rejection Filter",
            "family": "Detection",
            "description": "Flags and rejects ambiguous inputs whose prediction entropy or top margin falls below a high-confidence threshold.",
            "is_auto": False,
            "default_params": {"threshold": 0.55},
            "param_specs": [
                {
                    "name": "threshold",
                    "label": "Confidence Threshold",
                    "type": "slider",
                    "min_value": 0.1,
                    "max_value": 0.95,
                    "default": 0.55,
                    "step": 0.05,
                    "help": "Prediction confidence below which inputs are rejected as ambiguous.",
                }
            ],
        },
        {
            "id": "adversarial_detection",
            "name": "Adversarial Detection Wrapper",
            "family": "Detection",
            "description": "Screens input representations for out-of-manifold perturbations before model classification.",
            "is_auto": False,
            "default_params": {"threshold": 0.5, "method": "margin"},
            "param_specs": [
                {
                    "name": "method",
                    "label": "Detection Method",
                    "type": "select",
                    "options": ["margin", "entropy"],
                    "default": "margin",
                    "help": "Anomaly score heuristic for text representations.",
                },
                {
                    "name": "threshold",
                    "label": "Anomaly Score Threshold",
                    "type": "slider",
                    "min_value": 0.1,
                    "max_value": 0.9,
                    "default": 0.5,
                    "step": 0.05,
                    "help": "Threshold above which inputs are flagged and rejected as adversarial.",
                },
            ],
        },
    ],
    "time-series": [
        {
            "id": "auto",
            "name": "Auto (Iterative Defense Selection)",
            "family": "Iterative Auto-Tuning",
            "description": "Automatically tests candidate defenses for temporal sequence data.",
            "is_auto": True,
            "default_params": {},
        },
        {
            "id": "gaussian_filter",
            "name": "Temporal Gaussian Smoothing",
            "family": "Preprocessing",
            "description": "Smooths temporal fluctuations along time-series steps to isolate true signals.",
            "is_auto": False,
            "default_params": {"kernel_size": 3, "sigma": 1.0},
        },
        {
            "id": "randomized_smoothing",
            "name": "Randomized Smoothing",
            "family": "Smoothing",
            "description": "Adds stochastic temporal noise across sequence frames for robust consensus.",
            "is_auto": False,
            "default_params": {"noise_std": 0.05, "num_samples": 10},
        },
    ],
}


def get_available_defenses_for_domain(domain: str) -> List[Dict[str, Any]]:
    """Return available defense options for the specified modality/domain."""
    clean_domain = str(domain).lower().strip()
    if clean_domain in ("nlp", "language"):
        clean_domain = "text"
    return DOMAIN_DEFENSE_CATALOG.get(clean_domain, DOMAIN_DEFENSE_CATALOG["image"])


def execute_configured_defense(
    model: Any,
    domain: str,
    defense_choice: str,
    test_inputs: Any,
    test_labels: Any,
    adversarial_examples: Optional[Any],
    attack_result: Dict[str, Any],
    fallback_to_best: bool = True,
    custom_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute defense hardening using HardeningEngine and calculate empirical
    robustness metrics before vs. after defense.

    Returns:
        Structured result dict matching AdverScan standards, with full telemetry,
        candidate attempt details, restored prediction tables, and hardened model.
    """
    raw_model = getattr(model, "get_model", lambda: model)()
    if not (isinstance(raw_model, nn.Module) or hasattr(raw_model, "predict") or callable(raw_model)):
        raise TypeError(f"Target model must be a PyTorch nn.Module or Scikit-learn estimator, got {type(raw_model)}")

    # Ensure tensors are prepared
    labels_tensor = (
        test_labels if isinstance(test_labels, torch.Tensor) else torch.as_tensor(test_labels)
    )
    total_samples = len(labels_tensor)

    # Resolve device
    try:
        device = next(raw_model.parameters()).device
    except Exception:
        device = torch.device("cpu")

    # Never substitute clean inputs for adversarial examples (A08)
    # Support both pure torch.Tensor and dictionary-based tensor batches (e.g. HuggingFace NLP BatchEncoding)
    if adversarial_examples is None:
        # Check if attack result dictionary preserved adversarial examples under other standard keys
        adversarial_examples = (
            attack_result.get("_adversarial_examples")
            or attack_result.get("adversarial_examples")
        )

    is_valid_adv = False
    if isinstance(adversarial_examples, torch.Tensor):
        is_valid_adv = True
    elif isinstance(adversarial_examples, dict) and any(isinstance(v, torch.Tensor) for v in adversarial_examples.values()):
        is_valid_adv = True

    if not is_valid_adv:
        raise ValueError(
            "Adversarial examples tensor is missing or invalid. "
            "Cannot perform defense evaluation without genuine adversarial examples from M3. "
            "Clean inputs must never be substituted for adversarial examples."
        )
    adv_tensor = adversarial_examples

    # Ensure consistent batch dimensions across clean inputs, labels, and adversarial examples
    adv_len = adv_tensor.size(0) if isinstance(adv_tensor, torch.Tensor) else len(next(v for v in adv_tensor.values() if hasattr(v, "__len__")))
    min_samples = min(adv_len, len(labels_tensor))
    if isinstance(test_inputs, torch.Tensor):
        min_samples = min(min_samples, test_inputs.size(0))
    elif isinstance(test_inputs, dict):
        first_in = next((v for v in test_inputs.values() if hasattr(v, "__len__")), None)
        if first_in is not None:
            min_samples = min(min_samples, len(first_in))

    if isinstance(adv_tensor, torch.Tensor):
        adv_tensor = adv_tensor[:min_samples]
    elif isinstance(adv_tensor, dict):
        adv_tensor = {k: (v[:min_samples] if hasattr(v, "__getitem__") else v) for k, v in adv_tensor.items()}

    labels_tensor = labels_tensor[:min_samples]
    if isinstance(test_inputs, torch.Tensor):
        test_inputs = test_inputs[:min_samples]
    elif isinstance(test_inputs, dict):
        test_inputs = {k: (v[:min_samples] if hasattr(v, "__getitem__") else v) for k, v in test_inputs.items()}
    total_samples = min_samples

    params = attack_result.get("parameters") or attack_result.get("parameters_used") or {}
    is_targeted = bool(params.get("targeted", False) or attack_result.get("is_targeted", False))
    target_label = params.get("target_label", attack_result.get("target_label"))
    if target_label is not None:
        try:
            target_label = int(target_label)
        except (ValueError, TypeError):
            pass

    attack_name = str(attack_result.get("attack_name", "fgsm")).lower()
    base_clean_acc = float(attack_result.get("clean_accuracy", 100.0)) / 100.0
    base_adv_acc = float(attack_result.get("adversarial_accuracy", 0.0)) / 100.0
    base_asr = float(attack_result.get("asr_percentage", 0.0)) / 100.0
    
    # Use authentic M5 vulnerability score if available (A11)
    if attack_result.get("vulnerability_score") is not None:
        base_vuln = float(attack_result["vulnerability_score"])
    else:
        deg_pts = max(0.0, (base_clean_acc - base_adv_acc) * 100.0)
        base_vuln = round(0.50 * (base_asr * 100.0) + 0.35 * deg_pts, 2)

    # Early short-circuit: if baseline ASR is 0.0%, model is already robust; defense is not needed
    if base_asr <= 0.0:
        return {
            "success": True,
            "status": "not_needed",
            "defense_name": "None (Already Robust)",
            "selected_defense": "None (Already Robust)",
            "hardened_model": raw_model,
            "defense_type": "None",
            "defense_family": "None",
            "defense_stage": "none",
            "is_improved": False,
            "improvement_sufficient": False,
            "fallback_used": False,
            "thresholds": {},
            "threshold_compliance": [
                {
                    "Criterion": "Attack Success Rate (ASR) Baseline Check",
                    "Baseline (Before)": "0.0%",
                    "Hardened (Re-Test)": "0.0%",
                    "Observed Change": "0.0% (Attack Ineffective)",
                    "Acceptance Bound": "ASR = 0.0%",
                    "Compliance Verdict": "PASSED (Model Already Robust - No Defense Required)",
                }
            ],
            "defense_effectiveness": "Not Needed (Model Already Robust)",
            "execution_time_seconds": 0.01,
            "total_samples": total_samples,
            "before_clean_accuracy": base_clean_acc * 100.0,
            "before_adv_accuracy": base_adv_acc * 100.0,
            "before_asr": 0.0,
            "before_vulnerability_score": base_vuln,
            "post_clean_accuracy": base_clean_acc * 100.0,
            "after_clean_accuracy": base_clean_acc * 100.0,
            "after_adv_accuracy": base_adv_acc * 100.0,
            "post_asr": 0.0,
            "post_asr_defined": True,
            "after_asr": 0.0,
            "after_vulnerability_score": base_vuln,
            "asr_reduction_percentage": 0.0,
            "vulnerability_improvement": 0.0,
            "clean_accuracy_drop": 0.0,
            "candidate_attempts": [],
            "sample_comparisons": [],
            "sample_restorations": [],
            "recommendations": [
                "Baseline Attack Success Rate is 0.0%. The model is already fully robust against this attack vector. "
                "Applying defense transformations is not recommended as it risks degrading clean accuracy without providing security benefit."
            ],
            "_hardened_model": raw_model,
        }

    base_risk = "HIGH" if base_asr >= 0.6 else ("MEDIUM" if base_asr >= 0.3 else "LOW")

    # Build evaluation closure for candidate models
    def eval_fn(candidate_model: Any) -> Dict[str, Any]:
        if hasattr(candidate_model, "eval"):
            candidate_model.eval()
        t0 = time.time()
        with torch.no_grad():
            # 1. Clean accuracy
            if isinstance(test_inputs, dict):
                inp_clean = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in test_inputs.items()}
                logits_clean = _extract_model_logits(candidate_model(**inp_clean))
            elif isinstance(test_inputs, torch.Tensor):
                inp_clean = test_inputs.to(device)
                logits_clean = _extract_model_logits(candidate_model(inp_clean))
            else:
                logits_clean = _extract_model_logits(candidate_model(test_inputs))

            if not isinstance(logits_clean, torch.Tensor):
                logits_clean = torch.as_tensor(logits_clean)
            c_preds = logits_clean.argmax(dim=-1).cpu()
            c_correct = int((c_preds == labels_tensor.cpu()).sum().item())
            cand_clean_acc = float(c_correct / total_samples) if total_samples > 0 else 0.0

            # 2. Defended accuracy on adversarial examples
            if isinstance(adv_tensor, dict):
                inp_adv = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in adv_tensor.items()}
                logits_adv = _extract_model_logits(candidate_model(**inp_adv))
            elif isinstance(adv_tensor, torch.Tensor):
                inp_adv = adv_tensor.to(device)
                logits_adv = _extract_model_logits(candidate_model(inp_adv))
            else:
                logits_adv = _extract_model_logits(candidate_model(adv_tensor))

            if not isinstance(logits_adv, torch.Tensor):
                logits_adv = torch.as_tensor(logits_adv)
            a_preds = logits_adv.argmax(dim=-1).cpu()
            a_correct = int((a_preds == labels_tensor.cpu()).sum().item())
            cand_adv_acc = float(a_correct / total_samples) if total_samples > 0 else 0.0

        latency_ms = (time.time() - t0) * 1000.0

        # Attack Success Rate under defense: proportion of initially-correct samples still misclassified or targeted (A06, A07)
        clean_right = (c_preds == labels_tensor.cpu())
        num_clean_right = int(clean_right.sum().item())
        if num_clean_right > 0:
            if is_targeted and target_label is not None:
                still_succ = clean_right & (a_preds == target_label)
            else:
                still_succ = clean_right & (a_preds != labels_tensor.cpu())
            cand_asr = float(still_succ.sum().item() / num_clean_right)
        else:
            cand_asr = 0.0

        # Candidate vulnerability score from M5 weighting
        cand_deg = max(0.0, (cand_clean_acc - cand_adv_acc) * 100.0)
        cand_vuln = round(0.50 * (cand_asr * 100.0) + 0.35 * cand_deg, 2)

        return {
            "clean_accuracy": cand_clean_acc,
            "adversarial_accuracy": cand_adv_acc,
            "asr": cand_asr,
            "vulnerability_score": cand_vuln,
            "latency_ms": latency_ms,
        }

    # Baseline metrics dict
    base_metrics = {
        "clean_accuracy": base_clean_acc,
        "adversarial_accuracy": base_adv_acc,
        "asr": base_asr,
        "vulnerability_score": base_vuln,
        "latency_ms": 10.0,
    }

    # Setup HardeningEngine
    engine = HardeningEngine()
    is_auto = defense_choice.lower().strip() in ("auto", "iterative")
    candidate_defenses = None if is_auto else [defense_choice]
    candidate_params = {defense_choice: (custom_params or {})} if not is_auto else {}

    t_start = time.time()
    hard_res: HardeningResult = engine.harden(
        model=raw_model,
        defense=defense_choice,
        inputs=test_inputs,
        labels=labels_tensor,
        attack_name=attack_name,
        risk_level=base_risk,
        vulnerability_score=base_vuln,
        input_domain=(
            "image" if domain in ("image", "vision")
            else ("tabular" if domain in ("tabular", "csv")
            else ("time_series" if domain in ("time-series", "timeseries", "sequence")
            else "nlp"))
        ),
        eval_fn=eval_fn,
        baseline_metrics=base_metrics,
        candidate_defenses=candidate_defenses,
        candidate_params=candidate_params,
        iterative=is_auto,
        fallback_to_best=fallback_to_best,
    )
    total_exec_time = time.time() - t_start

    # Retrieve defended model (or original if hardening failed)
    defended_model = hard_res.hardened_model or raw_model

    # Run complete post-defense telemetry with confidences
    if hasattr(defended_model, "eval"):
        defended_model.eval()
    with torch.no_grad():
        if isinstance(test_inputs, dict):
            inp_c = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in test_inputs.items()}
            logits_c = _extract_model_logits(defended_model(**inp_c))
        elif isinstance(test_inputs, torch.Tensor):
            inp_c = test_inputs.to(device)
            logits_c = _extract_model_logits(defended_model(inp_c))
        else:
            logits_c = _extract_model_logits(defended_model(test_inputs))

        if not isinstance(logits_c, torch.Tensor):
            logits_c = torch.as_tensor(logits_c)
        probs_c = F.softmax(logits_c, dim=-1)
        def_clean_preds = probs_c.argmax(dim=-1).cpu()
        def_clean_confs = probs_c.max(dim=-1).values.cpu()

        if isinstance(adv_tensor, dict):
            inp_a = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in adv_tensor.items()}
            logits_a = _extract_model_logits(defended_model(**inp_a))
        elif isinstance(adv_tensor, torch.Tensor):
            inp_a = adv_tensor.to(device)
            logits_a = _extract_model_logits(defended_model(inp_a))
        else:
            logits_a = _extract_model_logits(defended_model(adv_tensor))

        if not isinstance(logits_a, torch.Tensor):
            logits_a = torch.as_tensor(logits_a)
        probs_a = F.softmax(logits_a, dim=-1)
        def_adv_preds = probs_a.argmax(dim=-1).cpu()
        def_adv_confs = probs_a.max(dim=-1).values.cpu()

    # Metrics computation
    def_clean_correct = int((def_clean_preds == labels_tensor.cpu()).sum().item())
    def_adv_correct = int((def_adv_preds == labels_tensor.cpu()).sum().item())

    post_clean_acc = (def_clean_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    post_adv_acc = (def_adv_correct / total_samples) * 100.0 if total_samples > 0 else 0.0

    # Calculate post_asr using clean_correct as denominator (A06, A07)
    clean_right_post = (def_clean_preds == labels_tensor.cpu())
    num_clean_right_post = int(clean_right_post.sum().item())
    clean_acc_drop = max(0.0, (base_clean_acc * 100.0) - post_clean_acc)

    if num_clean_right_post > 0:
        if is_targeted and target_label is not None:
            post_still_succ = clean_right_post & (def_adv_preds == target_label)
        else:
            post_still_succ = clean_right_post & (def_adv_preds != labels_tensor.cpu())
        post_asr = float(post_still_succ.sum().item() / num_clean_right_post) * 100.0
        asr_reduction_pct = max(0.0, (base_asr * 100.0) - post_asr)
        post_deg = max(0.0, (post_clean_acc - post_adv_acc))
        post_vuln = round(0.50 * post_asr + 0.35 * post_deg, 2)
        vuln_improvement_pts = max(0.0, base_vuln - post_vuln)
    else:
        post_asr = None
        asr_reduction_pct = 0.0
        post_deg = 0.0
        post_vuln = 100.0
        vuln_improvement_pts = 0.0

    # Extract sample comparisons
    clean_orig_preds = attack_result.get("original_predictions")
    adv_orig_preds = attack_result.get("adversarial_predictions")
    sample_comps: List[Dict[str, Any]] = []

    # Fall back safely if raw predictions are not tensors
    labels_list = labels_tensor.tolist()
    for idx in range(total_samples):
        y_true = labels_list[idx]
        y_orig = clean_orig_preds[idx] if clean_orig_preds is not None and idx < len(clean_orig_preds) else y_true
        y_adv = adv_orig_preds[idx] if adv_orig_preds is not None and idx < len(adv_orig_preds) else "N/A"
        y_def = int(def_adv_preds[idx].item())
        conf_def = float(def_adv_confs[idx].item() * 100.0)

        # Classify restoration outcome
        if y_adv != "N/A" and str(y_adv) != str(y_orig):
            if y_def == y_true:
                outcome = "Restored Correct ✅"
            elif y_def == y_orig:
                outcome = "Restored Clean ✅"
            else:
                outcome = "Remains Flipped ❌"
        else:
            if y_def == y_true:
                outcome = "Robust / Defended 🛡️"
            else:
                outcome = "Base Error ⚠️"

        sample_comps.append(
            {
                "Sample": f"#{idx + 1}",
                "Ground Truth": str(y_true),
                "Clean Pred": str(y_orig),
                "Adversarial Pred": str(y_adv),
                "Defended Pred": f"{y_def} ({conf_def:.1f}%)",
                "Outcome": outcome,
            }
        )

    # Format candidate attempt records
    attempt_records: List[Dict[str, Any]] = []
    if hard_res.defense_attempts:
        for att in hard_res.defense_attempts:
            asr_red = att.asr_reduction
            asr_red_disp = f"{asr_red * 100:.1f}%" if asr_red is not None else "N/A"
            v_imp = att.vuln_score_improvement
            v_imp_disp = f"{v_imp:+.2f} pts" if v_imp is not None else "N/A"
            c_drop = att.clean_accuracy_drop_pct_points
            c_drop_disp = f"{c_drop:.2f}%" if c_drop is not None else "0.00%"

            attempt_records.append(
                {
                    "Defense": str(att.defense_name).replace("_", " ").title(),
                    "Status": str(att.status).upper(),
                    "ASR Reduction": asr_red_disp,
                    "Vuln Improvement": v_imp_disp,
                    "Clean Acc Drop": c_drop_disp,
                    "Details": att.reason or "Evaluated against thresholds",
                    "Is Selected": att.defense_name == hard_res.selected_defense,
                }
            )

    if hard_res.status == "not_needed":
        selected_defense_name = "None (Already Robust)"
        defense_eff = "Not Needed (Model Already Robust)"
    elif hard_res.selected_defense:
        selected_defense_name = hard_res.selected_defense
    elif is_auto:
        selected_defense_name = "None (No Defense Accepted)"
        defense_eff = "Ineffective / No Defense Accepted"
    else:
        selected_defense_name = defense_choice
    defense_meta = hard_res.metadata

    # Threshold compliance breakdown
    thresh_dict = hard_res.thresholds or {}
    max_clean_drop = float(thresh_dict.get("max_clean_accuracy_drop", 0.02)) * 100.0
    min_asr_red = float(thresh_dict.get("min_asr_reduction", 0.10)) * 100.0
    min_vuln_imp = float(thresh_dict.get("min_vuln_score_improvement", 5.0))

    clean_drop_passed = clean_acc_drop <= max_clean_drop
    asr_red_passed = asr_reduction_pct >= min_asr_red and post_asr is not None
    vuln_imp_passed = vuln_improvement_pts >= min_vuln_imp

    if post_asr is None:
        defense_eff = "Inconclusive (Zero Clean Accuracy)"
    elif clean_acc_drop > (float(thresh_dict.get("severe_clean_drop_threshold", 0.10)) * 100.0):
        defense_eff = "Harmful (Severe Clean Accuracy Degradation)"
    elif asr_reduction_pct >= 50.0 and clean_drop_passed:
        defense_eff = "High"
    elif asr_red_passed:
        defense_eff = "Moderate"
    elif asr_reduction_pct > 0.0:
        defense_eff = "Low"
    else:
        defense_eff = "Ineffective"

    post_asr_str = f"{post_asr:.1f}%" if post_asr is not None else "Undefined (Zero Clean Acc)"

    threshold_compliance = [
        {
            "Criterion": "Clean Accuracy Preservation",
            "Baseline (Before)": f"{base_clean_acc * 100.0:.1f}%",
            "Hardened (Re-Test)": f"{post_clean_acc:.1f}%",
            "Observed Change": f"-{clean_acc_drop:.1f}% drop",
            "Acceptance Bound": f"Drop ≤ {max_clean_drop:.1f}%",
            "Compliance Verdict": "PASSED (Within Threshold)" if clean_drop_passed else "BEYOND THRESHOLD (Exceeded Drop)",
        },
        {
            "Criterion": "Attack Success Rate (ASR) Reduction",
            "Baseline (Before)": f"{base_asr * 100.0:.1f}%",
            "Hardened (Re-Test)": post_asr_str,
            "Observed Change": f"-{asr_reduction_pct:.1f}% drop",
            "Acceptance Bound": f"Reduction ≥ {min_asr_red:.1f}%",
            "Compliance Verdict": "PASSED (Within Threshold)" if asr_red_passed else ("INCONCLUSIVE" if post_asr is None else "BEYOND THRESHOLD (Insufficient Reduction)"),
        },
        {
            "Criterion": "Vulnerability Score Improvement",
            "Baseline (Before)": f"{base_vuln:.1f} / 100",
            "Hardened (Re-Test)": f"{post_vuln:.1f} / 100",
            "Observed Change": f"+{vuln_improvement_pts:.1f} pts gain",
            "Acceptance Bound": f"Gain ≥ {min_vuln_imp:.1f} pts",
            "Compliance Verdict": "PASSED (Within Threshold)" if vuln_imp_passed else "BEYOND THRESHOLD (Insufficient Gain)",
        },
    ]

    return {
        "success": hard_res.success,
        "status": hard_res.status,
        "defense_name": selected_defense_name,
        "selected_defense": selected_defense_name,
        "hardened_model": defended_model,
        "defense_type": getattr(defense_meta, "defense_type", "Preprocessing"),
        "is_improved": hard_res.is_improved,
        "improvement_sufficient": hard_res.improvement_sufficient,
        "fallback_used": (not hard_res.improvement_sufficient) and (hard_res.selected_defense is not None),
        "thresholds": thresh_dict,
        "threshold_compliance": threshold_compliance,
        "defense_effectiveness": defense_eff,
        "execution_time_seconds": total_exec_time,
        "total_samples": total_samples,
        # Before metrics
        "before_clean_accuracy": base_clean_acc * 100.0,
        "before_adv_accuracy": base_adv_acc * 100.0,
        "before_asr": base_asr * 100.0,
        "before_vulnerability_score": base_vuln,
        # After metrics
        "post_clean_accuracy": post_clean_acc,
        "after_clean_accuracy": post_clean_acc,
        "after_adv_accuracy": post_adv_acc,
        "post_asr": post_asr if post_asr is not None else 0.0,
        "post_asr_defined": post_asr is not None,
        "after_asr": post_asr if post_asr is not None else 0.0,
        "after_vulnerability_score": post_vuln,
        # Deltas
        "asr_reduction_percentage": asr_reduction_pct,
        "vulnerability_improvement": vuln_improvement_pts,
        "clean_accuracy_drop": clean_acc_drop,
        # Artifacts
        "candidate_attempts": attempt_records,
        "sample_comparisons": sample_comps,
        "sample_restorations": sample_comps,
        "recommendations": hard_res.recommendations or [],
        # In-memory handles
        "_hardened_model": defended_model,
        "_raw_hardening_result": hard_res,
    }
