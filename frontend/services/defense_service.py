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
        },
        {
            "id": "adversarial_detection",
            "name": "Adversarial Detection Wrapper",
            "family": "Detection",
            "description": "Screens input representations for out-of-manifold perturbations before model classification.",
            "is_auto": False,
            "default_params": {},
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
    if not isinstance(raw_model, nn.Module):
        raise TypeError(f"Target model must be a PyTorch nn.Module, got {type(raw_model)}")

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

    # If adversarial examples were not preserved, fall back to clean inputs
    adv_tensor = adversarial_examples
    if adv_tensor is None or not isinstance(adv_tensor, torch.Tensor):
        adv_tensor = test_inputs

    attack_name = str(attack_result.get("attack_name", "fgsm")).lower()
    base_clean_acc = float(attack_result.get("clean_accuracy", 100.0)) / 100.0
    base_adv_acc = float(attack_result.get("adversarial_accuracy", 0.0)) / 100.0
    base_asr = float(attack_result.get("asr_percentage", 0.0)) / 100.0
    base_vuln = round(base_asr * 100.0, 2)
    base_risk = "HIGH" if base_asr >= 0.6 else ("MEDIUM" if base_asr >= 0.3 else "LOW")

    # Build evaluation closure for candidate models
    def eval_fn(candidate_model: nn.Module) -> Dict[str, Any]:
        candidate_model.eval()
        t0 = time.time()
        with torch.no_grad():
            # 1. Clean accuracy
            logits_clean = candidate_model(test_inputs.to(device))
            if hasattr(logits_clean, "logits"):
                logits_clean = logits_clean.logits
            c_preds = logits_clean.argmax(dim=-1).cpu()
            c_correct = int((c_preds == labels_tensor.cpu()).sum().item())
            cand_clean_acc = float(c_correct / total_samples) if total_samples > 0 else 0.0

            # 2. Defended accuracy on adversarial examples
            logits_adv = candidate_model(adv_tensor.to(device))
            if hasattr(logits_adv, "logits"):
                logits_adv = logits_adv.logits
            a_preds = logits_adv.argmax(dim=-1).cpu()
            a_correct = int((a_preds == labels_tensor.cpu()).sum().item())
            cand_adv_acc = float(a_correct / total_samples) if total_samples > 0 else 0.0

        latency_ms = (time.time() - t0) * 1000.0

        # Attack Success Rate under defense: proportion of samples still misclassified or flipped
        cand_asr = max(0.0, float(1.0 - cand_adv_acc))
        cand_vuln = round(cand_asr * 100.0, 2)

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
        input_domain="image" if domain in ("image", "vision") else ("tabular" if domain == "tabular" else "nlp"),
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
    defended_model.eval()
    with torch.no_grad():
        logits_c = defended_model(test_inputs.to(device))
        if hasattr(logits_c, "logits"):
            logits_c = logits_c.logits
        probs_c = F.softmax(logits_c, dim=-1)
        def_clean_preds = probs_c.argmax(dim=-1).cpu()
        def_clean_confs = probs_c.max(dim=-1).values.cpu()

        logits_a = defended_model(adv_tensor.to(device))
        if hasattr(logits_a, "logits"):
            logits_a = logits_a.logits
        probs_a = F.softmax(logits_a, dim=-1)
        def_adv_preds = probs_a.argmax(dim=-1).cpu()
        def_adv_confs = probs_a.max(dim=-1).values.cpu()

    # Metrics computation
    def_clean_correct = int((def_clean_preds == labels_tensor.cpu()).sum().item())
    def_adv_correct = int((def_adv_preds == labels_tensor.cpu()).sum().item())

    post_clean_acc = (def_clean_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    post_adv_acc = (def_adv_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    post_asr = max(0.0, 100.0 - post_adv_acc)
    post_vuln = post_asr

    asr_reduction_pct = max(0.0, (base_asr * 100.0) - post_asr)
    vuln_improvement_pts = max(0.0, base_vuln - post_vuln)
    clean_acc_drop = max(0.0, (base_clean_acc * 100.0) - post_clean_acc)

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

    selected_defense_name = hard_res.selected_defense or defense_choice
    defense_meta = hard_res.metadata

    return {
        "status": hard_res.status,
        "selected_defense": selected_defense_name,
        "defense_type": getattr(defense_meta, "defense_type", "Preprocessing"),
        "is_improved": hard_res.is_improved,
        "improvement_sufficient": hard_res.improvement_sufficient,
        "fallback_used": (not hard_res.improvement_sufficient) and (hard_res.selected_defense is not None),
        "execution_time_seconds": total_exec_time,
        "total_samples": total_samples,
        # Before metrics
        "before_clean_accuracy": base_clean_acc * 100.0,
        "before_adv_accuracy": base_adv_acc * 100.0,
        "before_asr": base_asr * 100.0,
        "before_vulnerability_score": base_vuln,
        # After metrics
        "after_clean_accuracy": post_clean_acc,
        "after_adv_accuracy": post_adv_acc,
        "after_asr": post_asr,
        "after_vulnerability_score": post_vuln,
        # Deltas
        "asr_reduction_percentage": asr_reduction_pct,
        "vulnerability_improvement": vuln_improvement_pts,
        "clean_accuracy_drop": clean_acc_drop,
        # Artifacts
        "candidate_attempts": attempt_records,
        "sample_restorations": sample_comps,
        "recommendations": hard_res.recommendations or [],
        # In-memory handles
        "_hardened_model": defended_model,
        "_raw_hardening_result": hard_res,
    }
