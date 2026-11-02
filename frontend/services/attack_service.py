"""
Attack Engine Service for AdverScan Frontend.

Bridges the Streamlit UI to the existing backend attack engine (Module 3)
without duplicating attack discovery or execution logic.
Provides canonical domain attack routing and rich telemetry matching
scripts/run_pipeline_for_multimodel.py.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F

from app.attack_engine.attack_discovery import discover_attacks
from app.attack_engine.attack_executor import execute_attack
from app.attack_engine.attack_registry import get_attack, list_attacks
from app.attack_engine.config import AttackConfig
from app.attack_engine.exceptions import (
    AttackConfigurationError,
    AttackError,
    AttackExecutionError,
    UnsupportedModelError,
)
from app.attack_engine.models import AttackMetadata, AttackResult
from app.attack_engine.utils.validation import check_model_compatibility

# Canonical catalog of attacks per domain with full descriptive metadata
CANONICAL_DOMAIN_CATALOG: Dict[str, List[Dict[str, str]]] = {
    "image": [
        {
            "id": "fgsm",
            "name": "FGSM",
            "full_name": "Fast Gradient Sign Method",
            "category": "Gradient-based (One-step)",
            "threat_model": "White-Box",
            "description": "Fast single-step perturbation along the sign of the loss gradient.",
        },
        {
            "id": "pgd",
            "name": "PGD",
            "full_name": "Projected Gradient Descent",
            "category": "Gradient-based (Iterative)",
            "threat_model": "White-Box",
            "description": "Multi-step iterative gradient ascent projected onto the epsilon ball.",
        },
        {
            "id": "deepfool",
            "name": "DeepFool",
            "full_name": "DeepFool Minimal Perturbation",
            "category": "Boundary-based",
            "threat_model": "White-Box",
            "description": "Finds closest decision boundary via iterative linearized hyperplane projections.",
        },
        {
            "id": "cw",
            "name": "Carlini & Wagner (CW)",
            "full_name": "Carlini & Wagner L2 Optimization",
            "category": "Optimization-based",
            "threat_model": "White-Box",
            "description": "Optimization-based attack finding minimal L2 distortion to alter predictions.",
        },
        {
            "id": "fab",
            "name": "FAB",
            "full_name": "Fast Adaptive Boundary Attack",
            "category": "Boundary-based",
            "threat_model": "White-Box",
            "description": "Fast adaptive boundary projection minimizing Lp distance without gradient masking.",
        },
    ],
    "text": [
        {
            "id": "embedding_pgd",
            "name": "Embedding PGD",
            "full_name": "Embedding-Space Projected Gradient Descent",
            "category": "Gradient-based",
            "threat_model": "White-Box",
            "description": "Iterative perturbation in continuous word and subword embedding spaces.",
        },
        {
            "id": "hotflip",
            "name": "HotFlip",
            "full_name": "HotFlip Gradient Substitution",
            "category": "Gradient-based (Token-level)",
            "threat_model": "White-Box",
            "description": "Single-token substitution using directional gradient approximation.",
        },
        {
            "id": "textbugger_wb",
            "name": "TextBugger WB",
            "full_name": "TextBugger White-Box Attack",
            "category": "Gradient-based / Saliency",
            "threat_model": "White-Box",
            "description": "Character and word-level perturbations guided by token saliency.",
        },
        {
            "id": "universal_trigger",
            "name": "Universal Trigger",
            "full_name": "Universal Adversarial Trigger",
            "category": "Optimization / Trigger",
            "threat_model": "White-Box",
            "description": "Optimizes static token sequences that induce targeted misclassification.",
        },
        {
            "id": "gradient_substitution",
            "name": "Gradient Substitution",
            "full_name": "First-Order Gradient Token Substitution",
            "category": "Gradient-based",
            "threat_model": "White-Box",
            "description": "Token-level replacement guided by first-order loss gradients.",
        },
    ],
    "time-series": [
        {
            "id": "time_series_pgd",
            "name": "Time-Series PGD",
            "full_name": "Temporal Projected Gradient Descent",
            "category": "Gradient-based",
            "threat_model": "White-Box",
            "description": "Iterative Linf/L2 gradient perturbation constrained to temporal signal bounds.",
        },
        {
            "id": "smooth_pgd",
            "name": "Smooth PGD",
            "full_name": "Smooth Projected Gradient Descent",
            "category": "Gradient-based / TV Regularized",
            "threat_model": "White-Box",
            "description": "PGD with Total Variation penalization to preserve signal waveform smoothness.",
        },
        {
            "id": "wavelet",
            "name": "Wavelet Attack",
            "full_name": "Wavelet Spectral Perturbation Attack",
            "category": "Spectral / Frequency-Domain",
            "threat_model": "White-Box",
            "description": "Injects adversarial perturbations into discrete wavelet frequency sub-bands.",
        },
        {
            "id": "soft_dtw",
            "name": "Soft-DTW Attack",
            "full_name": "Soft Dynamic Time Warping Attack",
            "category": "Temporal Warping",
            "threat_model": "White-Box",
            "description": "Adversarial time-series distortion minimizing differentiable soft-DTW distance.",
        },
        {
            "id": "time_series_jsma",
            "name": "Time-Series JSMA",
            "full_name": "Jacobian Saliency Map for Time-Series",
            "category": "Saliency-based",
            "threat_model": "White-Box",
            "description": "Greedy saliency-guided perturbation of critical temporal timestamps.",
        },
    ],
    "tabular": [
        {
            "id": "cap_pgd",
            "name": "CapPGD",
            "full_name": "Constraint-Aware Projected Gradient Descent",
            "category": "Gradient-based / Constrained",
            "threat_model": "White-Box",
            "description": "Iterative PGD projecting perturbations onto continuous and categorical bounds.",
        },
        {
            "id": "lowprofool",
            "name": "LowProFool",
            "full_name": "LowProFool Feature-Weighted Saliency",
            "category": "Gradient-based / Weighted",
            "threat_model": "White-Box",
            "description": "Perturbs tabular features while penalizing distance according to feature mutability.",
        },
        {
            "id": "tabular_cw",
            "name": "Tabular CW",
            "full_name": "Carlini & Wagner for Tabular Data",
            "category": "Optimization-based",
            "threat_model": "White-Box",
            "description": "L2 optimization attack respecting tabular feature boundaries and immutability.",
        },
        {
            "id": "tabular_jsma",
            "name": "Tabular JSMA",
            "full_name": "Jacobian Saliency Map for Tabular",
            "category": "Saliency-based",
            "threat_model": "White-Box",
            "description": "Iteratively perturbs the most influential mutable tabular feature columns.",
        },
        {
            "id": "vae_attack",
            "name": "VAE Attack",
            "full_name": "On-Manifold VAE Latent Attack",
            "category": "Manifold-based",
            "threat_model": "White-Box",
            "description": "Generates realistic adversarial tabular records by perturbing in latent VAE space.",
        },
    ],
}


def get_compatible_attacks(
    model: Any, data_domain: str = "image"
) -> List[Dict[str, Any]]:
    """
    Retrieve registered attacks from the existing attack engine and filter
    only those that are strictly compatible with the loaded model and domain.
    Eliminates duplicates (such as 'cw' vs 'carlini_wagner') by prioritizing canonical IDs.

    Args:
        model: Target model or BaseModelAdapter instance.
        data_domain: Target data domain string ('image', 'text', 'tabular', 'time-series').

    Returns:
        List of dictionaries with attack metadata and parameters for compatible attacks.
    """
    # 1. Trigger dynamic package discovery across attack modules
    discover_attacks()

    clean_domain = data_domain.strip().lower().replace("-", "_")
    if clean_domain in ("timeseries", "sequence"):
        clean_domain = "time-series"
    elif clean_domain in ("vision",):
        clean_domain = "image"
    elif clean_domain in ("nlp",):
        clean_domain = "text"
    elif clean_domain in ("csv",):
        clean_domain = "tabular"

    canonical_list = CANONICAL_DOMAIN_CATALOG.get(clean_domain, [])
    compatible_attacks: List[Dict[str, Any]] = []
    seen_classes = set()

    for item in canonical_list:
        atk_id = item["id"]
        try:
            attack_cls = get_attack(atk_id)
        except Exception:
            continue

        if attack_cls in seen_classes:
            continue

        meta: Optional[AttackMetadata] = getattr(attack_cls, "metadata", None)
        if meta is None:
            continue

        # Check model capability compatibility
        try:
            check_model_compatibility(model, meta)
        except (UnsupportedModelError, AttackConfigurationError):
            continue

        seen_classes.add(attack_cls)
        compatible_attacks.append(
            {
                "id": atk_id,
                "name": item["name"],
                "full_name": item["full_name"],
                "class_name": attack_cls.__name__,
                "domain": clean_domain,
                "category": item["category"],
                "threat_model": item["threat_model"],
                "description": item["description"],
                "supported_norms": meta.supported_norms or ["Linf"],
                "targeted_supported": bool(meta.targeted_supported),
                "untargeted_supported": bool(meta.untargeted_supported),
                "iterative": bool(meta.iterative),
                "computational_cost": meta.computational_cost.capitalize(),
                "attack_cls": attack_cls,
            }
        )

    # Fallback to general registry lookup if canonical list had no hits
    if not compatible_attacks:
        for name in sorted(list_attacks()):
            try:
                attack_cls = get_attack(name)
            except Exception:
                continue

            if attack_cls in seen_classes:
                continue

            meta = getattr(attack_cls, "metadata", None)
            if not meta:
                continue

            meta_domain = meta.domain.strip().lower().replace("-", "_")
            if clean_domain.replace("-", "_") != meta_domain:
                continue

            try:
                check_model_compatibility(model, meta)
            except Exception:
                continue

            seen_classes.add(attack_cls)
            compatible_attacks.append(
                {
                    "id": name,
                    "name": meta.name.upper(),
                    "full_name": meta.name.upper(),
                    "class_name": attack_cls.__name__,
                    "domain": meta.domain,
                    "category": meta.category.capitalize(),
                    "threat_model": "White-Box",
                    "description": f"{meta.name.upper()} adversarial attack algorithm.",
                    "supported_norms": meta.supported_norms or ["Linf"],
                    "targeted_supported": bool(meta.targeted_supported),
                    "untargeted_supported": bool(meta.untargeted_supported),
                    "iterative": bool(meta.iterative),
                    "computational_cost": meta.computational_cost.capitalize(),
                    "attack_cls": attack_cls,
                }
            )

    return compatible_attacks


def execute_configured_attack(
    model: Any,
    attack_name: str,
    inputs: Any,
    labels: Any,
    epsilon: float = 0.05,
    clip_min: Optional[float] = 0.0,
    clip_max: Optional[float] = 1.0,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute selected attack on the model using the backend execution interface
    and compute all telemetry matching run_pipeline_for_multimodel.py.

    Returns:
        Structured result dictionary formatted for Streamlit UI display.
    """
    # 1. Retrieve attack class from backend registry
    try:
        attack_cls = get_attack(attack_name)
    except Exception as exc:
        raise AttackConfigurationError(
            f"Attack '{attack_name}' is not registered: {str(exc)}"
        ) from exc

    # 2. Build backend AttackConfig
    config = AttackConfig(
        epsilon=float(epsilon),
        clip_min=float(clip_min) if clip_min is not None else None,
        clip_max=float(clip_max) if clip_max is not None else None,
        params=params or {},
    )

    # 3. Execute via existing backend attack executor
    try:
        t0 = time.time()
        attack_res: AttackResult = execute_attack(
            model=model,
            attack_cls=attack_cls,
            inputs=inputs,
            labels=labels,
            config=config,
        )
        wall_time = time.time() - t0
    except (AttackExecutionError, AttackConfigurationError, AttackError) as known_err:
        raise known_err
    except Exception as exc:
        raise AttackExecutionError(
            f"Backend execution failure during '{attack_name}': {str(exc)}"
        ) from exc

    # 4. Extract real predictions and confidences
    raw_model = getattr(model, "get_model", lambda: model)()
    orig_preds = attack_res.original_predictions
    adv_preds = attack_res.adversarial_predictions
    success_mask = attack_res.success
    metrics = dict(attack_res.perturbation_metrics or {})
    if isinstance(inputs, torch.Tensor) and inputs.numel() > 0 and "l0" in metrics:
        elem_per_sample = inputs[0].numel()
        if elem_per_sample > 0:
            metrics["l0_pct"] = min(
                100.0, float(metrics["l0"] / elem_per_sample * 100.0)
            )
    exec_time = attack_res.execution_time or wall_time

    # Compute confidences if feasible
    clean_confs = None
    adv_confs = None
    if isinstance(inputs, torch.Tensor) and hasattr(raw_model, "eval"):
        try:
            device = (
                next(raw_model.parameters()).device
                if list(raw_model.parameters())
                else torch.device("cpu")
            )
            with torch.no_grad():
                raw_model.eval()
                logits_c = raw_model(inputs.to(device))
                if hasattr(logits_c, "logits"):
                    logits_c = logits_c.logits
                probs_c = F.softmax(logits_c, dim=-1)
                clean_confs = probs_c.max(dim=-1).values.cpu()

                if attack_res.adversarial_examples is not None:
                    logits_a = raw_model(attack_res.adversarial_examples.to(device))
                    if hasattr(logits_a, "logits"):
                        logits_a = logits_a.logits
                    probs_a = F.softmax(logits_a, dim=-1)
                    adv_confs = probs_a.max(dim=-1).values.cpu()
        except Exception:
            pass

    # Accuracy calculations
    total_samples = len(labels) if hasattr(labels, "__len__") else 1
    labels_tensor = (
        labels if isinstance(labels, torch.Tensor) else torch.as_tensor(labels)
    )

    clean_correct = 0
    adv_correct = 0
    if orig_preds is not None and isinstance(orig_preds, torch.Tensor):
        clean_correct = int((orig_preds.cpu() == labels_tensor.cpu()).sum().item())
    if adv_preds is not None and isinstance(adv_preds, torch.Tensor):
        adv_correct = int((adv_preds.cpu() == labels_tensor.cpu()).sum().item())

    clean_acc = (clean_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    adv_acc = (adv_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    acc_drop = clean_acc - adv_acc

    # Confidence averages
    clean_conf_avg = (
        float(clean_confs.mean().item() * 100.0) if clean_confs is not None else None
    )
    adv_conf_avg = (
        float(adv_confs.mean().item() * 100.0) if adv_confs is not None else None
    )
    conf_drop = (
        (clean_conf_avg - adv_conf_avg)
        if (clean_conf_avg is not None and adv_conf_avg is not None)
        else None
    )

    # Calculate Attack Success Rate (ASR)
    successful_count = 0
    if success_mask is not None:
        if isinstance(success_mask, torch.Tensor):
            successful_count = int(success_mask.sum().item())
        elif isinstance(success_mask, (list, tuple)):
            successful_count = sum(1 for s in success_mask if s)
        elif isinstance(success_mask, bool):
            successful_count = 1 if success_mask else 0
    elif orig_preds is not None and adv_preds is not None:
        successful_count = int((orig_preds != adv_preds).sum().item())

    asr_pct = (successful_count / total_samples) * 100.0 if total_samples > 0 else 0.0

    # Format predictions for UI table comparison
    sample_comparisons: List[Dict[str, Any]] = []
    labels_list = labels.tolist() if isinstance(labels, torch.Tensor) else list(labels)
    clean_list = (
        orig_preds.tolist()
        if isinstance(orig_preds, torch.Tensor)
        else [None] * total_samples
    )
    adv_list = (
        adv_preds.tolist()
        if isinstance(adv_preds, torch.Tensor)
        else [None] * total_samples
    )
    clean_conf_list = (
        clean_confs.tolist() if clean_confs is not None else [None] * total_samples
    )
    adv_conf_list = (
        adv_confs.tolist() if adv_confs is not None else [None] * total_samples
    )

    for idx in range(total_samples):
        y_true = labels_list[idx] if idx < len(labels_list) else "N/A"
        y_orig = clean_list[idx] if idx < len(clean_list) else "N/A"
        y_adv = adv_list[idx] if idx < len(adv_list) else "N/A"
        c_conf = (
            f"{clean_conf_list[idx] * 100:.1f}%"
            if clean_conf_list[idx] is not None
            else "N/A"
        )
        a_conf = (
            f"{adv_conf_list[idx] * 100:.1f}%"
            if adv_conf_list[idx] is not None
            else "N/A"
        )

        misclassified = (
            (y_orig != y_adv) if (y_orig != "N/A" and y_adv != "N/A") else False
        )
        clean_right = (
            (y_orig == y_true) if (y_orig != "N/A" and y_true != "N/A") else True
        )

        sample_comparisons.append(
            {
                "Sample": f"#{idx + 1}",
                "Ground Truth": str(y_true),
                "Clean Prediction": f"{y_orig} ({c_conf})",
                "Adversarial Prediction": f"{y_adv} ({a_conf})",
                "Outcome": (
                    "Adversarial Flip ⚠️"
                    if misclassified
                    else (
                        "Preserved Correct ✅" if clean_right else "Preserved Error ❌"
                    )
                ),
            }
        )

    # Detailed Attack Specs for Overview
    static_meta = getattr(attack_cls, "metadata", None)
    meta_info = {
        "id": attack_name,
        "name": attack_res.attack_name.upper(),
        "class_name": attack_cls.__name__,
        "threat_model": "White-Box",
        "category": getattr(static_meta, "category", "Gradient-based").capitalize(),
        "computational_cost": getattr(
            static_meta, "computational_cost", "Medium"
        ).capitalize(),
        "supported_norms": getattr(static_meta, "supported_norms", ["Linf"]),
    }

    return {
        "attack_name": attack_res.attack_name.upper(),
        "attack_class": attack_cls.__name__,
        "attack_meta": meta_info,
        "execution_time": exec_time,
        "total_samples": total_samples,
        "successful_count": successful_count,
        "asr_percentage": asr_pct,
        "clean_accuracy": clean_acc,
        "adversarial_accuracy": adv_acc,
        "accuracy_drop": acc_drop,
        "clean_confidence": clean_conf_avg,
        "adversarial_confidence": adv_conf_avg,
        "clean_confidence_avg": clean_conf_avg,
        "adv_confidence_avg": adv_conf_avg,
        "confidence_drop": conf_drop,
        "ms_per_sample": (
            (exec_time / total_samples * 1000.0) if total_samples > 0 else 0.0
        ),
        "perturbation_metrics": metrics,
        "sample_comparisons": sample_comparisons,
        "has_adversarial_tensor": attack_res.adversarial_examples is not None,
        "epsilon_used": epsilon,
        "parameters": params or {},
        "clip_bounds": f"[{clip_min}, {clip_max}]",
    }
