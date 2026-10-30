"""
Attack transformation evidence extraction for AdverScan explainability module.
Extracts empirical input diffs, modified tokens/features, positions, perturbation
counts and ratios, and evaluates configured constraint compliance.
Strictly decoupled from feature attributions to prevent inferring actual attack
modifications from model importance scores.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from app.explainability.utils import to_numpy


def extract_attack_transformation(
    clean_input: Any,
    adversarial_input: Any,
    modality: str = "text",
    metadata: Optional[Any] = None,
    feature_names: Optional[List[str]] = None,
    tokenizer: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Empirically extract attack transformation evidence by comparing original
    and adversarial inputs.

    Args:
        clean_input: Original baseline input.
        adversarial_input: Perturbed adversarial input.
        modality: Data modality ('text', 'tabular', 'image', 'timeseries').
        metadata: Optional attack metadata (AttackMetadata instance or dict).
        feature_names: Optional list of feature names (for tabular/time-series).
        tokenizer: Optional tokenizer for text decoding.

    Returns:
        Structured dictionary containing:
        - clean_input / adversarial_input summary
        - changed_tokens or changed_features
        - positions
        - perturbation_count
        - perturbation_ratio
        - constraint_compliance
    """
    mod = modality.lower() if isinstance(modality, str) else "general"

    # Extract constraint configuration from metadata if available
    constraint_info: Dict[str, Any] = {"configured": False, "compliant": True}
    max_budget: Optional[float] = None
    constraint_type: Optional[str] = None

    if metadata is not None:
        params: Dict[str, Any] = {}
        if hasattr(metadata, "parameters") and isinstance(metadata.parameters, dict):
            params = metadata.parameters
        elif isinstance(metadata, dict):
            params = metadata.get("parameters", metadata)

        # Check epsilon or budget constraints
        if hasattr(metadata, "epsilon") and metadata.epsilon not in (None, 0.0):
            max_budget = float(metadata.epsilon)
            constraint_type = "epsilon"
        elif "epsilon" in params and params["epsilon"] not in (None, 0.0):
            max_budget = float(params["epsilon"])
            constraint_type = "epsilon"
        elif "max_words_perturbed" in params:
            max_budget = float(params["max_words_perturbed"])
            constraint_type = "max_words_perturbed"
        elif "max_tokens_changed" in params:
            max_budget = float(params["max_tokens_changed"])
            constraint_type = "max_tokens_changed"
        elif "budget" in params:
            max_budget = float(params["budget"])
            constraint_type = "budget"

    # 1. Text modality: Token-level textual diff
    if mod == "text":
        clean_str = clean_input if isinstance(clean_input, str) else str(clean_input)
        adv_str = adversarial_input if isinstance(adversarial_input, str) else str(adversarial_input)

        clean_tokens = clean_str.split()
        adv_tokens = adv_str.split()

        n_clean = len(clean_tokens)
        n_adv = len(adv_tokens)
        max_len = max(n_clean, n_adv)

        changed_tokens: List[Dict[str, Any]] = []
        positions: List[int] = []

        for i in range(max_len):
            c_tok = clean_tokens[i] if i < n_clean else "<missing>"
            a_tok = adv_tokens[i] if i < n_adv else "<missing>"
            if c_tok != a_tok:
                positions.append(i)
                changed_tokens.append({
                    "position": i,
                    "clean_token": c_tok,
                    "adversarial_token": a_tok,
                })

        pert_count = len(positions)
        pert_ratio = float(pert_count / n_clean) if n_clean > 0 else 0.0

        if max_budget is not None and constraint_type is not None:
            if constraint_type in ("max_words_perturbed", "max_tokens_changed", "budget"):
                compliant = pert_count <= max_budget
            elif constraint_type == "epsilon":
                compliant = pert_ratio <= (max_budget + 1e-5)
            else:
                compliant = True
            constraint_info = {
                "configured": True,
                "constraint_type": constraint_type,
                "threshold": max_budget,
                "actual_value": pert_count if constraint_type in ("max_words_perturbed", "max_tokens_changed") else pert_ratio,
                "compliant": compliant,
            }

        return {
            "modality": "text",
            "clean_input_preview": clean_str[:120] + ("..." if len(clean_str) > 120 else ""),
            "adversarial_input_preview": adv_str[:120] + ("..." if len(adv_str) > 120 else ""),
            "total_tokens": n_clean,
            "perturbation_count": pert_count,
            "perturbation_ratio": pert_ratio,
            "positions": positions,
            "changed_tokens": changed_tokens,
            "constraint_compliance": constraint_info,
        }

    # 2. Continuous / Array modalities: Tabular, Time-series, Image
    try:
        clean_arr = to_numpy(clean_input, dtype=np.float64)
        adv_arr = to_numpy(adversarial_input, dtype=np.float64)
    except Exception:
        clean_arr = None
        adv_arr = None

    if clean_arr is None or adv_arr is None or clean_arr.shape != adv_arr.shape:
        return {
            "modality": mod,
            "status": "unavailable_or_shape_mismatch",
            "perturbation_count": 0,
            "perturbation_ratio": 0.0,
            "positions": [],
            "changed_features": [],
            "constraint_compliance": constraint_info,
        }

    diff = np.abs(adv_arr - clean_arr)
    # Detect numerical perturbations above floating point noise
    changed_mask = diff > 1e-6
    pert_count = int(np.count_nonzero(changed_mask))
    total_elements = int(clean_arr.size)
    pert_ratio = float(pert_count / total_elements) if total_elements > 0 else 0.0

    linf_norm = float(np.max(diff)) if diff.size > 0 else 0.0
    l2_norm = float(np.linalg.norm(diff)) if diff.size > 0 else 0.0

    if max_budget is not None and constraint_type == "epsilon":
        compliant = bool(linf_norm <= (max_budget + 1e-5))
        constraint_info = {
            "configured": True,
            "constraint_type": "epsilon_Linf",
            "threshold": max_budget,
            "actual_value": linf_norm,
            "compliant": compliant,
        }

    changed_features: List[Dict[str, Any]] = []
    positions = []

    if mod == "tabular" or clean_arr.ndim <= 2:
        clean_flat = clean_arr.ravel()
        adv_flat = adv_arr.ravel()
        diff_flat = diff.ravel()
        f_names = feature_names or [f"feature_{i}" for i in range(len(clean_flat))]

        changed_indices = np.where(diff_flat > 1e-6)[0]
        positions = [int(idx) for idx in changed_indices[:100]]

        for idx in changed_indices[:50]:  # Cap detail list at 50 to maintain compact report
            fname = f_names[idx] if idx < len(f_names) else f"feature_{idx}"
            changed_features.append({
                "feature": fname,
                "position": int(idx),
                "clean_value": float(clean_flat[idx]),
                "adversarial_value": float(adv_flat[idx]),
                "absolute_diff": float(diff_flat[idx]),
            })
    else:
        # High dimensional (image or 3D time-series)
        positions = [int(idx) for idx in np.where(changed_mask.ravel())[0][:50]]

    return {
        "modality": mod,
        "perturbation_count": pert_count,
        "total_elements": total_elements,
        "perturbation_ratio": pert_ratio,
        "linf_norm": linf_norm,
        "l2_norm": l2_norm,
        "positions": positions,
        "changed_features": changed_features,
        "constraint_compliance": constraint_info,
    }
