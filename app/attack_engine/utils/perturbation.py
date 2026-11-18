"""
Perturbation metric calculations for evaluating adversarial examples.
"""

from typing import Any, Dict
import torch


def compute_perturbation(
    x_orig: torch.Tensor, x_adv: torch.Tensor
) -> torch.Tensor:
    """
    Compute raw perturbation delta: x_adv - x_orig.
    """
    if x_orig.shape != x_adv.shape:
        raise ValueError(
            f"Shape mismatch: x_orig shape {x_orig.shape} does not match x_adv shape {x_adv.shape}."
        )
    return x_adv - x_orig


def compute_l0_norm(
    x_orig: torch.Tensor, x_adv: torch.Tensor, threshold: float = 1e-5
) -> float:
    """
    Compute average L0 norm (fraction or count of altered elements per sample).
    How many elements changed?
    """
    if x_orig.shape != x_adv.shape:
        raise ValueError(
            f"Shape mismatch: x_orig shape {x_orig.shape} does not match x_adv shape {x_adv.shape}."
        )
    delta = (x_adv.float() - x_orig.float()).abs()
    batch_size = x_orig.size(0)
    changed = (delta > threshold).float().view(batch_size, -1).sum(dim=1)
    return float(changed.mean().item())


def compute_l2_norm(x_orig: torch.Tensor, x_adv: torch.Tensor) -> float:
    """
    Compute average L2 perturbation norm per sample.
    How large is the overall perturbation?
    """
    if x_orig.shape != x_adv.shape:
        raise ValueError(
            f"Shape mismatch: x_orig shape {x_orig.shape} does not match x_adv shape {x_adv.shape}."
        )
    delta = (x_adv.float() - x_orig.float()).view(x_orig.size(0), -1)
    l2 = torch.norm(delta, p=2, dim=1)
    return float(l2.mean().item())


def compute_linf_norm(x_orig: torch.Tensor, x_adv: torch.Tensor) -> float:
    """
    Compute average L_infinity perturbation norm per sample.
    What was the largest individual change?
    """
    if x_orig.shape != x_adv.shape:
        raise ValueError(
            f"Shape mismatch: x_orig shape {x_orig.shape} does not match x_adv shape {x_adv.shape}."
        )
    delta = (x_adv.float() - x_orig.float()).abs().view(x_orig.size(0), -1)
    linf = torch.max(delta, dim=1)[0]
    return float(linf.mean().item())


def compute_perturbation_metrics(
    x_orig: torch.Tensor, x_adv: torch.Tensor
) -> Dict[str, Any]:
    """
    Compute standard perturbation metrics comparing original inputs to adversarial inputs.

    Args:
        x_orig: Original input batch tensor.
        x_adv: Adversarial input batch tensor.

    Returns:
        Dict containing standard perturbation norms (linf, l2, l0, mean_abs_error)
        as well as detailed mean/max breakdown matching backend evaluation contracts.
        For sequence-length-changing attacks (where shapes do not match), returns None
        for point-wise norm values with explicit shape_mismatch metadata.
    """
    if not isinstance(x_orig, torch.Tensor) or not isinstance(x_adv, torch.Tensor):
        return {}

    # Guard against mismatched tensor shapes (e.g. sequence-length-changing text attacks)
    if x_orig.shape != x_adv.shape:
        return {
            "linf": None,
            "linf_mean": None,
            "linf_max": None,
            "linf_min": None,
            "l2": None,
            "l2_mean": None,
            "l2_max": None,
            "l2_min": None,
            "l0": None,
            "l0_mean_count": None,
            "l0_mean_pct": None,
            "l0_pct": None,
            "mean_abs_error": None,
            "mae_mean": None,
            "shape_mismatch": True,
            "orig_shape": list(x_orig.shape),
            "adv_shape": list(x_adv.shape),
        }

    delta = (x_adv.float() - x_orig.float()).abs()
    batch_size = x_orig.size(0)
    delta_flat = delta.view(batch_size, -1)

    linf_per_sample = torch.max(delta_flat, dim=1)[0]
    linf_mean = float(linf_per_sample.mean().item())
    linf_max = float(linf_per_sample.max().item())
    linf_min = float(linf_per_sample.min().item())

    l2_per_sample = torch.norm(delta_flat, p=2, dim=1)
    l2_mean = float(l2_per_sample.mean().item())
    l2_max = float(l2_per_sample.max().item())
    l2_min = float(l2_per_sample.min().item())

    total_elem = delta_flat.size(1)
    l0_counts = (delta_flat > 1e-5).float().sum(dim=1)
    l0_mean_count = float(l0_counts.mean().item())
    l0_mean_pct = (
        float((l0_counts / max(1, total_elem) * 100.0).mean().item())
        if total_elem > 0
        else 0.0
    )

    mae_mean = float(delta.mean().item())

    return {
        "linf": linf_mean,
        "linf_mean": linf_mean,
        "linf_max": linf_max,
        "linf_min": linf_min,
        "l2": l2_mean,
        "l2_mean": l2_mean,
        "l2_max": l2_max,
        "l2_min": l2_min,
        "l0": l0_mean_count,
        "l0_mean_count": l0_mean_count,
        "l0_mean_pct": l0_mean_pct,
        "l0_pct": l0_mean_pct,
        "mean_abs_error": mae_mean,
        "mae_mean": mae_mean,
    }
