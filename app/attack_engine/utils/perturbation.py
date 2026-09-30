"""
Perturbation metric calculations for evaluating adversarial examples.
"""

from typing import Dict
import torch


def compute_perturbation(
    x_orig: torch.Tensor, x_adv: torch.Tensor
) -> torch.Tensor:
    """
    Compute raw perturbation delta: x_adv - x_orig.
    """
    return x_adv - x_orig


def compute_l0_norm(
    x_orig: torch.Tensor, x_adv: torch.Tensor, threshold: float = 1e-5
) -> float:
    """
    Compute average L0 norm (fraction or count of altered elements per sample).
    How many elements changed?
    """
    delta = (x_adv - x_orig).abs()
    batch_size = x_orig.size(0)
    changed = (delta > threshold).float().view(batch_size, -1).sum(dim=1)
    return float(changed.mean().item())


def compute_l2_norm(x_orig: torch.Tensor, x_adv: torch.Tensor) -> float:
    """
    Compute average L2 perturbation norm per sample.
    How large is the overall perturbation?
    """
    delta = (x_adv - x_orig).view(x_orig.size(0), -1)
    l2 = torch.norm(delta, p=2, dim=1)
    return float(l2.mean().item())


def compute_linf_norm(x_orig: torch.Tensor, x_adv: torch.Tensor) -> float:
    """
    Compute average L_infinity perturbation norm per sample.
    What was the largest individual change?
    """
    delta = (x_adv - x_orig).abs().view(x_orig.size(0), -1)
    linf = torch.max(delta, dim=1)[0]
    return float(linf.mean().item())


def compute_perturbation_metrics(
    x_orig: torch.Tensor, x_adv: torch.Tensor
) -> Dict[str, float]:
    """
    Compute standard perturbation metrics comparing original inputs to adversarial inputs.

    Args:
        x_orig: Original input batch tensor.
        x_adv: Adversarial input batch tensor.

    Returns:
        Dict containing 'l0', 'l2', 'linf', and 'mean_abs_error'.
    """
    if not isinstance(x_orig, torch.Tensor) or not isinstance(x_adv, torch.Tensor):
        return {}

    delta = (x_adv - x_orig).abs()
    mae = float(delta.mean().item())

    return {
        "l0": compute_l0_norm(x_orig, x_adv),
        "l2": compute_l2_norm(x_orig, x_adv),
        "linf": compute_linf_norm(x_orig, x_adv),
        "mean_abs_error": mae,
    }
