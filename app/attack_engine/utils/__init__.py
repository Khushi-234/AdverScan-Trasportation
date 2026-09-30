"""
Utility modules for the Adversarial Attack Engine.
"""

from app.attack_engine.utils.tensor_utils import (
    get_device,
    to_tensor,
    clip_tensor,
    project_lp,
)
from app.attack_engine.utils.validation import (
    validate_inputs,
    validate_bounds,
    validate_attack_config,
    check_model_compatibility,
)
from app.attack_engine.utils.perturbation import (
    compute_perturbation,
    compute_l0_norm,
    compute_l2_norm,
    compute_linf_norm,
    compute_perturbation_metrics,
)

__all__ = [
    "get_device",
    "to_tensor",
    "clip_tensor",
    "project_lp",
    "validate_inputs",
    "validate_bounds",
    "validate_attack_config",
    "check_model_compatibility",
    "compute_perturbation",
    "compute_l0_norm",
    "compute_l2_norm",
    "compute_linf_norm",
    "compute_perturbation_metrics",
]
