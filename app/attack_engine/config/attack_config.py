"""
Attack configuration dataclass for storing attack parameters.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AttackConfig:
    """
    Configuration container for adversarial attack parameters.

    Attributes:
        epsilon: Perturbation magnitude (e.g. 0.1 for FGSM).
        clip_min: Minimum clipping value for adversarial examples (e.g. 0.0).
        clip_max: Maximum clipping value for adversarial examples (e.g. 1.0).
        loss_fn: Loss function instance or name to compute gradients against.
        params: Dictionary for additional attack-specific parameters.
    """

    epsilon: float = 0.1
    clip_min: Optional[float] = 0.0
    clip_max: Optional[float] = 1.0
    loss_fn: Any = None
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.params is None:
            self.params = {}
