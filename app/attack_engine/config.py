"""
Attack configuration dataclass for storing attack parameters.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from app.attack_engine.exceptions import AttackConfigurationError


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
        if self.epsilon < 0:
            raise AttackConfigurationError(
                f"Epsilon cannot be negative, got {self.epsilon}"
            )
        if (
            self.clip_min is not None
            and self.clip_max is not None
            and self.clip_min > self.clip_max
        ):
            raise AttackConfigurationError(
                f"clip_min ({self.clip_min}) cannot be greater than clip_max ({self.clip_max})"
            )
