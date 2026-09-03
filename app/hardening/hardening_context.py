"""
Hardening Context DTO for Module 7 (Hardening) in AdverScan.

Captures attack characteristics, model metadata, operational constraints, and vulnerability metrics.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class HardeningContext:
    """
    Evaluation context describing attack, model, environment, and vulnerability metrics.
    """

    # 1. Attack characteristics
    attack_name: str = "generic"
    is_iterative: bool = False
    perturbation_norm: str = "Linf"
    epsilon: float = 0.03
    attack_success_rate: float = 0.5

    # 2. Model characteristics
    parameter_count: Optional[int] = None
    architecture_type: str = "CNN"
    has_training_data: bool = True
    device: str = "cpu"
    supports_gradients: bool = True
    input_domain: str = "image"

    # 3. Operational constraints
    latency_sensitive: bool = False
    resource_limits: Optional[Dict[str, Any]] = None
    max_hardening_time: float = 300.0
    allow_retraining: bool = True
    has_labels: bool = True

    # 4. Vulnerability characteristics
    risk_level: str = "MEDIUM"
    vulnerability_score: float = 50.0
    accuracy_drop: float = 0.2
    confidence_drop: float = 0.3
    perturbation_magnitude: float = 0.03

    def __post_init__(self) -> None:
        """Normalize attributes upon creation."""
        self.attack_name = (self.attack_name or "").lower().strip()
        self.risk_level = (self.risk_level or "MEDIUM").upper().strip()
        self.input_domain = (self.input_domain or "image").lower().strip()
        if self.resource_limits is None:
            self.resource_limits = {}
        if self.attack_name in ("pgd", "bim", "cw", "carlini_wagner", "autoattack"):
            self.is_iterative = True

    @property
    def norm(self) -> str:
        """Alias for perturbation_norm."""
        return self.perturbation_norm

