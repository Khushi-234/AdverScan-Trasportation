"""
Adversarial Attack Engine Module for AdverScan.
"""

from app.attack_engine.base.base_attack import BaseAttack
from app.attack_engine.attacks.fgsm import FGSM
from app.attack_engine.config.attack_config import AttackConfig
from app.attack_engine.exceptions import (
    AttackError,
    AttackConfigurationError,
    AttackExecutionError,
    UnsupportedModelError,
)
from app.attack_engine.registry.attack_registry import (
    register_attack,
    get_attack,
    list_attacks,
)
from app.attack_engine.selector.attack_selector import (
    select_attacks,
    select_compatible_attacks,
)
from app.attack_engine.executor.attack_executor import execute_attack
from app.attack_engine.attack_engine import AttackEngine, AttackOrchestrator, run_attack_pipeline

__all__ = [
    "BaseAttack",
    "FGSM",
    "AttackConfig",
    "AttackError",
    "AttackConfigurationError",
    "AttackExecutionError",
    "UnsupportedModelError",
    "register_attack",
    "get_attack",
    "list_attacks",
    "select_attacks",
    "select_compatible_attacks",
    "execute_attack",
    "AttackEngine",
    "AttackOrchestrator",
    "run_attack_pipeline",
]
