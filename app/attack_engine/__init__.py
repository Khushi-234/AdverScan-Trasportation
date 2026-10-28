"""
Adversarial Attack Engine Module for AdverScan.
"""

from app.attack_engine.attacks.base_attack import BaseAttack
from app.attack_engine.attacks.image.fgsm import FGSM
from app.attack_engine.attacks.image.pgd import PGD
from app.attack_engine.attacks.image.deepfool import DeepFool
from app.attack_engine.attacks.image.cw import CW
from app.attack_engine.attacks.image.fab import FAB
from app.attack_engine.config import AttackConfig
from app.attack_engine.models import AttackMetadata, AttackResult, AttackResults
from app.attack_engine.exceptions import (
    AttackError,
    AttackConfigurationError,
    AttackExecutionError,
    UnsupportedModelError,
)
from app.attack_engine.attack_registry import (
    register_attack,
    get_attack,
    list_attacks,
    clear_registry,
)
from app.attack_engine.attack_discovery import discover_attacks
from app.attack_engine.attack_selector import (
    select_attacks,
    select_compatible_attacks,
)
from app.attack_engine.attack_executor import execute_attack
from app.attack_engine.attack_engine import AttackEngine, run_attack_pipeline

__all__ = [
    "BaseAttack",
    "AttackMetadata",
    "FGSM",
    "PGD",
    "DeepFool",
    "CW",
    "FAB",
    "AttackConfig",
    "AttackResult",
    "AttackResults",
    "AttackError",
    "AttackConfigurationError",
    "AttackExecutionError",
    "UnsupportedModelError",
    "register_attack",
    "get_attack",
    "list_attacks",
    "clear_registry",
    "discover_attacks",
    "select_attacks",
    "select_compatible_attacks",
    "execute_attack",
    "AttackEngine",
    "run_attack_pipeline",
]
