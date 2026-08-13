"""
Attack registry for cataloging and retrieving available adversarial attack classes.
"""

from typing import Dict, List, Type
from app.attack_engine.base.base_attack import BaseAttack
from app.attack_engine.attacks.fgsm import FGSM
from app.attack_engine.exceptions import AttackConfigurationError

# Centralized attack registry mapping lowercase attack names to attack classes
_ATTACK_REGISTRY: Dict[str, Type[BaseAttack]] = {
    "fgsm": FGSM,
}


def register_attack(name: str, attack_cls: Type[BaseAttack]) -> None:
    """
    Register a new attack class under a given name.

    Args:
        name: String identifier for the attack.
        attack_cls: Class inheriting from BaseAttack.
    """
    if not issubclass(attack_cls, BaseAttack):
        raise AttackConfigurationError(
            f"Class {attack_cls} must inherit from BaseAttack."
        )
    _ATTACK_REGISTRY[name.lower()] = attack_cls


def get_attack(name: str) -> Type[BaseAttack]:
    """
    Retrieve an attack class by its registered identifier.

    Args:
        name: String identifier of the attack (e.g., 'fgsm').

    Returns:
        The registered attack class.

    Raises:
        AttackConfigurationError: If the attack name is not registered.
    """
    name_lower = name.lower()
    if name_lower not in _ATTACK_REGISTRY:
        available = ", ".join(list_attacks())
        raise AttackConfigurationError(
            f"Attack '{name}' is not registered. Available attacks: [{available}]"
        )
    return _ATTACK_REGISTRY[name_lower]


def list_attacks() -> List[str]:
    """
    Return a list of all registered attack identifiers.

    Returns:
        List of registered attack name strings.
    """
    return list(_ATTACK_REGISTRY.keys())
