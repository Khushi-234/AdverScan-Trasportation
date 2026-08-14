"""
Attack registry for cataloging and retrieving available adversarial attack classes.
"""

import inspect
from typing import Dict, List, Type
from app.attack_engine.base.base_attack import BaseAttack
from app.attack_engine.exceptions import AttackConfigurationError

# Registry dictionary mapping lowercase attack names to attack classes. Starts empty.
_ATTACK_REGISTRY: Dict[str, Type[BaseAttack]] = {}


def register_attack(name: str, attack_cls: Type[BaseAttack]) -> None:
    """
    Register an attack class under a given name.

    Args:
        name: String identifier for the attack.
        attack_cls: Class inheriting from BaseAttack.

    Raises:
        AttackConfigurationError: If attack_cls does not inherit from BaseAttack.
    """
    if not (inspect.isclass(attack_cls) and issubclass(attack_cls, BaseAttack)):
        raise AttackConfigurationError(
            f"Class '{attack_cls}' must be a subclass of BaseAttack."
        )

    _ATTACK_REGISTRY[name.lower()] = attack_cls


def get_attack(name: str) -> Type[BaseAttack]:
    """
    Retrieve a registered attack class by identifier name.

    Args:
        name: String identifier of the attack (e.g. 'fgsm').

    Returns:
        Registered attack class.

    Raises:
        AttackConfigurationError: If the attack is not registered.
    """
    name_lower = name.lower()
    if name_lower not in _ATTACK_REGISTRY:
        available = ", ".join(list_attacks()) if _ATTACK_REGISTRY else "none"
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


def clear_registry() -> None:
    """
    Clear all registered attacks from the registry and reset discovery state.
    """
    _ATTACK_REGISTRY.clear()
    from app.attack_engine.attack_discovery import reset_discovery_state
    reset_discovery_state()
