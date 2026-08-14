"""
Attack selector for validating, retrieving, and selecting attack classes from the registry.
"""

from typing import Any, List, Type, Union
from app.attack_engine.base.base_attack import BaseAttack
from app.attack_engine.attack_registry import get_attack, list_attacks
from app.attack_engine.attack_discovery import discover_attacks
from app.attack_engine.exceptions import AttackConfigurationError


def select_attacks(
    attack_names: Union[str, List[str]]
) -> List[Type[BaseAttack]]:
    """
    Ensure attack discovery has run, validate requested attack names, and return attack classes.

    Args:
        attack_names: Single attack identifier string or list of attack identifier strings.

    Returns:
        List of selected attack classes (subclasses of BaseAttack).

    Raises:
        AttackConfigurationError: If any requested attack is not registered.
    """
    # Ensure all available attack modules are discovered and self-registered
    discover_attacks()

    if isinstance(attack_names, str):
        attack_names = [attack_names]

    available = set(list_attacks())
    unknown = [name for name in attack_names if name.lower() not in available]
    if unknown:
        available_str = ", ".join(sorted(available)) if available else "none"
        raise AttackConfigurationError(
            f"Unknown attack(s): {', '.join(unknown)}. Available registered attacks: [{available_str}]"
        )

    return [get_attack(name) for name in attack_names]


def select_compatible_attacks(
    model: Any, attack_names: Union[str, List[str]]
) -> List[Type[BaseAttack]]:
    """
    Select attacks and evaluate compatibility with the provided model.

    Args:
        model: Target model or adapter instance.
        attack_names: Single attack name or list of attack names.

    Returns:
        List of compatible attack classes.
    """
    if model is None:
        raise AttackConfigurationError("Model must be provided to evaluate attack compatibility.")
    return select_attacks(attack_names)
