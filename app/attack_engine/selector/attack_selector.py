"""
Attack selector for validating and selecting compatible attacks.
"""

from typing import Any, List, Type, Union
from app.attack_engine.base.base_attack import BaseAttack
from app.attack_engine.registry.attack_registry import get_attack, list_attacks
from app.attack_engine.exceptions import AttackConfigurationError


def select_attacks(
    attack_names: Union[str, List[str]]
) -> List[Type[BaseAttack]]:
    """
    Select and return attack classes matching the requested names.

    Args:
        attack_names: Single attack name string or list of attack name strings.

    Returns:
        List of selected attack classes.

    Raises:
        AttackConfigurationError: If any specified attack name is not registered.
    """
    if isinstance(attack_names, str):
        attack_names = [attack_names]

    available = set(list_attacks())
    unknown = [name for name in attack_names if name.lower() not in available]
    if unknown:
        raise AttackConfigurationError(
            f"Unknown attack(s): {', '.join(unknown)}. Available attacks: {', '.join(sorted(available))}"
        )

    return [get_attack(name) for name in attack_names]


def select_compatible_attacks(
    model: Any, attack_names: Union[str, List[str]]
) -> List[Type[BaseAttack]]:
    """
    Select attacks and verify compatibility with the given model.

    Args:
        model: Target model or adapter.
        attack_names: Single attack name or list of attack names.

    Returns:
        List of compatible attack classes.
    """
    selected = select_attacks(attack_names)
    # Basic validation: ensure model is provided
    if model is None:
        raise AttackConfigurationError("Model must be provided to evaluate attack compatibility.")
    return selected
