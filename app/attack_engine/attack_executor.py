"""
Attack executor for instantiating and running selected attack classes.
"""

from typing import Any, Optional, Type
from app.attack_engine.base.base_attack import BaseAttack
from app.attack_engine.config import AttackConfig
from app.attack_engine.exceptions import AttackError, AttackExecutionError


def execute_attack(
    model: Any,
    attack_cls: Type[BaseAttack],
    inputs: Any,
    labels: Any,
    config: Optional[AttackConfig] = None,
) -> Any:
    """
    Execute an already selected attack class on a given model.

    Args:
        model: Target PyTorch model or BaseModelAdapter.
        attack_cls: Already selected class inheriting from BaseAttack.
        inputs: Original input batch tensor.
        labels: True class labels tensor.
        config: Optional AttackConfig instance. If None, default AttackConfig is used.

    Returns:
        Adversarial examples output tensor.

    Raises:
        AttackExecutionError: If attack execution fails or class is invalid.
    """
    if config is None:
        config = AttackConfig()

    try:
        if not (isinstance(attack_cls, type) and issubclass(attack_cls, BaseAttack)):
            raise AttackExecutionError(
                f"Expected attack_cls to be a subclass of BaseAttack, got {attack_cls}"
            )

        attack_instance = attack_cls(model)
        adv_inputs = attack_instance.generate(inputs, labels, config)
        return adv_inputs

    except AttackError as ae:
        raise ae
    except Exception as e:
        attack_name = getattr(attack_cls, "__name__", str(attack_cls))
        raise AttackExecutionError(
            f"Failed to execute attack '{attack_name}': {str(e)}"
        ) from e
