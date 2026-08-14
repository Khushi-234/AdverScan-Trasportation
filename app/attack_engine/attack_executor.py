"""
Attack executor for instantiating and running selected attack classes.
"""

import time
from typing import Any, Optional, Type
from app.attack_engine.base.base_attack import BaseAttack
from app.attack_engine.config import AttackConfig
from app.attack_engine.models import AttackMetadata, AttackResult
from app.attack_engine.exceptions import AttackError, AttackExecutionError


def execute_attack(
    model: Any,
    attack_cls: Type[BaseAttack],
    inputs: Any,
    labels: Any,
    config: Optional[AttackConfig] = None,
) -> AttackResult:
    """
    Execute an already selected attack class on a given model and return standardized AttackResult.

    Args:
        model: Target PyTorch model or BaseModelAdapter.
        attack_cls: Already selected class inheriting from BaseAttack.
        inputs: Original input batch tensor.
        labels: True class labels tensor.
        config: Optional AttackConfig instance. If None, default AttackConfig is used.

    Returns:
        AttackResult containing adversarial examples, metadata, original inputs, and labels.

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

        attack_name = getattr(attack_cls, "attack_name", attack_cls.__name__.lower())
        attack_instance = attack_cls(model)

        start_time = time.time()
        adv_inputs = attack_instance.generate(inputs, labels, config)
        elapsed_time = time.time() - start_time

        metadata = AttackMetadata(
            attack_name=attack_name,
            attack_class=attack_cls.__name__,
            epsilon=config.epsilon,
            clip_min=config.clip_min,
            clip_max=config.clip_max,
            execution_time_seconds=elapsed_time,
            parameters=config.params or {},
        )

        return AttackResult(
            adversarial_examples=adv_inputs,
            metadata=metadata,
            original_inputs=inputs,
            labels=labels,
        )

    except AttackError as ae:
        raise ae
    except Exception as e:
        attack_name_str = getattr(attack_cls, "__name__", str(attack_cls))
        raise AttackExecutionError(
            f"Failed to execute attack '{attack_name_str}': {str(e)}"
        ) from e
