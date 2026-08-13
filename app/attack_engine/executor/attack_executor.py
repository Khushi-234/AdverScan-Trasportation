"""
Attack executor for coordinating attack execution on models.
"""

from typing import Any, Optional
from app.attack_engine.config.attack_config import AttackConfig
from app.attack_engine.registry.attack_registry import get_attack
from app.attack_engine.exceptions import AttackError, AttackExecutionError


def execute_attack(
    model: Any,
    attack_name: str,
    inputs: Any,
    labels: Any,
    config: Optional[AttackConfig] = None,
) -> Any:
    """
    Execute a specified adversarial attack on a given model.

    Args:
        model: Target PyTorch model or BaseModelAdapter.
        attack_name: Identifier of attack (e.g., 'fgsm').
        inputs: Original input batch tensor.
        labels: True class labels tensor.
        config: Optional AttackConfig instance. If None, default AttackConfig is used.

    Returns:
        Adversarial examples output tensor.

    Raises:
        AttackExecutionError: If attack fails or is improperly configured.
    """
    if config is None:
        config = AttackConfig()

    try:
        attack_cls = get_attack(attack_name)
        attack_instance = attack_cls(model)
        adv_inputs = attack_instance.generate(inputs, labels, config)
        return adv_inputs
    except AttackError as ae:
        raise ae
    except Exception as e:
        raise AttackExecutionError(
            f"Failed to execute attack '{attack_name}': {str(e)}"
        ) from e
