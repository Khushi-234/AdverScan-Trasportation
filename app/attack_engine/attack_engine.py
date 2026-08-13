"""
Attack Engine Orchestrator for managing multi-attack execution pipelines.
"""

from typing import Any, Dict, List, Optional, Union
from app.attack_engine.config.attack_config import AttackConfig
from app.attack_engine.executor.attack_executor import execute_attack
from app.attack_engine.selector.attack_selector import select_attacks
from app.attack_engine.exceptions import AttackConfigurationError, AttackExecutionError


class AttackEngine:
    """
    Orchestrates the selection, configuration, and execution of adversarial attacks on models.
    """

    def __init__(self, model: Any):
        """
        Initialize the AttackEngine orchestrator.

        Args:
            model: Target model or adapter.
        """
        if model is None:
            raise AttackConfigurationError("AttackEngine requires a valid model instance.")
        self.model = model

    def run_attack(
        self,
        attack_name: str,
        inputs: Any,
        labels: Any,
        config: Optional[AttackConfig] = None,
    ) -> Any:
        """
        Run a single attack by name.

        Args:
            attack_name: Registered attack name.
            inputs: Input data.
            labels: Ground truth labels.
            config: Optional AttackConfig.

        Returns:
            Adversarial examples tensor.
        """
        return execute_attack(
            model=self.model,
            attack_name=attack_name,
            inputs=inputs,
            labels=labels,
            config=config,
        )

    def run_pipeline(
        self,
        attack_names: Union[str, List[str]],
        inputs: Any,
        labels: Any,
        configs: Optional[Dict[str, AttackConfig]] = None,
    ) -> Dict[str, Any]:
        """
        Run multiple attacks in a pipeline and return dictionary of adversarial examples.

        Args:
            attack_names: List of attack identifiers or single attack string.
            inputs: Input data batch.
            labels: Ground truth labels.
            configs: Optional dict mapping attack_name -> AttackConfig.

        Returns:
            Dict mapping attack_name to adversarial examples.
        """
        if isinstance(attack_names, str):
            attack_names = [attack_names]

        # Validate attack selection
        select_attacks(attack_names)

        if configs is None:
            configs = {}

        results: Dict[str, Any] = {}
        for name in attack_names:
            config = configs.get(name)
            adv_examples = self.run_attack(
                attack_name=name, inputs=inputs, labels=labels, config=config
            )
            results[name] = adv_examples

        return results


# Alias for backward compatibility / clarity
AttackOrchestrator = AttackEngine


def run_attack_pipeline(
    model: Any,
    attack_names: Union[str, List[str]],
    inputs: Any,
    labels: Any,
    configs: Optional[Dict[str, AttackConfig]] = None,
) -> Dict[str, Any]:
    """
    Functional entry point for running an attack pipeline.

    Args:
        model: Target model or adapter.
        attack_names: Single attack name or list of attack names.
        inputs: Input data batch.
        labels: Ground truth labels.
        configs: Optional dict mapping attack_name -> AttackConfig.

    Returns:
        Dict mapping attack_name to adversarial examples.
    """
    engine = AttackEngine(model)
    return engine.run_pipeline(
        attack_names=attack_names, inputs=inputs, labels=labels, configs=configs
    )
