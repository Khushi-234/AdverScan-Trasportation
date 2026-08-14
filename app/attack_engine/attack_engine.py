"""
Attack Engine Orchestrator for managing multi-attack execution pipelines.
"""

from typing import Any, Dict, List, Optional, Union
from app.attack_engine.config import AttackConfig
from app.attack_engine.models import AttackResult, AttackResults
from app.attack_engine.attack_discovery import discover_attacks
from app.attack_engine.attack_selector import select_attacks
from app.attack_engine.attack_executor import execute_attack
from app.attack_engine.exceptions import AttackConfigurationError


class AttackEngine:
    """
    Orchestrates discovery, selection, configuration, and execution of adversarial attacks.
    """

    def __init__(self, model: Any):
        """
        Initialize the AttackEngine orchestrator.

        Args:
            model: Target model or adapter instance.
        """
        if model is None:
            raise AttackConfigurationError("AttackEngine requires a valid model instance.")
        self.model = model
        # Ensure attack discovery runs on engine initialization
        discover_attacks()

    def run_attack(
        self,
        attack_name: str,
        inputs: Any,
        labels: Any,
        config: Optional[AttackConfig] = None,
    ) -> AttackResult:
        """
        Run a single attack by name string and return AttackResult.

        Args:
            attack_name: Registered attack name.
            inputs: Input data batch.
            labels: Ground truth labels.
            config: Optional AttackConfig instance.

        Returns:
            AttackResult containing adversarial examples and execution metadata.
        """
        selected_classes = select_attacks(attack_name)
        attack_cls = selected_classes[0]
        return execute_attack(
            model=self.model,
            attack_cls=attack_cls,
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
    ) -> AttackResults:
        """
        Run multiple attacks dynamically in a pipeline and return AttackResults collection.

        Args:
            attack_names: List of attack identifiers or single attack string.
            inputs: Input data batch.
            labels: Ground truth labels.
            configs: Optional dict mapping attack_name -> AttackConfig.

        Returns:
            AttackResults collection mapping attack_name to AttackResult.
        """
        if isinstance(attack_names, str):
            attack_names = [attack_names]

        if configs is None:
            configs = {}

        pipeline_results = AttackResults()
        for name in attack_names:
            selected_classes = select_attacks(name)
            attack_cls = selected_classes[0]
            config = configs.get(name.lower())
            attack_result = execute_attack(
                model=self.model,
                attack_cls=attack_cls,
                inputs=inputs,
                labels=labels,
                config=config,
            )
            pipeline_results[name] = attack_result

        return pipeline_results


def run_attack_pipeline(
    model: Any,
    attack_names: Union[str, List[str]],
    inputs: Any,
    labels: Any,
    configs: Optional[Dict[str, AttackConfig]] = None,
) -> AttackResults:
    """
    Functional entry point for running an attack pipeline.

    Args:
        model: Target model or adapter.
        attack_names: Single attack name or list of attack names.
        inputs: Input data batch.
        labels: Ground truth labels.
        configs: Optional dict mapping attack_name -> AttackConfig.

    Returns:
        AttackResults collection mapping attack_name to AttackResult.
    """
    engine = AttackEngine(model)
    return engine.run_pipeline(
        attack_names=attack_names, inputs=inputs, labels=labels, configs=configs
    )
