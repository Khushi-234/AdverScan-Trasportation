"""
Main Coordinator Module for Module 7 (Hardening) in AdverScan.

Coordinates the complete model hardening process using DefenseSelector and defensive strategies.
"""

import time
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn

from app.hardening.defense_selector import DefenseSelector
from app.hardening.defenses import BaseDefense, get_defense_class
from app.hardening.hardening_result import HardeningResult
from app.hardening.exceptions import DefenseNotFoundError, HardeningConfigurationError, HardeningError


class HardeningEngine:
    """
    Main coordinator engine for model hardening and defensive reinforcement.

    Accepts target PyTorch models, input/label tensors, and vulnerability profiles to select,
    instantiate, execute, and evaluate defenses.
    """

    def __init__(
        self,
        defense_selector: Optional[DefenseSelector] = None,
    ) -> None:
        """
        Initialize HardeningEngine.

        Args:
            defense_selector: Optional custom DefenseSelector instance.
        """
        self.selector = defense_selector or DefenseSelector()

    def harden(
        self,
        model: nn.Module,
        defense: Union[str, BaseDefense] = "auto",
        inputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attack_name: Optional[str] = None,
        risk_level: Optional[str] = None,
        epsilon: Optional[float] = None,
        vulnerability_score: Optional[float] = None,
        defense_config: Optional[Dict[str, Any]] = None,
        eval_fn: Optional[Any] = None,
    ) -> HardeningResult:
        """
        Execute model hardening pipeline.

        Args:
            model: PyTorch model to harden.
            defense: Registered defense name ('spatial_smoothing', 'randomized_smoothing', 'adversarial_training'),
                     'auto' for smart selection via DefenseSelector, or a BaseDefense instance.
            inputs: Optional clean or adversarial input tensor.
            labels: Optional ground truth target label tensor.
            attack_name: Attack identifier for auto-selection.
            risk_level: Risk level for auto-selection.
            epsilon: Perturbation magnitude for auto-selection.
            vulnerability_score: Vulnerability score for auto-selection.
            defense_config: Optional kwargs passed to the defense instantiation.
            eval_fn: Optional evaluation function taking model and returning metrics dict.

        Returns:
            HardeningResult: DTO containing hardened model, inputs, execution metadata, and evaluation metrics.
        """
        if not isinstance(model, nn.Module):
            raise HardeningConfigurationError(f"Expected model to be PyTorch nn.Module, got {type(model)}")

        defense_config = defense_config or {}

        # Evaluate metrics before hardening if evaluation function is provided
        metrics_before: Dict[str, Any] = {}
        if eval_fn is not None and callable(eval_fn):
            try:
                metrics_before = eval_fn(model)
            except Exception as e:
                metrics_before = {"eval_error": str(e)}

        # Resolve Defense instance
        if isinstance(defense, BaseDefense):
            defense_obj = defense
        elif isinstance(defense, str):
            def_str = defense.lower().strip()
            if def_str == "auto":
                defense_obj = self.selector.select(
                    attack_name=attack_name,
                    risk_level=risk_level,
                    epsilon=epsilon,
                    vulnerability_score=vulnerability_score,
                )
            else:
                defense_cls = get_defense_class(def_str)
                defense_obj = defense_cls(**defense_config)
        else:
            raise HardeningConfigurationError(f"Invalid defense parameter type: {type(defense)}")

        # Execute defense application
        result: HardeningResult = defense_obj.apply(
            model=model,
            inputs=inputs,
            labels=labels,
        )

        # Attach metrics before
        result.metrics_before = metrics_before

        # Evaluate metrics after hardening if evaluation function is provided
        if eval_fn is not None and callable(eval_fn):
            try:
                result.metrics_after = eval_fn(result.hardened_model)
            except Exception as e:
                result.metrics_after = {"eval_error": str(e)}

        # Add selector recommendations if available
        rec_info = self.selector.recommend(
            attack_name=attack_name,
            risk_level=risk_level,
            epsilon=epsilon,
            vulnerability_score=vulnerability_score,
        )
        if rec_info["rationale"] not in result.recommendations:
            result.recommendations.append(f"Selector Context: {rec_info['rationale']}")

        return result
