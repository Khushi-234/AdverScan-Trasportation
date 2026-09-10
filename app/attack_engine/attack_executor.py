"""
Attack executor for instantiating and running selected attack classes.
"""

import time
from typing import Any, Optional, Type
import torch

from app.attack_engine.attacks.base_attack import BaseAttack
from app.attack_engine.config import AttackConfig
from app.attack_engine.models import AttackMetadata, AttackResult
from app.attack_engine.exceptions import AttackError, AttackExecutionError
from app.attack_engine.utils.perturbation import compute_perturbation_metrics


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

        static_metadata = getattr(attack_cls, "metadata", None)
        attack_name = getattr(static_metadata, "name", getattr(attack_cls, "attack_name", attack_cls.__name__.lower()))
        attack_instance = attack_cls(model)

        # Validate configuration against attack requirements
        attack_instance.validate_config(config)

        start_time = time.time()
        adv_inputs = attack_instance.generate(inputs, labels, config)
        elapsed_time = time.time() - start_time

        # Compute perturbation metrics if inputs and adv_inputs are torch.Tensors
        metrics = {}
        if isinstance(inputs, torch.Tensor) and isinstance(adv_inputs, torch.Tensor):
            metrics = compute_perturbation_metrics(inputs.cpu(), adv_inputs.cpu())

        # Compute predictions & success mask if feasible
        orig_preds = None
        adv_preds = None
        success = None

        if isinstance(inputs, torch.Tensor) and isinstance(adv_inputs, torch.Tensor):
            try:
                raw_model = attack_instance._get_raw_model()
                with torch.no_grad():
                    device = next(raw_model.parameters()).device if list(raw_model.parameters()) else torch.device("cpu")
                    out_orig = raw_model(inputs.to(device))
                    logits_orig = out_orig.logits if hasattr(out_orig, "logits") else out_orig
                    orig_preds = logits_orig.argmax(dim=-1).cpu()

                    out_adv = raw_model(adv_inputs.to(device))
                    logits_adv = out_adv.logits if hasattr(out_adv, "logits") else out_adv
                    adv_preds = logits_adv.argmax(dim=-1).cpu()

                    success = (orig_preds != adv_preds)
            except Exception:
                pass

        # Execution metadata
        metadata = AttackMetadata(
            name=attack_name,
            attack_name=attack_name,
            attack_class=attack_cls.__name__,
            domain=getattr(static_metadata, "domain", "image"),
            category=getattr(static_metadata, "category", "gradient"),
            attack_type=getattr(static_metadata, "attack_type", "white_box"),
            requires_gradient=getattr(static_metadata, "requires_gradient", True),
            requires_loss=getattr(static_metadata, "requires_loss", True),
            requires_model_weights=getattr(static_metadata, "requires_model_weights", True),
            iterative=getattr(static_metadata, "iterative", False),
            targeted_supported=getattr(static_metadata, "targeted_supported", False),
            untargeted_supported=getattr(static_metadata, "untargeted_supported", True),
            query_based=getattr(static_metadata, "query_based", False),
            supported_norms=getattr(static_metadata, "supported_norms", ["Linf"]),
            computational_cost=getattr(static_metadata, "computational_cost", "low"),
            epsilon=config.epsilon,
            clip_min=config.clip_min,
            clip_max=config.clip_max,
            execution_time_seconds=elapsed_time,
            parameters=config.params or {},
        )

        query_count = getattr(attack_instance, "query_count", None)

        return AttackResult(
            adversarial_examples=adv_inputs,
            metadata=metadata,
            original_inputs=inputs,
            labels=labels,
            attack_name=attack_name,
            success=success,
            original_predictions=orig_preds,
            adversarial_predictions=adv_preds,
            perturbation_metrics=metrics,
            execution_time=elapsed_time,
            query_count=query_count,
            attack_metadata=static_metadata,
        )

    except AttackError as ae:
        raise ae
    except Exception as e:
        attack_name_str = getattr(attack_cls, "__name__", str(attack_cls))
        raise AttackExecutionError(
            f"Failed to execute attack '{attack_name_str}': {str(e)}"
        ) from e
