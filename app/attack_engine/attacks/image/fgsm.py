"""
Fast Gradient Sign Method (FGSM) adversarial attack implementation.
"""

from typing import Any, Optional
import torch
import torch.nn as nn

from app.attack_engine.attacks.base_attack import BaseAttack
from app.attack_engine.models import AttackMetadata
from app.attack_engine.config import AttackConfig
from app.attack_engine.exceptions import AttackExecutionError
from app.attack_engine.attack_registry import register_attack
from app.attack_engine.utils.tensor_utils import get_device, clip_tensor
from app.attack_engine.utils.validation import validate_inputs, validate_attack_config


class FGSM(BaseAttack):
    """
    Fast Gradient Sign Method (FGSM) attack implementation.

    Perturbs input tensor according to:
        x_adv = x + epsilon * sign(grad_x(Loss(model(x), y)))  [untargeted]
        x_adv = x - epsilon * sign(grad_x(Loss(model(x), y_target)))  [targeted]
    """

    metadata = AttackMetadata(
        name="fgsm",
        domain="image",
        category="gradient",
        attack_type="white_box",
        requires_gradient=True,
        requires_loss=True,
        requires_model_weights=True,
        iterative=False,
        targeted_supported=True,
        untargeted_supported=True,
        query_based=False,
        supported_norms=["Linf"],
        computational_cost="low",
        supported_model_types=["pytorch"],
    )

    def __init__(self, model: Any):
        """
        Initialize FGSM attack.

        Args:
            model: PyTorch nn.Module or BaseModelAdapter wrapping PyTorch model.
        """
        super().__init__(model)
        self.raw_model = self._get_raw_model()

    def validate_config(self, config: Optional[AttackConfig] = None) -> None:
        """
        Validate FGSM-specific configuration.
        """
        validate_attack_config(config)
        if config is not None:
            params = config.params or {}
            eps = params.get("epsilon", config.epsilon)
            if eps is not None and eps < 0:
                raise AttackConfigurationError(f"Epsilon cannot be negative, got {eps}")

    def generate(self, inputs: Any, labels: Any, config: Optional[AttackConfig] = None) -> Any:
        """
        Generate FGSM adversarial examples.

        Args:
            inputs: Input tensor (e.g. torch.Tensor).
            labels: True class labels tensor (or target labels if targeted=True).
            config: AttackConfig containing epsilon, clip_min, clip_max, loss_fn.

        Returns:
            Adversarial input tensor.
        """
        if config is None:
            config = AttackConfig()

        self.validate_config(config)
        validate_inputs(inputs, labels, attack_name="FGSM")

        params = config.params or {}
        targeted = bool(params.get("targeted", False))
        target_label = params.get("target_label", params.get("target", None))
        epsilon = float(params.get("epsilon", config.epsilon if config.epsilon is not None else 0.1))
        clip_min = params.get("clip_min", config.clip_min)
        clip_max = params.get("clip_max", config.clip_max)
        loss_fn = params.get("loss_fn", config.loss_fn) or nn.CrossEntropyLoss()

        # Device selection
        device = get_device(self.raw_model)

        # Move inputs and labels to target device
        inputs = inputs.clone().detach().to(device)
        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels)
        labels = labels.to(device)
        inputs.requires_grad_(True)

        # Determine target tensor if explicitly provided
        if targeted and target_label is not None:
            if isinstance(target_label, int):
                loss_labels = torch.full((inputs.size(0),), target_label, dtype=torch.long, device=device)
            else:
                target_tensor = torch.as_tensor(target_label, dtype=torch.long, device=device)
                if target_tensor.dim() == 0:
                    target_tensor = target_tensor.repeat(inputs.size(0))
                loss_labels = target_tensor
        else:
            loss_labels = labels

        was_training = self.raw_model.training
        self.raw_model.eval()

        try:
            outputs = self.raw_model(inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = loss_fn(logits, loss_labels)

            grad = torch.autograd.grad(
                loss,
                inputs,
                retain_graph=False,
                create_graph=False
            )[0]

            if grad is None:
                raise AttackExecutionError(
                    "Gradients w.r.t input tensor were not computed."
                )

            grad_sign = grad.sign()

            if targeted:
                adv_inputs = inputs - epsilon * grad_sign
            else:
                adv_inputs = inputs + epsilon * grad_sign

            adv_inputs = clip_tensor(
                adv_inputs,
                clip_min,
                clip_max
            )

            return adv_inputs.detach()

        except AttackExecutionError:
            raise
        except Exception as e:
            raise AttackExecutionError(
                f"Error during FGSM attack generation: {str(e)}"
            ) from e

        finally:
            self.raw_model.train(was_training)

# Self-registration
register_attack("fgsm", FGSM)
