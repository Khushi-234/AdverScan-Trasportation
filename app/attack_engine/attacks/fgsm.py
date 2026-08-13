"""
Fast Gradient Sign Method (FGSM) adversarial attack implementation.
"""

from typing import Any
import torch
import torch.nn as nn

from app.attack_engine.base.base_attack import BaseAttack
from app.attack_engine.config.attack_config import AttackConfig
from app.attack_engine.exceptions import AttackExecutionError


class FGSM(BaseAttack):
    """
    Fast Gradient Sign Method (FGSM) attack implementation.

    Perturbs input tensor according to:
        x_adv = x + epsilon * sign(grad_x(Loss(model(x), y)))
    """

    def __init__(self, model: Any):
        """
        Initialize FGSM attack.

        Args:
            model: PyTorch nn.Module or BaseModelAdapter wrapping PyTorch model.
        """
        super().__init__(model)
        self.raw_model = self._get_raw_model()
        if hasattr(self.raw_model, "eval"):
            self.raw_model.eval()

    def generate(self, inputs: Any, labels: Any, config: AttackConfig = None) -> Any:
        """
        Generate FGSM adversarial examples.

        Args:
            inputs: Input tensor (e.g. torch.Tensor).
            labels: True class labels tensor.
            config: AttackConfig containing epsilon, clip_min, clip_max, loss_fn.

        Returns:
            Adversarial input tensor.
        """
        if config is None:
            config = AttackConfig()

        if not isinstance(inputs, torch.Tensor):
            raise AttackExecutionError(
                f"FGSM attack expects inputs to be torch.Tensor, got {type(inputs)}"
            )

        if not isinstance(labels, torch.Tensor):
            if isinstance(labels, (int, list, tuple)):
                labels = torch.tensor(labels)
            else:
                raise AttackExecutionError(
                    f"FGSM attack expects labels to be torch.Tensor or iterable, got {type(labels)}"
                )

        device = next(self.raw_model.parameters()).device if list(self.raw_model.parameters()) else torch.device("cpu")

        # Move inputs and labels to target device
        inputs = inputs.clone().detach().to(device)
        labels = labels.to(device)
        inputs.requires_grad = True

        # Determine loss function
        if config.loss_fn is not None:
            loss_fn = config.loss_fn
        else:
            loss_fn = nn.CrossEntropyLoss()

        try:
            # Forward pass
            outputs = self.raw_model(inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = loss_fn(logits, labels)

            # Zero gradients
            self.raw_model.zero_grad()
            if inputs.grad is not None:
                inputs.grad.zero_()

            # Backward pass
            loss.backward()

            if inputs.grad is None:
                raise AttackExecutionError("Gradients w.r.t input tensor were not computed.")

            # Compute perturbation
            epsilon = config.epsilon
            grad_sign = inputs.grad.sign()
            adv_inputs = inputs + epsilon * grad_sign

            # Clip adversarial examples if bounds are specified
            if config.clip_min is not None or config.clip_max is not None:
                clip_min = config.clip_min if config.clip_min is not None else float("-inf")
                clip_max = config.clip_max if config.clip_max is not None else float("inf")
                adv_inputs = torch.clamp(adv_inputs, clip_min, clip_max)

            return adv_inputs.detach()

        except Exception as e:
            if isinstance(e, AttackExecutionError):
                raise e
            raise AttackExecutionError(f"Error during FGSM attack generation: {str(e)}") from e
