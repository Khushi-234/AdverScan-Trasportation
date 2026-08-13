"""
Fast Gradient Sign Method (FGSM) adversarial attack generator for AdverScan.
"""

from typing import Any
import torch
import torch.nn as nn

from app.attacks.base_attack import BaseAttack


class FGSMAttack(BaseAttack):
    """
    Fast Gradient Sign Method (FGSM) attack generator.
    x_adv = x + epsilon * sign(grad_x(Loss(model(x), y)))
    """

    def __init__(
        self,
        model_adapter: Any,
        epsilon: float = 0.03137,  # Default 8/255
        num_classes: int = 43,
        device: str = "cpu",
    ):
        """
        Initialize FGSM attack generator.

        Args:
            model_adapter: Module 1 model adapter instance or nn.Module.
            epsilon: Perturbation bound epsilon (default 8/255 ~ 0.03137).
            num_classes: Number of active target classes (e.g. 43 for GTSRB).
            device: Target execution device ('cuda' or 'cpu').
        """
        super().__init__(model_adapter, device=device)
        self.epsilon = epsilon
        self.num_classes = num_classes
        self.loss_fn = nn.CrossEntropyLoss()

    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Generate FGSM adversarial perturbations.

        Args:
            x: Input image tensor (batch_size, channels, H, W) on target device.
            y: Ground-truth target labels tensor (batch_size,) on target device.

        Returns:
            torch.Tensor: Perturbed adversarial image tensor on target device.
        """
        self.raw_model.eval()

        # Clone input tensor and enable gradient tracking
        x_adv = x.clone().detach().to(self.device).requires_grad_(True)
        targets = y.to(self.device)

        # Forward pass through PyTorch model
        outputs = self.raw_model(x_adv)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        # Slice logits to active target classes if necessary (e.g. 44 -> 43 for ViT)
        if logits.shape[-1] > self.num_classes:
            logits = logits[:, : self.num_classes]

        # Compute Cross-Entropy Loss w.r.t targets
        loss = self.loss_fn(logits, targets)

        # Zero gradients and compute backward pass w.r.t x_adv
        self.raw_model.zero_grad()
        loss.backward()

        # Compute sign of input gradient and apply perturbation
        if x_adv.grad is not None:
            gradient = x_adv.grad.data
            perturbation = self.epsilon * gradient.sign()
            x_adv = x.detach() + perturbation
        else:
            x_adv = x.detach()

        return x_adv
