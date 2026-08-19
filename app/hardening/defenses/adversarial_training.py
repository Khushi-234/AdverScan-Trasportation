"""
Adversarial Training Defense Module for Module 7 (Hardening).

Implements adversarial training / fine-tuning of PyTorch models using FGSM or PGD adversarial batch generation.
"""

import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

from app.hardening.defenses.base import BaseDefense
from app.hardening.hardening_result import HardeningMetadata, HardeningResult
from app.hardening.utils import clone_model
from app.hardening.exceptions import DefenseExecutionError, HardeningConfigurationError


def generate_fgsm_batch(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.05,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    criterion: Optional[nn.Module] = None,
) -> torch.Tensor:
    """
    Generate FGSM adversarial perturbations for a batch during training.
    """
    criterion = criterion or nn.CrossEntropyLoss()
    x_adv = inputs.clone().detach().requires_grad_(True)
    outputs = model(x_adv)
    loss = criterion(outputs, labels)

    model.zero_grad()
    loss.backward()

    if x_adv.grad is not None:
        grad = x_adv.grad.data
        x_adv = x_adv + epsilon * torch.sign(grad)
        x_adv = torch.clamp(x_adv, min=clip_min, max=clip_max)

    return x_adv.detach()


def generate_pgd_batch(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.05,
    alpha: float = 0.01,
    steps: int = 5,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    criterion: Optional[nn.Module] = None,
) -> torch.Tensor:
    """
    Generate PGD adversarial perturbations for a batch during training.
    """
    criterion = criterion or nn.CrossEntropyLoss()
    x_original = inputs.clone().detach()
    x_adv = inputs.clone().detach() + torch.randn_like(inputs) * (epsilon * 0.5)
    x_adv = torch.clamp(x_adv, min=clip_min, max=clip_max)

    for _ in range(steps):
        x_adv.requires_grad = True
        outputs = model(x_adv)
        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()

        if x_adv.grad is not None:
            with torch.no_grad():
                grad = x_adv.grad.data
                x_adv = x_adv + alpha * torch.sign(grad)
                # Projection back into L-infinity epsilon ball
                eta = torch.clamp(x_adv - x_original, min=-epsilon, max=epsilon)
                x_adv = torch.clamp(x_original + eta, min=clip_min, max=clip_max)

    return x_adv.detach()


class AdversarialTrainingDefense(BaseDefense):
    """
    Adversarial Training Defense.
    Fine-tunes PyTorch models on adversarial batches (FGSM or PGD) to embed robustness directly into weights.
    """

    def __init__(
        self,
        epochs: int = 2,
        lr: float = 1e-4,
        epsilon: float = 0.05,
        alpha: float = 0.01,
        attack_type: str = "pgd",
        attack_steps: int = 5,
        ratio_adv: float = 0.5,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name="adversarial_training", defense_type="adversarial_training", config=config)
        self.epochs = epochs
        self.lr = lr
        self.epsilon = epsilon
        self.alpha = alpha
        self.attack_type = attack_type.lower()
        self.attack_steps = attack_steps
        self.ratio_adv = ratio_adv
        self.clip_min = clip_min
        self.clip_max = clip_max

    def train_epoch(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """
        Execute one fine-tuning epoch on input/label batch.
        """
        model.train()
        optimizer.zero_grad()

        # Generate adversarial batch
        if self.attack_type == "fgsm":
            adv_inputs = generate_fgsm_batch(
                model=model,
                inputs=inputs,
                labels=labels,
                epsilon=self.epsilon,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
                criterion=criterion,
            )
        else:
            adv_inputs = generate_pgd_batch(
                model=model,
                inputs=inputs,
                labels=labels,
                epsilon=self.epsilon,
                alpha=self.alpha,
                steps=self.attack_steps,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
                criterion=criterion,
            )

        # Combine clean and adversarial batch according to ratio_adv
        if self.ratio_adv >= 1.0:
            train_inputs = adv_inputs
        elif self.ratio_adv <= 0.0:
            train_inputs = inputs
        else:
            num_adv = int(inputs.shape[0] * self.ratio_adv)
            train_inputs = torch.cat([adv_inputs[:num_adv], inputs[num_adv:]], dim=0)

        # Forward & Backprop
        outputs = model(train_inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        return float(loss.item())

    def apply(
        self,
        model: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> HardeningResult:
        start_time = time.time()
        try:
            if inputs is None or labels is None:
                raise HardeningConfigurationError(
                    "AdversarialTrainingDefense requires both 'inputs' and 'labels' tensors for fine-tuning."
                )

            # Clone model to avoid modifying original model in-place unexpectedly
            hardened_model = clone_model(model)
            optimizer = optim.Adam(hardened_model.parameters(), lr=self.lr)
            criterion = nn.CrossEntropyLoss()

            losses = []
            for epoch in range(self.epochs):
                loss_val = self.train_epoch(
                    model=hardened_model,
                    inputs=inputs,
                    labels=labels,
                    optimizer=optimizer,
                    criterion=criterion,
                )
                losses.append(loss_val)

            hardened_model.eval()

            exec_time = time.time() - start_time
            meta = HardeningMetadata(
                defense_name=self.name,
                defense_type=self.defense_type,
                parameters={
                    "epochs": self.epochs,
                    "lr": self.lr,
                    "epsilon": self.epsilon,
                    "alpha": self.alpha,
                    "attack_type": self.attack_type,
                    "attack_steps": self.attack_steps,
                    "ratio_adv": self.ratio_adv,
                    "losses": losses,
                },
                execution_time_seconds=exec_time,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            return HardeningResult(
                hardened_model=hardened_model,
                metadata=meta,
                hardened_inputs=inputs,
                success=True,
                recommendations=[
                    f"Fine-tuned model weights with {self.epochs} epoch(s) of {self.attack_type.upper()} adversarial training.",
                    "Adversarial training modifies internal network parameters to resist perturbation gradients.",
                ],
            )
        except Exception as e:
            if isinstance(e, HardeningConfigurationError):
                raise
            raise DefenseExecutionError(f"AdversarialTrainingDefense failed: {str(e)}") from e
