"""
Constraint-Aware Projected Gradient Descent (CapPGD) adversarial attack for tabular data.

Implements iterative PGD with projection after every update onto the tabular constraint manifold:
- Continuous bounds [min, max]
- Immutable feature preservation
- Integer rounding and constraints
- Binary constraints {0, 1}
- Discrete categorical projections
- Feature-specific perturbation limits
"""

from typing import Any, Optional
import torch
import torch.nn as nn

from app.attack_engine.attacks.base_attack import BaseAttack
from app.attack_engine.models import AttackMetadata
from app.attack_engine.config import AttackConfig
from app.attack_engine.exceptions import (
    AttackConfigurationError,
    AttackExecutionError,
)
from app.attack_engine.attack_registry import register_attack
from app.attack_engine.utils.tensor_utils import get_device
from app.attack_engine.attacks.tabular._tabular_utils import (
    get_raw_model,
    parse_tabular_metadata,
    validate_tabular_inputs,
)


class CapPGD(BaseAttack):
    """
    Constraint-Aware Projected Gradient Descent (CapPGD) attack for tabular data.
    """

    metadata = AttackMetadata(
        name="cap_pgd",
        domain="tabular",
        category="gradient",
        attack_type="white_box",
        requires_gradient=True,
        requires_loss=True,
        requires_model_weights=True,
        iterative=True,
        targeted_supported=True,
        untargeted_supported=True,
        query_based=False,
        supported_norms=["Linf", "L2"],
        computational_cost="medium",
        supported_model_types=["pytorch"],
        parameters={
            "constraint_aware": True,
            "supports_categorical": True,
            "supports_integer": True,
            "supports_continuous": True,
        },
    )

    def __init__(self, model: Any):
        super().__init__(model)
        self.raw_model = get_raw_model(self.model)

    def validate_config(self, config: Optional[AttackConfig] = None) -> None:
        """
        Validate CapPGD configuration parameters.
        """
        if config is None:
            return

        params = config.params or {}

        epsilon = params.get("epsilon", config.epsilon)
        if epsilon < 0:
            raise AttackConfigurationError(f"Epsilon cannot be negative, got {epsilon}")

        alpha = params.get("alpha", params.get("step_size", 0.05))
        if alpha <= 0:
            raise AttackConfigurationError(
                f"Step size (alpha) must be positive, got {alpha}"
            )

        num_steps = params.get(
            "num_steps", params.get("steps", params.get("iterations", 10))
        )
        if not isinstance(num_steps, int) or num_steps <= 0:
            raise AttackConfigurationError(
                f"num_steps must be a positive integer, got {num_steps}"
            )

        norm = str(params.get("norm", "Linf")).upper()
        if norm not in ("LINF", "INF", "L2", "2"):
            raise AttackConfigurationError(
                f"Unsupported norm '{norm}'. Supported norms: ['Linf', 'L2']"
            )

    def generate(
        self, inputs: Any, labels: Any, config: Optional[AttackConfig] = None
    ) -> torch.Tensor:
        """
        Generate constraint-aware PGD adversarial examples on tabular data.

        Args:
            inputs: Tabular tensor of shape (batch_size, num_features).
            labels: Ground truth labels (or target labels if targeted).
            config: Optional AttackConfig.

        Returns:
            Adversarial tensor matching inputs shape satisfying tabular constraints.
        """
        if config is None:
            config = AttackConfig()

        self.validate_config(config)
        params = config.params or {}

        epsilon = float(params.get("epsilon", config.epsilon))
        alpha = float(params.get("alpha", params.get("step_size", 0.05)))
        num_steps = int(
            params.get("num_steps", params.get("steps", params.get("iterations", 10)))
        )
        norm = str(params.get("norm", "Linf")).upper()
        random_init = bool(params.get("random_init", False))
        targeted = bool(params.get("targeted", False))
        target_label = params.get("target_label", None)
        loss_fn = config.loss_fn or nn.CrossEntropyLoss()

        device = get_device(self.raw_model)
        x_orig, y = validate_tabular_inputs(inputs, labels, attack_name="CapPGD")
        x_orig = x_orig.to(device)
        y = y.to(device)

        batch_size, num_features = x_orig.shape

        if targeted:
            if target_label is not None:
                if isinstance(target_label, int):
                    y_target = torch.full(
                        (batch_size,), target_label, dtype=torch.long, device=device
                    )
                else:
                    y_target = torch.as_tensor(
                        target_label, dtype=torch.long, device=device
                    )
                    if y_target.dim() == 0:
                        y_target = y_target.repeat(batch_size)
            else:
                y_target = y
            labels_eval = y_target
        else:
            labels_eval = y

        # Parse tabular feature metadata
        metadata = parse_tabular_metadata(self.model, params, num_features=num_features)
        mutable_mask = metadata.get_mutable_mask(device)  # shape (num_features,)

        clip_min = params.get(
            "clip_min",
            (
                config.clip_min
                if (
                    "clip_min" in params
                    or config.clip_min != 0.0
                    or config.clip_max != 1.0
                )
                else None
            ),
        )
        clip_max = params.get(
            "clip_max",
            (
                config.clip_max
                if (
                    "clip_max" in params
                    or config.clip_min != 0.0
                    or config.clip_max != 1.0
                )
                else None
            ),
        )

        was_training = self.raw_model.training
        self.raw_model.eval()

        # Initialize adversarial tensor
        x_adv = x_orig.clone()

        if random_init and epsilon > 0:
            if norm in ("LINF", "INF"):
                noise = (torch.rand_like(x_adv) * 2.0 - 1.0) * epsilon
            else:
                noise = torch.randn_like(x_adv)
                noise_norm = torch.norm(noise, p=2, dim=1, keepdim=True) + 1e-12
                noise = noise * (
                    torch.rand(batch_size, 1, device=device) * epsilon / noise_norm
                )

            # Only apply noise to mutable features
            noise = noise * mutable_mask.unsqueeze(0).float()
            x_adv = x_adv + noise
            x_adv = metadata.project(
                x_adv, x_orig, clip_min=clip_min, clip_max=clip_max
            )

        for step in range(num_steps):
            x_adv.requires_grad_(True)
            outputs = self.raw_model(x_adv)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            if targeted:
                # Minimize loss toward target class
                loss = -loss_fn(logits, labels_eval)
            else:
                # Maximize loss away from true class
                loss = loss_fn(logits, labels_eval)

            grad = torch.autograd.grad(
                loss,
                x_adv,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]

            if grad is None:
                raise AttackExecutionError(
                    "CapPGD failed to compute gradients w.r.t input features."
                )

            with torch.no_grad():
                # Gradient ascent step
                if norm in ("LINF", "INF"):
                    step_update = alpha * grad.sign()
                else:  # L2
                    grad_norm = torch.norm(grad, p=2, dim=1, keepdim=True) + 1e-12
                    step_update = alpha * (grad / grad_norm)

                # Zero out updates for immutable features
                step_update = step_update * mutable_mask.unsqueeze(0).float()
                x_adv = x_adv + step_update

                # Norm projection onto epsilon-ball around x_orig
                delta = x_adv - x_orig
                if norm in ("LINF", "INF"):
                    delta = torch.clamp(delta, -epsilon, epsilon)
                else:
                    delta_norm = torch.norm(delta, p=2, dim=1, keepdim=True) + 1e-12
                    factor = torch.clamp(epsilon / delta_norm, max=1.0)
                    delta = delta * factor

                x_adv = x_orig + delta

                # Tabular manifold constraint projection
                x_adv = metadata.project(
                    x_adv=x_adv,
                    x_orig=x_orig,
                    clip_min=clip_min,
                    clip_max=clip_max,
                )

        self.raw_model.train(was_training)

        # Final guarantee of constraints
        return metadata.project(
            x_adv, x_orig, clip_min=clip_min, clip_max=clip_max
        ).detach()


# Self-register
register_attack("cap_pgd", CapPGD)
register_attack("tabular_pgd", CapPGD)
