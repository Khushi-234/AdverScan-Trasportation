"""
Projected Gradient Descent (PGD) adversarial attack implementation.
"""

from typing import Any
import torch
import torch.nn as nn

from app.attack_engine.base.base_attack import BaseAttack
from app.attack_engine.config import AttackConfig
from app.attack_engine.exceptions import AttackExecutionError
from app.attack_engine.attack_registry import register_attack


class PGD(BaseAttack):
    """
    Projected Gradient Descent (PGD) attack implementation.

    Iteratively updates adversarial perturbations using the gradient sign:
        x_0 = x + Uniform(-epsilon, epsilon) [if random_start]
        x_{t+1} = Clip_{x, epsilon}(x_t + alpha * sign(grad_{x_t}(Loss(model(x_t), y))))
    """

    def __init__(self, model: Any):
        """
        Initialize PGD attack.

        Args:
            model: PyTorch nn.Module or BaseModelAdapter wrapping PyTorch model.
        """
        super().__init__(model)
        self.raw_model = self._get_raw_model()
        if hasattr(self.raw_model, "eval"):
            self.raw_model.eval()

    def generate(self, inputs: Any, labels: Any, config: AttackConfig = None) -> Any:
        """
        Generate PGD adversarial examples.

        Args:
            inputs: Input tensor (e.g. torch.Tensor).
            labels: True class labels tensor.
            config: AttackConfig containing epsilon, clip_min, clip_max, loss_fn,
                    and params (num_steps/steps/iters, alpha/step_size, random_start).

        Returns:
            Adversarial input tensor.
        """
        if config is None:
            config = AttackConfig()

        if not isinstance(inputs, torch.Tensor):
            raise AttackExecutionError(
                f"PGD attack expects inputs to be torch.Tensor, got {type(inputs)}"
            )

        if not isinstance(labels, torch.Tensor):
            if isinstance(labels, (int, list, tuple)):
                labels = torch.tensor(labels)
            else:
                raise AttackExecutionError(
                    f"PGD attack expects labels to be torch.Tensor or iterable, got {type(labels)}"
                )

        epsilon = config.epsilon
        params = config.params or {}

        # Extract PGD specific parameters with sensible defaults
        num_steps = params.get("num_steps", params.get("steps", params.get("iters", 10)))
        random_start = params.get("random_start", True)

        if "alpha" in params:
            alpha = params["alpha"]
        elif "step_size" in params:
            alpha = params["step_size"]
        else:
            alpha = (2.0 * epsilon / num_steps) if num_steps > 0 else (epsilon / 4.0)

        # Device selection
        device = next(self.raw_model.parameters()).device if list(self.raw_model.parameters()) else torch.device("cpu")

        # Move inputs and labels to target device
        x_orig = inputs.clone().detach().to(device)
        labels = labels.to(device)

        clip_min = config.clip_min if config.clip_min is not None else float("-inf")
        clip_max = config.clip_max if config.clip_max is not None else float("inf")

        # Determine loss function
        if config.loss_fn is not None:
            loss_fn = config.loss_fn
        else:
            loss_fn = nn.CrossEntropyLoss()

        try:
            # Initialize adversarial inputs
            x_adv = x_orig.clone().detach()

            if random_start and epsilon > 0:
                # Add uniform noise in [-epsilon, epsilon]
                noise = torch.zeros_like(x_adv).uniform_(-epsilon, epsilon)
                x_adv = x_adv + noise
                if config.clip_min is not None or config.clip_max is not None:
                    x_adv = torch.clamp(x_adv, clip_min, clip_max)

            # Iterative gradient ascent steps
            for _ in range(num_steps):
                x_adv.requires_grad = True

                # Forward pass
                outputs = self.raw_model(x_adv)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = loss_fn(logits, labels)

                # Compute gradient w.r.t x_adv
                self.raw_model.zero_grad()
                grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]

                if grad is None:
                    raise AttackExecutionError("Gradients w.r.t input tensor were not computed during PGD step.")

                # Gradient step
                x_adv = x_adv.detach() + alpha * grad.sign()

                # Projection step: Clip to epsilon-ball around original input
                eta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
                x_adv = x_orig + eta

                # Clipping bounds
                if config.clip_min is not None or config.clip_max is not None:
                    x_adv = torch.clamp(x_adv, clip_min, clip_max)

            return x_adv.detach()

        except Exception as e:
            if isinstance(e, AttackExecutionError):
                raise e
            raise AttackExecutionError(f"Error during PGD attack generation: {str(e)}") from e


# Self-registration under both 'pgd' and 'pgm'
register_attack("pgd", PGD)
 