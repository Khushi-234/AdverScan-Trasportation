"""
Projected Gradient Descent (PGD) adversarial attack implementation.
"""

from typing import Any, Optional
import torch
import torch.nn as nn

from app.attack_engine.attacks.base_attack import BaseAttack
from app.attack_engine.models import AttackMetadata
from app.attack_engine.config import AttackConfig
from app.attack_engine.exceptions import AttackExecutionError
from app.attack_engine.attack_registry import register_attack
from app.attack_engine.utils.tensor_utils import get_device, clip_tensor, project_lp
from app.attack_engine.utils.validation import validate_inputs, validate_attack_config


class PGD(BaseAttack):
    """
    Projected Gradient Descent (PGD) attack implementation.

    Iteratively updates adversarial perturbations using the gradient sign:
        x_0 = x + Uniform(-epsilon, epsilon) [if random_start]
        x_{t+1} = Proj_{x, epsilon}(x_t + alpha * sign(grad_{x_t}(Loss(model(x_t), y))))
    """

    metadata = AttackMetadata(
        name="pgd",
        domain="image",
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
    )

    def __init__(self, model: Any):
        """
        Initialize PGD attack.

        Args:
            model: PyTorch nn.Module or BaseModelAdapter wrapping PyTorch model.
        """
        super().__init__(model)
        self.raw_model = self._get_raw_model()

    def validate_config(self, config: Optional[AttackConfig] = None) -> None:
        """
        Validate PGD-specific configuration.
        """
        validate_attack_config(config)

    def generate(self, inputs: Any, labels: Any, config: Optional[AttackConfig] = None) -> Any:
        """
        Generate PGD adversarial examples.

        Args:
            inputs: Input tensor (e.g. torch.Tensor).
            labels: True class labels tensor (or target labels if targeted=True).
            config: AttackConfig containing epsilon, clip_min, clip_max, loss_fn,
                    and params (num_steps/steps/iters, alpha/step_size, random_start, targeted).

        Returns:
            Adversarial input tensor.
        """
        if config is None:
            config = AttackConfig()

        self.validate_config(config)
        validate_inputs(inputs, labels, attack_name="PGD")

        epsilon = config.epsilon
        params = config.params or {}

        # Extract PGD specific parameters with sensible defaults
        num_steps = params.get("num_steps", params.get("steps", params.get("iters", 10)))
        if num_steps <= 0:
            raise AttackExecutionError("PGD num_steps must be greater than 0.")

        random_start = params.get("random_start", True)
        targeted = params.get("targeted", False)
        norm = params.get("norm", "Linf")

        if "alpha" in params:
            alpha = params["alpha"]
        elif "step_size" in params:
            alpha = params["step_size"]
        else:
            alpha = 2.0 * epsilon / num_steps

        # Device selection
        device = get_device(self.raw_model)

        # Move inputs and labels to target device
        x_orig = inputs.clone().detach().to(device)
        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels)
        labels = labels.to(device)

        # Determine loss function
        if config.loss_fn is not None:
            loss_fn = config.loss_fn
        else:
            loss_fn = nn.CrossEntropyLoss()

        was_training = self.raw_model.training
        self.raw_model.eval()

        try:
            # Initialize adversarial inputs
            x_adv = x_orig.clone().detach()

            if random_start and epsilon > 0:
                if norm.upper() in ("LINF", "INF"):
                    noise = torch.zeros_like(x_adv).uniform_(-epsilon, epsilon)
                else:
                    noise = torch.randn_like(x_adv)
                    noise = noise / (
                        torch.norm(noise.view(noise.size(0), -1), p=2, dim=1, keepdim=True)
                        .view(-1, *([1] * (noise.dim() - 1))) + 1e-12
                    )
                    radius = torch.rand(
                        x_adv.size(0),
                        device=device
                    ).view(-1, *([1] * (x_adv.dim() - 1))) * epsilon
                    noise = noise * radius
                x_adv = x_adv + noise
                x_adv = clip_tensor(x_adv, config.clip_min, config.clip_max)

            # Iterative gradient ascent/descent steps
            for _ in range(num_steps):
                x_adv.requires_grad_(True)

                # Forward pass
                outputs = self.raw_model(x_adv)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = loss_fn(logits, labels)

                # Compute gradient w.r.t x_adv
                grad = torch.autograd.grad(
                    loss,
                    x_adv,
                    retain_graph=False,
                    create_graph=False,
                )[0]

                if grad is None:
                    raise AttackExecutionError("Gradients w.r.t input tensor were not computed during PGD step.")

                # Gradient step
                if targeted:
                    step_dir = -grad.sign()
                else:
                    step_dir = grad.sign()

                x_adv = x_adv.detach() + alpha * step_dir

                # Projection step: Clip to epsilon-ball around original input
                x_adv = project_lp(x_adv, x_orig, epsilon=epsilon, norm=norm)

                # Clipping bounds
                x_adv = clip_tensor(x_adv, config.clip_min, config.clip_max)

            return x_adv.detach()

        except AttackExecutionError:
            raise
        except Exception as e:
            raise AttackExecutionError(f"Error during PGD attack generation: {str(e)}") from e
        finally:
            self.raw_model.train(was_training)


# Self-registration
register_attack("pgd", PGD)
