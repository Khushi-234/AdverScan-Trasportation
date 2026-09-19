"""
Carlini & Wagner (C&W) L2 optimization-based adversarial attack implementation.
"""

from typing import Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim

from app.attack_engine.attacks.base_attack import BaseAttack
from app.attack_engine.models import AttackMetadata
from app.attack_engine.config import AttackConfig
from app.attack_engine.exceptions import AttackExecutionError
from app.attack_engine.attack_registry import register_attack
from app.attack_engine.utils.tensor_utils import get_device, clip_tensor
from app.attack_engine.utils.validation import validate_inputs, validate_attack_config


class CW(BaseAttack):
    """
    Carlini & Wagner (C&W) L2 optimization attack.

    Formulates adversarial example generation as minimizing:
        ||x_adv - x||_2^2 + c * f(x_adv)
    subject to box constraints in [clip_min, clip_max] via a tanh change-of-variables.
    """

    metadata = AttackMetadata(
        name="cw",
        domain="image",
        category="optimization",
        attack_type="white_box",
        requires_gradient=True,
        requires_loss=True,
        requires_model_weights=True,
        iterative=True,
        targeted_supported=True,
        untargeted_supported=True,
        query_based=False,
        supported_norms=["L2"],
        computational_cost="high",
        supported_model_types=["pytorch"],
    )

    def __init__(self, model: Any):
        """
        Initialize C&W attack.

        Args:
            model: PyTorch nn.Module or BaseModelAdapter wrapping PyTorch model.
        """
        super().__init__(model)
        self.raw_model = self._get_raw_model()

    def validate_config(self, config: Optional[AttackConfig] = None) -> None:
        """
        Validate C&W-specific configuration.
        """
        validate_attack_config(config)

    def _to_tanh_space(
        self, x: torch.Tensor, clip_min: float, clip_max: float
    ) -> torch.Tensor:
        """
        Map tensor from [clip_min, clip_max] to unconstrained tanh space (-inf, +inf).
        """
        box_mul = (clip_max - clip_min) / 2.0
        box_plus = (clip_min + clip_max) / 2.0
        val = (x - box_plus) / box_mul
        val = torch.clamp(val, -0.999999, 0.999999)
        return torch.atanh(val)

    def _from_tanh_space(
        self, w: torch.Tensor, clip_min: float, clip_max: float
    ) -> torch.Tensor:
        """
        Map tensor from unconstrained tanh space (-inf, +inf) back to [clip_min, clip_max].
        """
        box_mul = (clip_max - clip_min) / 2.0
        box_plus = (clip_min + clip_max) / 2.0
        return torch.tanh(w) * box_mul + box_plus

    def generate(
        self, inputs: Any, labels: Any, config: Optional[AttackConfig] = None
    ) -> Any:
        """
        Generate C&W L2 adversarial examples.

        Args:
            inputs: Input tensor (e.g. torch.Tensor).
            labels: True class labels tensor (or target labels if targeted=True).
            config: AttackConfig containing clip_min, clip_max, and params:
                - max_iterations / iters: Maximum optimization steps (default 20).
                - learning_rate / lr: Optimization learning rate (default 0.01).
                - confidence / kappa: Margin parameter kappa (default 0.0).
                - binary_search_steps: Binary search steps for constant c (default 1).
                - initial_const / c: Initial trade-off constant c (default 1.0).
                - targeted: Boolean flag indicating targeted mode (default False).

        Returns:
            Adversarial input tensor.
        """
        if config is None:
            config = AttackConfig()

        self.validate_config(config)
        validate_inputs(inputs, labels, attack_name="CW")

        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels)

        params = config.params or {}
        max_iter = params.get("max_iterations", params.get("iters", params.get("num_steps", 20)))
        lr = params.get("learning_rate", params.get("lr", 0.01))
        confidence = params.get("confidence", params.get("kappa", 0.0))
        binary_search_steps = params.get("binary_search_steps", 1)
        initial_const = params.get("initial_const", params.get("c", 1.0))
        targeted = params.get("targeted", False)

        clip_min = config.clip_min if config.clip_min is not None else 0.0
        clip_max = config.clip_max if config.clip_max is not None else 1.0

        device = get_device(self.raw_model)
        x_orig = inputs.clone().detach().to(device)
        labels = labels.to(device)
        batch_size = x_orig.size(0)

        was_training = self.raw_model.training
        self.raw_model.eval()

        try:
            # Overall best adversarial examples
            best_adv = x_orig.clone().detach()
            best_l2 = torch.full((batch_size,), float("inf"), device=device)

            # Lower and upper bounds for constant c in binary search
            c_lower = torch.zeros(batch_size, device=device)
            c_upper = torch.full((batch_size,), float("inf"), device=device)
            c_val = torch.full((batch_size,), initial_const, device=device)

            for _ in range(binary_search_steps):
                # Map inputs to tanh space
                w = self._to_tanh_space(x_orig, clip_min, clip_max).clone().detach()
                w.requires_grad_(True)

                optimizer = optim.Adam([w], lr=lr)
                step_success = torch.zeros(batch_size, dtype=torch.bool, device=device)

                for _ in range(max_iter):
                    optimizer.zero_grad()
                    x_adv = self._from_tanh_space(w, clip_min, clip_max)

                    # L2 distance term: ||x_adv - x||_2^2
                    delta = (x_adv - x_orig).view(batch_size, -1)
                    l2_dist = torch.sum(delta ** 2, dim=1)

                    # Forward pass
                    outputs = self.raw_model(x_adv)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs

                    num_classes = logits.size(-1)
                    one_hot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1.0)

                    real_logits = torch.sum(one_hot * logits, dim=1)
                    other_logits = torch.max((1.0 - one_hot) * logits - one_hot * 1e4, dim=1)[0]

                    if targeted:
                        # Target class should have higher logit than others
                        loss_f = torch.clamp(other_logits - real_logits + confidence, min=0.0)
                        successful = (logits.argmax(dim=1) == labels)
                    else:
                        # Other classes should have higher logit than true class
                        loss_f = torch.clamp(real_logits - other_logits + confidence, min=0.0)
                        successful = (logits.argmax(dim=1) != labels)

                    loss = torch.sum(l2_dist + c_val * loss_f)
                    grad_w = torch.autograd.grad(loss, w, retain_graph=False, create_graph=False)[0]
                    w.grad = grad_w
                    optimizer.step()

                    # Update best adversarial examples for samples that achieved attack success
                    with torch.no_grad():
                        for b in range(batch_size):
                            if successful[b]:
                                step_success[b] = True
                                if l2_dist[b] < best_l2[b]:
                                    best_l2[b] = l2_dist[b]
                                    best_adv[b] = x_adv[b].clone().detach()

                # Update binary search bounds for constant c
                for b in range(batch_size):
                    if step_success[b]:
                        c_upper[b] = min(c_upper[b], c_val[b])
                        if c_upper[b] < 1e9:
                            c_val[b] = (c_lower[b] + c_upper[b]) / 2.0
                    else:
                        c_lower[b] = max(c_lower[b], c_val[b])
                        if c_upper[b] < 1e9:
                            c_val[b] = (c_lower[b] + c_upper[b]) / 2.0
                        else:
                            c_val[b] = c_val[b] * 10.0

            # Fallback: if no successful adversarial example was found, return latest x_adv clipped
            with torch.no_grad():
                unsuccessful = (best_l2 == float("inf"))
                if unsuccessful.any():
                    latest_adv = self._from_tanh_space(w, clip_min, clip_max).detach()
                    best_adv[unsuccessful] = latest_adv[unsuccessful]

            best_adv = clip_tensor(best_adv, clip_min, clip_max)
            return best_adv.detach()

        except AttackExecutionError:
            raise
        except Exception as e:
            raise AttackExecutionError(f"Error during C&W attack generation: {str(e)}") from e
        finally:
            self.raw_model.train(was_training)


# Self-registration
register_attack("cw", CW)
register_attack("carlini_wagner", CW)
