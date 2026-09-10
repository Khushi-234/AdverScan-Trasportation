"""
DeepFool adversarial attack implementation for minimal decision boundary perturbations.
                 Original image
                       │
                       ▼
                    x_i = x
                       │
                       ▼
                Model prediction
                       │
                       ▼
              Is prediction changed?
                 /             \
               YES              NO
                │                │
                ▼                ▼
              STOP        Calculate gradient
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
             Current class             Candidate classes
                    │                           │
                    ▼                           ▼
               grad_orig                  grad_k for each
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                         w_k = grad_k - grad_orig
                                  │
                                  ▼
                    Calculate boundary distance
                                  │
                                  ▼
                      Find closest boundary
                                  │
                                  ▼
                    Calculate minimal r_i
                                  │
                                  ▼
                         Add overshoot
                                  │
                                  ▼
                         x_i = x_i + delta
                                  │
                                  ▼
                       Epsilon constraint
                                  │
                                  ▼
                         Input clipping
                                  │
                                  ▼
                         Next iteration
                                  │
                                  └───────────────┐
                                                  │
                                                  ▼
                                         Check prediction
"""

from typing import Any, Optional
import torch
import torch.nn as nn

from app.attack_engine.base.base_attack import BaseAttack
from app.attack_engine.base.attack_metadata import AttackMetadata
from app.attack_engine.config import AttackConfig
from app.attack_engine.exceptions import AttackExecutionError
from app.attack_engine.registry import register_attack
from app.attack_engine.utils.tensor_utils import get_device, clip_tensor
from app.attack_engine.utils.validation import validate_inputs, validate_attack_config


class DeepFool(BaseAttack):
    """
    DeepFool adversarial attack implementation for computing minimal decision boundary perturbations.
    """

    metadata = AttackMetadata(
        name="deepfool",
        domain="image",
        category="boundary",
        attack_type="white_box",
        requires_gradient=True,
        requires_loss=False,
        requires_model_weights=True,
        iterative=True,
        targeted_supported=False,
        untargeted_supported=True,
        query_based=False,
        supported_norms=["L2", "Linf"],
        computational_cost="medium",
        supported_model_types=["pytorch"],
    )

    def __init__(self, model: Any):
        """
        Initialize DeepFool attack.

        Args:
            model: PyTorch nn.Module or BaseModelAdapter wrapping PyTorch model.
        """
        super().__init__(model)
        self.raw_model = self._get_raw_model()

    def validate_config(self, config: Optional[AttackConfig] = None) -> None:
        """
        Validate DeepFool-specific configuration.
        """
        validate_attack_config(config)
        if config is not None and config.params:
            params = config.params
            max_iter = params.get("max_iter", params.get("num_steps", params.get("steps")))
            if max_iter is not None and (not isinstance(max_iter, int) or max_iter <= 0):
                raise AttackExecutionError("DeepFool max_iter must be greater than 0.")
            top_k = params.get("top_k")
            if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
                raise AttackExecutionError("DeepFool top_k must be greater than 0.")

    def generate(self, inputs: Any, labels: Any, config: Optional[AttackConfig] = None) -> Any:
        """
        Generate DeepFool adversarial examples.

        Args:
            inputs: Input tensor (e.g. torch.Tensor).
            labels: True class labels tensor.
            config: AttackConfig containing clip_min, clip_max, and params (num_steps, overshoot).

        Returns:
            Adversarial input tensor.
        """
        if config is None:
            config = AttackConfig()

        self.validate_config(config)
        validate_inputs(inputs, labels, attack_name="DeepFool")

        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels)

        params = config.params or {}
        max_iter = params.get("max_iter", params.get("num_steps", params.get("steps", 10)))
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise AttackExecutionError("DeepFool max_iter must be greater than 0.")

        top_k = params.get("top_k", 10)
        if not isinstance(top_k, int) or top_k <= 0:
            raise AttackExecutionError("DeepFool top_k must be greater than 0.")

        overshoot = params.get("overshoot", 0.02)
        epsilon = config.epsilon if config.epsilon is not None else float("inf")

        # Device selection
        device = get_device(self.raw_model)

        # Move inputs to target device
        x_orig = inputs.clone().detach().to(device)

        was_training = self.raw_model.training
        self.raw_model.eval()

        try:
            adv_samples = []
            for idx in range(x_orig.size(0)):
                x_i = x_orig[idx : idx + 1].clone().detach()
                x_start = x_orig[idx : idx + 1].clone().detach()

                for _ in range(max_iter):
                    x_i.requires_grad_(True)
                    outputs = self.raw_model(x_i)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs

                    num_classes = logits.size(-1)
                    current_pred = logits.argmax(dim=-1).item()
                    original_label = labels[idx].item() if idx < len(labels) else current_pred

                    if current_pred != original_label:
                        break

                    # Compute gradients for original predicted class
                    grad_orig = torch.autograd.grad(
                        logits[0, current_pred],
                        x_i,
                        retain_graph=True,
                        create_graph=False,
                    )[0].clone()

                    min_dist = float("inf")
                    best_w = None
                    best_f = None

                    # Top-K candidate class selection for efficient decision boundary calculation
                    k_limit = min(top_k, num_classes)
                    _, top_indices = torch.topk(logits[0], k=k_limit)
                    candidate_classes = top_indices.tolist()

                    # Find closest decision boundary across top candidate classes
                    for k in candidate_classes:
                        if k == current_pred:
                            continue

                        grad_k = torch.autograd.grad(
                            logits[0, k],
                            x_i,
                            retain_graph=True,
                            create_graph=False,
                        )[0].clone()

                        w_k = grad_k - grad_orig
                        f_k = (logits[0, k] - logits[0, current_pred]).item()

                        w_norm = torch.norm(w_k.view(-1), p=2).item()
                        if w_norm == 0:
                            continue

                        dist = abs(f_k) / (w_norm + 1e-8)
                        if dist < min_dist:
                            min_dist = dist
                            best_w = w_k
                            best_f = f_k

                    if best_w is None:
                        break

                    # Compute minimal perturbation step
                    w_norm_sq = (torch.norm(best_w.view(-1), p=2).item()) ** 2 + 1e-8
                    r_i = (abs(best_f) / w_norm_sq) * best_w
                    delta = (1.0 + overshoot) * r_i

                    x_i = x_i.detach() + delta

                    # Check epsilon ball constraints if epsilon is finite
                    if epsilon < float("inf"):
                        perturbation = torch.clamp(x_i - x_start, min=-epsilon, max=epsilon)
                        x_i = x_start + perturbation

                    # Apply clipping bounds
                    x_i = clip_tensor(x_i, config.clip_min, config.clip_max)

                adv_samples.append(x_i.detach())

            adv_tensor = torch.cat(adv_samples, dim=0)
            return adv_tensor

        except AttackExecutionError:
            raise
        except Exception as e:
            raise AttackExecutionError(f"Error during DeepFool attack generation: {str(e)}") from e
        finally:
            self.raw_model.train(was_training)


# Self-registration
register_attack("deepfool", DeepFool)
