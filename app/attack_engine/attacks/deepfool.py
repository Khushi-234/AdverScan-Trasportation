"""
DeepFool adversarial attack implementation.
"""

from typing import Any
import torch
import torch.nn as nn

from app.attack_engine.base.base_attack import BaseAttack
from app.attack_engine.config import AttackConfig
from app.attack_engine.exceptions import AttackExecutionError
from app.attack_engine.attack_registry import register_attack


class DeepFool(BaseAttack):
    """
    DeepFool adversarial attack implementation for computing minimal decision boundary perturbations.
    """

    def __init__(self, model: Any):
        """
        Initialize DeepFool attack.

        Args:
            model: PyTorch nn.Module or BaseModelAdapter wrapping PyTorch model.
        """
        super().__init__(model)
        self.raw_model = self._get_raw_model()
        if hasattr(self.raw_model, "eval"):
            self.raw_model.eval()

    def generate(self, inputs: Any, labels: Any, config: AttackConfig = None) -> Any:
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

        if not isinstance(inputs, torch.Tensor):
            raise AttackExecutionError(
                f"DeepFool attack expects inputs to be torch.Tensor, got {type(inputs)}"
            )

        if not isinstance(labels, torch.Tensor):
            if isinstance(labels, (int, list, tuple)):
                labels = torch.tensor(labels)
            else:
                raise AttackExecutionError(
                    f"DeepFool attack expects labels to be torch.Tensor or iterable, got {type(labels)}"
                )

        params = config.params or {}
        max_iter = params.get("max_iter", params.get("num_steps", params.get("steps", 10)))
        overshoot = params.get("overshoot", 0.02)
        top_k = params.get("top_k", 10)
        epsilon = config.epsilon if config.epsilon is not None else float("inf")

        clip_min = config.clip_min if config.clip_min is not None else float("-inf")
        clip_max = config.clip_max if config.clip_max is not None else float("inf")

        # Device selection
        device = next(self.raw_model.parameters()).device if list(self.raw_model.parameters()) else torch.device("cpu")

        # Move inputs to target device
        x_orig = inputs.clone().detach().to(device)

        try:
            adv_samples = []
            for idx in range(x_orig.size(0)):
                x_i = x_orig[idx : idx + 1].clone().detach()
                x_start = x_orig[idx : idx + 1].clone().detach()

                for _ in range(max_iter):
                    x_i = x_i.clone().detach()
                    x_i.requires_grad = True
                    outputs = self.raw_model(x_i)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs

                    num_classes = logits.size(-1)
                    current_pred = logits.argmax(dim=-1).item()
                    original_label = labels[idx].item() if idx < len(labels) else current_pred

                    if current_pred != original_label:
                        break

                    # Compute gradients for original predicted class using autograd.grad
                    grad_orig = torch.autograd.grad(
                        outputs=logits[0, current_pred],
                        inputs=x_i,
                        retain_graph=True,
                        create_graph=False,
                    )[0].detach()

                    min_dist = float("inf")
                    best_w = None
                    best_f = None

                    # Top-K candidate class selection for efficient decision boundary calculation
                    k_limit = min(top_k, num_classes)
                    _, top_indices = torch.topk(logits[0], k=k_limit)
                    candidate_classes = [c for c in top_indices.tolist() if c != current_pred]

                    if not candidate_classes:
                        break

                    # Find closest decision boundary across top candidate classes
                    for i, k in enumerate(candidate_classes):
                        is_last = (i == len(candidate_classes) - 1)
                        grad_k = torch.autograd.grad(
                            outputs=logits[0, k],
                            inputs=x_i,
                            retain_graph=(not is_last),
                            create_graph=False,
                        )[0].detach()

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

                    x_i = (x_i.detach() + delta).detach()

                    # Check epsilon ball constraints if epsilon is finite
                    if epsilon < float("inf"):
                        perturbation = torch.clamp(x_i - x_start, min=-epsilon, max=epsilon)
                        x_i = (x_start + perturbation).detach()

                    # Apply clipping bounds
                    if config.clip_min is not None or config.clip_max is not None:
                        x_i = torch.clamp(x_i, clip_min, clip_max).detach()

                adv_samples.append(x_i.detach())

            adv_tensor = torch.cat(adv_samples, dim=0)
            return adv_tensor

        except Exception as e:
            if isinstance(e, AttackExecutionError):
                raise e
            raise AttackExecutionError(f"Error during DeepFool attack generation: {str(e)}") from e


# Self-registration
register_attack("deepfool", DeepFool)
