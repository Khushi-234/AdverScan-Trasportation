"""
Fast Adaptive Boundary (FAB) adversarial attack implementation.
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


class FAB(BaseAttack):
    """
    Fast Adaptive Boundary (FAB) attack implementation.

    An iterative, gradient-based white-box boundary attack that finds minimal adversarial
    perturbations near classifier decision boundaries using local first-order approximations.
    """

    metadata = AttackMetadata(
        name="fab",
        domain="image",
        category="boundary",
        attack_type="white_box",
        requires_gradient=True,
        requires_loss=False,
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
        Initialize FAB attack.

        Args:
            model: PyTorch nn.Module or BaseModelAdapter wrapping PyTorch model.
        """
        super().__init__(model)
        self.raw_model = self._get_raw_model()

    def validate_config(self, config: Optional[AttackConfig] = None) -> None:
        """
        Validate FAB-specific configuration.
        """
        validate_attack_config(config)
        if config is not None and config.params:
            params = config.params
            max_iter = params.get("max_iter", params.get("n_iter", params.get("num_steps")))
            if max_iter is not None and (not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter <= 0):
                raise AttackExecutionError("FAB max_iter must be greater than 0.")

            top_k = params.get("top_k")
            if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0):
                raise AttackExecutionError("FAB top_k must be greater than 0.")

            norm = params.get("norm")
            if norm is not None:
                if isinstance(norm, str):
                    if norm.upper() not in ("LINF", "INF", "L2"):
                        raise AttackExecutionError(
                            f"Unsupported norm '{norm}' for FAB. Supported norms: ['Linf', 'L2']."
                        )
                elif norm not in (2, float("inf")):
                    raise AttackExecutionError(
                        f"Unsupported norm '{norm}' for FAB. Supported norms: ['Linf', 'L2']."
                    )

    def generate(
        self, inputs: Any, labels: Any, config: Optional[AttackConfig] = None
    ) -> Any:
        """
        Generate FAB adversarial examples.

        Args:
            inputs: Input tensor (e.g. torch.Tensor).
            labels: True class labels tensor (or target labels if targeted=True).
            config: AttackConfig containing clip_min, clip_max, epsilon, and params:
                - max_iter / n_iter: Maximum number of iterations (default 10).
                - alpha_max: Maximum step size for adaptive boundary step (default 0.1).
                - eta: Overshoot factor (default 1.05).
                - top_k: Number of candidate classes to check (default 10).
                - norm: Norm constraint ('Linf' or 'L2', default 'Linf').
                - targeted: Boolean flag indicating targeted mode (default False).

        Returns:
            Adversarial input tensor.
        """
        if config is None:
            config = AttackConfig()

        self.validate_config(config)
        validate_inputs(inputs, labels, attack_name="FAB")

        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels)

        params = config.params or {}
        max_iter = params.get("max_iter", params.get("n_iter", params.get("num_steps", 10)))
        if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter <= 0:
            raise AttackExecutionError("FAB max_iter must be greater than 0.")

        alpha_max = params.get("alpha_max", 0.1)
        eta = params.get("eta", 1.05)
        top_k = params.get("top_k", 10)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise AttackExecutionError("FAB top_k must be greater than 0.")

        norm = params.get("norm", "Linf")
        if isinstance(norm, str):
            if norm.upper() not in ("LINF", "INF", "L2"):
                raise AttackExecutionError(
                    f"Unsupported norm '{norm}' for FAB. Supported norms: ['Linf', 'L2']."
                )
        elif norm not in (2, float("inf")):
            raise AttackExecutionError(
                f"Unsupported norm '{norm}' for FAB. Supported norms: ['Linf', 'L2']."
            )
        else:
            norm = "Linf" if norm == float("inf") else "L2"

        targeted = params.get("targeted", False)
        epsilon = config.epsilon if config.epsilon is not None else float("inf")

        device = get_device(self.raw_model)
        x_orig = inputs.clone().detach().to(device)
        labels = labels.to(device)

        was_training = self.raw_model.training
        self.raw_model.eval()

        try:
            adv_samples = []
            for idx in range(x_orig.size(0)):
                x_0 = x_orig[idx : idx + 1].clone().detach()
                x_i = x_0.clone().detach()
                y_true = labels[idx].item()

                best_adv = x_0.clone().detach()
                best_norm = float("inf")

                for t in range(max_iter):
                    x_i.requires_grad_(True)
                    outputs = self.raw_model(x_i)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    if logits.dim() > 2:
                        logits = logits.view(1, -1)

                    current_pred = logits.argmax(dim=-1).item()
                    num_classes = logits.size(-1)

                    is_adv = (current_pred == y_true) if targeted else (current_pred != y_true)

                    # Track best adversarial candidate
                    if is_adv:
                        diff = x_i.detach() - x_0
                        if norm.upper() in ("LINF", "INF"):
                            cur_norm = diff.abs().max().item()
                        else:
                            cur_norm = torch.norm(diff.view(-1), p=2).item()

                        if cur_norm < best_norm:
                            best_norm = cur_norm
                            best_adv = x_i.detach().clone()

                        # If already adversarial, interpolate back toward x_0 to minimize perturbation
                        x_i = (x_0 + x_i.detach()) / 2.0
                        continue

                    # Compute gradient for true class (or target class)
                    grad_true = torch.autograd.grad(
                        logits[0, y_true],
                        x_i,
                        retain_graph=True,
                        create_graph=False,
                    )[0].clone()

                    k_limit = min(top_k, num_classes)
                    _, top_indices = torch.topk(logits[0], k=k_limit)
                    candidate_classes = top_indices.tolist()

                    min_dist = float("inf")
                    best_proj = None

                    # Find hyperplane of closest decision boundary
                    for c in candidate_classes:
                        if c == y_true:
                            continue

                        grad_c = torch.autograd.grad(
                            logits[0, c],
                            x_i,
                            retain_graph=True,
                            create_graph=False,
                        )[0].clone()

                        w_c = grad_c - grad_true
                        df_c = (logits[0, c] - logits[0, y_true]).item()

                        if norm.upper() in ("LINF", "INF"):
                            w_norm = torch.norm(w_c.view(-1), p=1).item() + 1e-12
                            dist = abs(df_c) / w_norm
                            if dist < min_dist:
                                min_dist = dist
                                # L1 dual projection gives sign step for Linf
                                step = (abs(df_c) / w_norm) * w_c.sign() * eta
                                best_proj = x_i.detach() + step
                        else:
                            w_norm_sq = (torch.norm(w_c.view(-1), p=2).item()) ** 2 + 1e-12
                            dist = abs(df_c) / (w_norm_sq ** 0.5)
                            if dist < min_dist:
                                min_dist = dist
                                step = (abs(df_c) / w_norm_sq) * w_c * eta
                                best_proj = x_i.detach() + step

                    if best_proj is None:
                        break

                    # Adaptive step combining current point and boundary projection
                    alpha = min(alpha_max, 1.0 / (t + 1))
                    x_i = (1.0 - alpha) * x_i.detach() + alpha * best_proj

                    # Projection onto epsilon ball if bounded
                    if epsilon < float("inf"):
                        x_i = project_lp(x_i, x_0, epsilon=epsilon, norm=norm)

                    # Clipping bounds
                    x_i = clip_tensor(x_i, config.clip_min, config.clip_max)

                # Use best found adversarial sample if any, else latest x_i
                if best_norm < float("inf"):
                    adv_samples.append(best_adv)
                else:
                    adv_samples.append(x_i.detach())

            adv_tensor = torch.cat(adv_samples, dim=0)
            return adv_tensor

        except AttackExecutionError:
            raise
        except Exception as e:
            raise AttackExecutionError(f"Error during FAB attack generation: {str(e)}") from e
        finally:
            self.raw_model.train(was_training)


# Self-registration
register_attack("fab", FAB)
