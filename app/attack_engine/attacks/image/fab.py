"""
Fast Adaptive Boundary (FAB) adversarial attack implementation.
"""

from typing import Any, Optional, Union
import torch
import torch.nn as nn

from app.attack_engine.attacks.base_attack import BaseAttack
from app.attack_engine.models import AttackMetadata
from app.attack_engine.config import AttackConfig
from app.attack_engine.exceptions import AttackExecutionError, AttackConfigurationError
from app.attack_engine.attack_registry import register_attack
from app.attack_engine.utils.tensor_utils import get_device, clip_tensor, project_lp
from app.attack_engine.utils.validation import validate_inputs, validate_attack_config


class FAB(BaseAttack):
    """
    Fast Adaptive Boundary (FAB) attack implementation.

    An iterative, gradient-based white-box boundary attack that finds minimal adversarial
    perturbations near classifier decision boundaries using local first-order approximations.
    Supports untargeted and targeted modes with Linf and L2 norm distance constraints.
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
        if config is not None:
            if (
                config.clip_min is not None
                and config.clip_max is not None
                and config.clip_min >= config.clip_max
            ):
                raise AttackConfigurationError(
                    f"clip_min ({config.clip_min}) must be strictly less than clip_max ({config.clip_max})"
                )
            params = config.params or {}
            max_iter = params.get("max_iter", params.get("n_iter", params.get("num_steps")))
            if max_iter is not None and (not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter <= 0):
                raise AttackExecutionError("FAB max_iter must be greater than 0.")

            top_k = params.get("top_k")
            if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0):
                raise AttackExecutionError("FAB top_k must be greater than 0.")

            alpha_max = params.get("alpha_max")
            if alpha_max is not None and (not isinstance(alpha_max, (int, float)) or isinstance(alpha_max, bool) or alpha_max <= 0):
                raise AttackExecutionError("FAB alpha_max must be a positive float.")

            eta = params.get("eta")
            if eta is not None and (not isinstance(eta, (int, float)) or isinstance(eta, bool) or eta < 1.0):
                raise AttackExecutionError("FAB eta must be a float >= 1.0.")

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
            inputs: Input tensor (e.g. torch.Tensor) of shape (B, C, H, W) or (B, ...).
            labels: True class labels tensor (or target labels if targeted=True).
            config: AttackConfig containing clip_min, clip_max, epsilon, and params:
                - max_iter / n_iter / num_steps: Maximum iterations (default 10).
                - alpha_max: Maximum step size for adaptive boundary step (default 0.1).
                - eta: Overshoot factor >= 1.0 (default 1.05).
                - top_k: Number of candidate classes to check (default 10).
                - norm: Norm constraint ('Linf' or 'L2', default 'Linf').
                - targeted: Boolean flag indicating targeted mode (default False).
                - target_labels: Optional explicit target labels tensor for targeted mode.

        Returns:
            Adversarial input tensor with shape matching inputs.
        """
        if config is None:
            config = AttackConfig()

        self.validate_config(config)
        validate_inputs(inputs, labels, attack_name="FAB")

        params = config.params or {}
        max_iter = params.get("max_iter", params.get("n_iter", params.get("num_steps", 10)))
        if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter <= 0:
            raise AttackExecutionError("FAB max_iter must be greater than 0.")

        alpha_max = params.get("alpha_max", 0.1)
        if not isinstance(alpha_max, (int, float)) or isinstance(alpha_max, bool) or alpha_max <= 0:
            raise AttackExecutionError("FAB alpha_max must be a positive float.")

        eta = params.get("eta", 1.05)
        if not isinstance(eta, (int, float)) or isinstance(eta, bool) or eta < 1.0:
            raise AttackExecutionError("FAB eta must be a float >= 1.0.")

        top_k = params.get("top_k", 10)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise AttackExecutionError("FAB top_k must be greater than 0.")

        norm = params.get("norm", "Linf")
        if isinstance(norm, str):
            if norm.upper() not in ("LINF", "INF", "L2"):
                raise AttackExecutionError(
                    f"Unsupported norm '{norm}' for FAB. Supported norms: ['Linf', 'L2']."
                )
            norm_str = "Linf" if norm.upper() in ("LINF", "INF") else "L2"
        elif norm == float("inf"):
            norm_str = "Linf"
        elif norm == 2:
            norm_str = "L2"
        else:
            raise AttackExecutionError(
                f"Unsupported norm '{norm}' for FAB. Supported norms: ['Linf', 'L2']."
            )

        targeted = bool(params.get("targeted", False))
        epsilon = float(config.epsilon) if config.epsilon is not None else float("inf")
        if epsilon < 0:
            raise AttackExecutionError("FAB epsilon cannot be negative.")

        device = get_device(self.raw_model)
        x_orig = inputs.clone().detach().to(device)
        batch_size = x_orig.size(0)

        # Validate labels dtype and device
        if not isinstance(labels, torch.Tensor):
            labels_tensor = torch.as_tensor(labels, dtype=torch.long, device=device)
        else:
            labels_tensor = labels.to(dtype=torch.long, device=device)

        if labels_tensor.dim() == 0 and batch_size == 1:
            labels_tensor = labels_tensor.unsqueeze(0)
        elif labels_tensor.dim() != 1 or labels_tensor.size(0) != batch_size:
            raise AttackExecutionError(
                f"Labels shape {labels_tensor.shape} does not match batch size {batch_size}."
            )

        was_training = self.raw_model.training
        self.raw_model.eval()

        try:
            # Model output validation
            with torch.no_grad():
                init_out = self.raw_model(x_orig)
                init_logits = init_out.logits if hasattr(init_out, "logits") else init_out
                if isinstance(init_logits, (tuple, list)):
                    init_logits = init_logits[0]
                if init_logits.dim() != 2:
                    raise AttackExecutionError(
                        f"Model output must be 2D logits (batch_size, num_classes), got shape {init_logits.shape}."
                    )
                num_classes = init_logits.size(-1)
                if num_classes < 2:
                    raise AttackExecutionError(f"Model must have at least 2 classes, got {num_classes}.")

                clean_preds = init_logits.argmax(dim=-1)

            # Validate class label ranges
            if (labels_tensor < 0).any() or (labels_tensor >= num_classes).any():
                raise AttackExecutionError(
                    f"Labels must be in range [0, {num_classes - 1}]."
                )

            # Determine original vs target labels
            target_labels_tensor = None
            if targeted:
                param_targets = params.get("target_labels", params.get("targets", params.get("target")))
                if param_targets is not None:
                    if not isinstance(param_targets, torch.Tensor):
                        target_labels_tensor = torch.as_tensor(param_targets, dtype=torch.long, device=device)
                    else:
                        target_labels_tensor = param_targets.to(dtype=torch.long, device=device)
                    if target_labels_tensor.dim() == 0 and batch_size == 1:
                        target_labels_tensor = target_labels_tensor.unsqueeze(0)
                    orig_labels_tensor = labels_tensor
                else:
                    # labels_tensor was passed as target labels
                    target_labels_tensor = labels_tensor
                    orig_labels_tensor = clean_preds

                if (target_labels_tensor < 0).any() or (target_labels_tensor >= num_classes).any():
                    raise AttackExecutionError(
                        f"Target labels must be in range [0, {num_classes - 1}]."
                    )

                if (target_labels_tensor == orig_labels_tensor).any():
                    raise AttackExecutionError(
                        "Target labels must be supplied and differ from original labels."
                    )
            else:
                orig_labels_tensor = labels_tensor

            # Determine appropriate clipping bounds
            # If input is normalized outside [0, 1] (e.g. HuggingFace normalized with mean/std),
            # avoid blindly clipping to [0, 1] unless bounds were explicitly customized.
            clip_min = config.clip_min
            clip_max = config.clip_max
            if clip_min is not None or clip_max is not None:
                c_min_val = clip_min if clip_min is not None else float("-inf")
                c_max_val = clip_max if clip_max is not None else float("inf")
                x_min = x_orig.min().item()
                x_max = x_orig.max().item()
                if x_min < c_min_val - 1e-3 or x_max > c_max_val + 1e-3:
                    # Input is outside default box bounds (likely normalized). Disable clipping.
                    clip_min = None
                    clip_max = None

            adv_samples = []
            for idx in range(batch_size):
                x_0 = x_orig[idx : idx + 1].clone().detach()
                x_i = x_0.clone().detach()
                y_orig = orig_labels_tensor[idx].item()
                y_target = target_labels_tensor[idx].item() if targeted else None

                best_adv = x_0.clone().detach()
                best_norm = float("inf")

                for t in range(max_iter):
                    x_i.requires_grad_(True)
                    outputs = self.raw_model(x_i)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    if isinstance(logits, (tuple, list)):
                        logits = logits[0]
                    if logits.dim() > 2:
                        logits = logits.view(1, -1)

                    current_pred = logits.argmax(dim=-1).item()

                    is_adv = (current_pred == y_target) if targeted else (current_pred != y_orig)

                    if is_adv:
                        diff = x_i.detach() - x_0
                        if norm_str == "Linf":
                            cur_norm = diff.abs().max().item()
                        else:
                            cur_norm = torch.norm(diff.view(-1), p=2).item()

                        if cur_norm <= epsilon + 1e-6 and cur_norm < best_norm:
                            best_norm = cur_norm
                            best_adv = x_i.detach().clone()

                        # Interpolate back toward x_0 to seek minimal perturbation boundary
                        x_i = (x_0 + x_i.detach()) / 2.0
                        continue

                    # If model is not differentiable with respect to input, stop
                    if logits.grad_fn is None:
                        break

                    # Current point is not adversarial. Compute boundary projection.
                    s = current_pred

                    if targeted:
                        # Boundary between target class y_target and current leading class s
                        grad_target = torch.autograd.grad(
                            logits[0, y_target],
                            x_i,
                            retain_graph=True,
                            create_graph=False,
                        )[0].clone()

                        grad_s = torch.autograd.grad(
                            logits[0, s],
                            x_i,
                            retain_graph=True,
                            create_graph=False,
                        )[0].clone()

                        w = grad_target - grad_s
                        df = (logits[0, y_target] - logits[0, s]).item()

                        if norm_str == "Linf":
                            # L1 dual norm for Linf
                            w_norm = torch.norm(w.view(-1), p=1).item() + 1e-12
                            if w_norm < 1e-8:
                                break
                            step = - (df / w_norm) * torch.sign(w) * eta
                        else:
                            # L2 dual norm for L2
                            w_norm_sq = (torch.norm(w.view(-1), p=2).item()) ** 2 + 1e-12
                            if w_norm_sq < 1e-8:
                                break
                            step = - (df / w_norm_sq) * w * eta

                        best_proj = x_i.detach() + step

                    else:
                        # Untargeted: evaluate closest candidate boundary among top_k classes
                        grad_s = torch.autograd.grad(
                            logits[0, s],
                            x_i,
                            retain_graph=True,
                            create_graph=False,
                        )[0].clone()

                        k_limit = min(top_k, num_classes)
                        _, top_indices = torch.topk(logits[0], k=k_limit)
                        candidate_classes = [c.item() for c in top_indices if c.item() != s]
                        if not candidate_classes:
                            candidate_classes = [c for c in range(num_classes) if c != s]

                        min_dist = float("inf")
                        best_step = None

                        for c in candidate_classes:
                            grad_c = torch.autograd.grad(
                                logits[0, c],
                                x_i,
                                retain_graph=True,
                                create_graph=False,
                            )[0].clone()

                            w_c = grad_c - grad_s
                            df_c = (logits[0, c] - logits[0, s]).item()

                            if norm_str == "Linf":
                                # Dual norm for Linf is L1 norm
                                w_norm = torch.norm(w_c.view(-1), p=1).item() + 1e-12
                                dist = abs(df_c) / w_norm
                                if dist < min_dist:
                                    min_dist = dist
                                    best_step = - (df_c / w_norm) * torch.sign(w_c) * eta
                            else:
                                # Dual norm for L2 is L2 norm
                                w_norm_sq = (torch.norm(w_c.view(-1), p=2).item()) ** 2 + 1e-12
                                dist = abs(df_c) / (w_norm_sq ** 0.5)
                                if dist < min_dist:
                                    min_dist = dist
                                    best_step = - (df_c / w_norm_sq) * w_c * eta

                        if best_step is None:
                            break

                        best_proj = x_i.detach() + best_step

                    # Evaluate candidate boundary projection directly
                    cand_proj = best_proj
                    if epsilon < float("inf"):
                        cand_proj = project_lp(cand_proj, x_0, epsilon=epsilon, norm=norm_str)
                    if clip_min is not None or clip_max is not None:
                        cand_proj = clip_tensor(cand_proj, clip_min, clip_max)

                    with torch.no_grad():
                        proj_out = self.raw_model(cand_proj)
                        proj_logits = proj_out.logits if hasattr(proj_out, "logits") else proj_out
                        if isinstance(proj_logits, (tuple, list)):
                            proj_logits = proj_logits[0]
                        proj_pred = proj_logits.argmax(dim=-1).item()
                        proj_is_adv = (proj_pred == y_target) if targeted else (proj_pred != y_orig)
                        if proj_is_adv:
                            diff = cand_proj.detach() - x_0
                            if norm_str == "Linf":
                                p_norm = diff.abs().max().item()
                            else:
                                p_norm = torch.norm(diff.view(-1), p=2).item()
                            if p_norm <= epsilon + 1e-6 and p_norm < best_norm:
                                best_norm = p_norm
                                best_adv = cand_proj.detach().clone()

                    # Adaptive step combining current point and boundary projection
                    alpha = min(alpha_max, 1.0 / (t + 1))
                    x_i = (1.0 - alpha) * x_i.detach() + alpha * best_proj

                    # Epsilon ball projection
                    if epsilon < float("inf"):
                        x_i = project_lp(x_i, x_0, epsilon=epsilon, norm=norm_str)

                    # Clipping bounds
                    if clip_min is not None or clip_max is not None:
                        x_i = clip_tensor(x_i, clip_min, clip_max)

                # Evaluate final optimization candidate produced after the last step
                with torch.no_grad():
                    final_out = self.raw_model(x_i)
                    final_logits = final_out.logits if hasattr(final_out, "logits") else final_out
                    if isinstance(final_logits, (tuple, list)):
                        final_logits = final_logits[0]
                    if final_logits.dim() > 2:
                        final_logits = final_logits.view(1, -1)
                    final_pred = final_logits.argmax(dim=-1).item()

                    final_is_adv = (final_pred == y_target) if targeted else (final_pred != y_orig)
                    if final_is_adv:
                        diff = x_i.detach() - x_0
                        if norm_str == "Linf":
                            final_norm = diff.abs().max().item()
                        else:
                            final_norm = torch.norm(diff.view(-1), p=2).item()

                        if final_norm <= epsilon + 1e-6 and final_norm < best_norm:
                            best_norm = final_norm
                            best_adv = x_i.detach().clone()

                # If successful adversarial example found within budget, return best candidate;
                # otherwise preserve original input
                if best_norm < float("inf"):
                    adv_samples.append(best_adv)
                else:
                    adv_samples.append(x_0.detach())

            adv_tensor = torch.cat(adv_samples, dim=0)
            return adv_tensor

        except (AttackExecutionError, AttackConfigurationError):
            raise
        except Exception as e:
            raise AttackExecutionError(f"Error during FAB attack generation: {str(e)}") from e
        finally:
            self.raw_model.train(was_training)


# Self-registration
register_attack("fab", FAB)
