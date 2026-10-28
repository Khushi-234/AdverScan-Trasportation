"""
LowProFool adversarial attack implementation for tabular data.

Generates feature-weighted adversarial perturbations by penalizing changes to sensitive,
costly, or high-variance features while searching for misclassification.

Reference:
    Ballet et al., "LowProFool: Deep Learning Adversarial Examples for Tabular Data" (2019)
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


class LowProFool(BaseAttack):
    """
    LowProFool tabular feature-weighted perturbation attack.
    """

    metadata = AttackMetadata(
        name="lowprofool",
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
        supported_norms=["L2", "L1"],
        computational_cost="medium",
        supported_model_types=["pytorch"],
        parameters={
            "feature_weighted": True,
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
        Validate LowProFool configuration parameters.
        """
        if config is None:
            return

        params = config.params or {}

        alpha = params.get("alpha", params.get("step_size", 0.05))
        if alpha <= 0:
            raise AttackConfigurationError(
                f"Step size (alpha) must be positive, got {alpha}"
            )

        num_steps = params.get(
            "num_steps", params.get("steps", params.get("iterations", 20))
        )
        if not isinstance(num_steps, int) or num_steps <= 0:
            raise AttackConfigurationError(
                f"num_steps must be a positive integer, got {num_steps}"
            )

        lambda_weight = params.get("lambda_weight", params.get("tradeoff", 0.1))
        if lambda_weight < 0:
            raise AttackConfigurationError(
                f"lambda_weight cannot be negative, got {lambda_weight}"
            )

        budget = params.get("budget", params.get("epsilon", None))
        if budget is not None:
            if not isinstance(budget, (int, float)) or budget <= 0:
                raise AttackConfigurationError(
                    f"budget/epsilon must be a positive number, got {budget}"
                )

    def generate(
        self, inputs: Any, labels: Any, config: Optional[AttackConfig] = None
    ) -> torch.Tensor:
        """
        Generate LowProFool adversarial examples on tabular data.

        Args:
            inputs: Tabular tensor of shape (batch_size, num_features).
            labels: Ground truth labels (or target labels if targeted).
            config: Optional AttackConfig.

        Returns:
            Adversarial tensor with minimal feature-weighted perturbation.
        """
        if config is None:
            config = AttackConfig()

        self.validate_config(config)
        params = config.params or {}

        alpha = float(params.get("alpha", params.get("step_size", 0.05)))
        num_steps = int(
            params.get("num_steps", params.get("steps", params.get("iterations", 20)))
        )
        lambda_weight = float(params.get("lambda_weight", params.get("tradeoff", 0.1)))
        targeted = bool(params.get("targeted", False))
        target_label = params.get("target_label", None)
        loss_fn = config.loss_fn or nn.CrossEntropyLoss()
        budget = params.get("budget", params.get("epsilon", None))
        if budget is not None:
            budget = float(budget)

        device = get_device(self.raw_model)
        x_orig, y = validate_tabular_inputs(inputs, labels, attack_name="LowProFool")
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

        # Parse tabular metadata and feature weights
        metadata = parse_tabular_metadata(self.model, params, num_features=num_features)
        weights = metadata.get_weights_tensor(device)  # shape (num_features,)
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

        # Best adversarial samples tracking
        best_adv = x_orig.clone()
        best_distance = torch.full((batch_size,), float("inf"), device=device)
        sample_succeeded = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Check initial success
        with torch.no_grad():
            init_out = self.raw_model(x_orig)
            init_logits = init_out.logits if hasattr(init_out, "logits") else init_out
            init_preds = init_logits.argmax(dim=-1)
            if targeted:
                initial_success = init_preds == labels_eval
            else:
                initial_success = init_preds != labels_eval

            sample_succeeded |= initial_success
            best_distance[initial_success] = 0.0

        x_curr = x_orig.clone()

        for step in range(num_steps):
            x_curr.requires_grad_(True)
            outputs = self.raw_model(x_curr)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            if targeted:
                clf_loss = -loss_fn(logits, labels_eval)
            else:
                clf_loss = loss_fn(logits, labels_eval)

            # Feature-weighted distance penalty: sum_j w_j * (x_j - x_orig_j)^2
            dist_penalty = (weights.unsqueeze(0) * (x_curr - x_orig) ** 2).sum()
            total_loss = clf_loss - lambda_weight * dist_penalty

            grad = torch.autograd.grad(
                total_loss,
                x_curr,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]

            if grad is None:
                raise AttackExecutionError(
                    "LowProFool failed to compute gradients w.r.t input features."
                )

            with torch.no_grad():
                # Modulate gradient inversely by feature weights: higher weight = smaller step
                weighted_grad = grad / (weights.unsqueeze(0) + 1e-8)

                # Normalize per sample
                grad_norm = torch.norm(weighted_grad, p=2, dim=1, keepdim=True) + 1e-12
                update_dir = weighted_grad / grad_norm

                # Zero out updates for immutable features
                update_dir = update_dir * mutable_mask.unsqueeze(0).float()
                x_curr = x_curr + alpha * update_dir

                # Project onto tabular constraints
                x_projected = metadata.project(
                    x_adv=x_curr,
                    x_orig=x_orig,
                    clip_min=clip_min,
                    clip_max=clip_max,
                )

                # Enforce perturbation budget if specified
                if budget is not None:
                    pert = x_projected - x_orig
                    w_dist = (weights.unsqueeze(0) * pert.abs()).sum(
                        dim=1, keepdim=True
                    )
                    scale = torch.clamp(budget / (w_dist + 1e-12), max=1.0)
                    x_projected = metadata.project(
                        x_adv=x_orig + pert * scale,
                        x_orig=x_orig,
                        clip_min=clip_min,
                        clip_max=clip_max,
                    )

                x_curr = x_projected

                # Evaluate current candidate
                eval_out = self.raw_model(x_projected)
                eval_logits = (
                    eval_out.logits if hasattr(eval_out, "logits") else eval_out
                )
                preds = eval_logits.argmax(dim=-1)

                if targeted:
                    succ = preds == labels_eval
                else:
                    succ = preds != labels_eval

                # Compute weighted L1 distance
                curr_dist = (weights.unsqueeze(0) * (x_projected - x_orig).abs()).sum(
                    dim=1
                )

                # Update best sample for each batch item
                for b in range(batch_size):
                    if succ[b] and (
                        not sample_succeeded[b] or curr_dist[b] < best_distance[b]
                    ):
                        best_adv[b] = x_projected[b]
                        best_distance[b] = curr_dist[b]
                        sample_succeeded[b] = True
                    elif not sample_succeeded[b] and curr_dist[b] < best_distance[b]:
                        best_adv[b] = x_projected[b]

        self.raw_model.train(was_training)

        # Record perturbation statistics into config params and self.last_stats
        final_adv = metadata.project(
            best_adv, x_orig, clip_min=clip_min, clip_max=clip_max
        ).detach()
        final_w_dist = (
            (weights.unsqueeze(0) * (final_adv - x_orig).abs()).sum(dim=1).mean().item()
        )
        final_l1_dist = (final_adv - x_orig).abs().sum(dim=1).mean().item()
        final_linf_dist = (final_adv - x_orig).abs().max(dim=1)[0].mean().item()
        succ_rate = sample_succeeded.float().mean().item()

        stats = {
            "weighted_distance": float(final_w_dist),
            "l1_distance": float(final_l1_dist),
            "linf_distance": float(final_linf_dist),
            "success_rate": float(succ_rate),
        }
        self.last_stats = stats
        params["perturbation_stats"] = stats
        params["weighted_distance"] = float(final_w_dist)
        params["success_rate"] = float(succ_rate)

        return final_adv

    def get_perturbation_stats(self) -> dict:
        """Return perturbation statistics from the most recent attack run."""
        return getattr(self, "last_stats", {})


# Self-register
register_attack("lowprofool", LowProFool)
