"""
Carlini & Wagner (C&W) adversarial attack adapted for tabular data.

Formulates adversarial generation as an optimization problem balancing classification margin loss
with feature-aware perturbation costs and tabular manifold constraints.

Reference:
    Carlini & Wagner, "Towards Evaluating the Robustness of Neural Networks" (IEEE S&P 2017)
"""

from typing import Any, Optional
import torch
import torch.optim as optim

from app.attack_engine.attacks.base_attack import BaseAttack
from app.attack_engine.models import AttackMetadata
from app.attack_engine.config import AttackConfig
from app.attack_engine.exceptions import AttackConfigurationError
from app.attack_engine.attack_registry import register_attack
from app.attack_engine.utils.tensor_utils import get_device
from app.attack_engine.attacks.tabular._tabular_utils import (
    get_raw_model,
    parse_tabular_metadata,
    validate_tabular_inputs,
)


class TabularCW(BaseAttack):
    """
    Carlini & Wagner (C&W) attack adapted for tabular data.
    """

    metadata = AttackMetadata(
        name="carlini_wagner_tabular",
        domain="tabular",
        category="optimization",
        attack_type="white_box",
        requires_gradient=True,
        requires_loss=False,
        requires_model_weights=True,
        iterative=True,
        targeted_supported=True,
        untargeted_supported=True,
        query_based=False,
        supported_norms=["L2"],
        computational_cost="high",
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
        Validate TabularCW configuration parameters.
        """
        if config is None:
            return

        params = config.params or {}

        lr = params.get("learning_rate", params.get("lr", 0.05))
        if lr <= 0:
            raise AttackConfigurationError(f"learning_rate must be positive, got {lr}")

        max_iter = params.get(
            "max_iterations", params.get("num_steps", params.get("steps", 50))
        )
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise AttackConfigurationError(
                f"max_iterations must be a positive integer, got {max_iter}"
            )

        confidence = params.get("confidence", 0.0)
        if confidence < 0:
            raise AttackConfigurationError(
                f"confidence cannot be negative, got {confidence}"
            )

        c_penalty = params.get(
            "c_penalty", params.get("c", params.get("initial_const", 1.0))
        )
        if c_penalty <= 0:
            raise AttackConfigurationError(
                f"c_penalty must be positive, got {c_penalty}"
            )

    def generate(
        self, inputs: Any, labels: Any, config: Optional[AttackConfig] = None
    ) -> torch.Tensor:
        """
        Generate tabular C&W adversarial examples.

        Args:
            inputs: Tabular tensor of shape (batch_size, num_features).
            labels: Ground truth labels (or target labels if targeted).
            config: Optional AttackConfig.

        Returns:
            Adversarial tensor satisfying tabular constraints.
        """
        if config is None:
            config = AttackConfig()

        self.validate_config(config)
        params = config.params or {}

        learning_rate = float(params.get("learning_rate", params.get("lr", 0.05)))
        max_iterations = int(
            params.get(
                "max_iterations", params.get("num_steps", params.get("steps", 50))
            )
        )
        confidence = float(params.get("confidence", 0.0))
        c_penalty = float(
            params.get("c_penalty", params.get("c", params.get("initial_const", 1.0)))
        )
        targeted = bool(params.get("targeted", False))
        target_label = params.get("target_label", None)
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

        device = get_device(self.raw_model)
        x_orig, y = validate_tabular_inputs(inputs, labels, attack_name="TabularCW")
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

        metadata = parse_tabular_metadata(self.model, params, num_features=num_features)
        weights = metadata.get_weights_tensor(device)  # (num_features,)
        mutable_mask = metadata.get_mutable_mask(device)  # (num_features,)

        was_training = self.raw_model.training
        self.raw_model.eval()

        # Best sample tracking
        best_adv = x_orig.clone()
        best_dist = torch.full((batch_size,), float("inf"), device=device)
        sample_succeeded = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Optimization variable delta
        delta = torch.zeros_like(x_orig, requires_grad=True)
        optimizer = optim.Adam([delta], lr=learning_rate)

        for step in range(max_iterations):
            # Enforce immutable features strictly zero in delta
            with torch.no_grad():
                delta.data = delta.data * mutable_mask.unsqueeze(0).float()

            x_curr = x_orig + delta
            outputs = self.raw_model(x_curr)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            # C&W margin loss
            if targeted:
                # Want Z(x')_t - max_{i!=t} Z(x')_i >= confidence
                target_logits = logits.gather(1, labels_eval.unsqueeze(1)).squeeze(1)

                # Mask out target class to get rival class max
                other_logits = logits.clone()
                other_logits.scatter_(1, labels_eval.unsqueeze(1), float("-inf"))
                max_other, _ = other_logits.max(dim=1)

                margin_loss = torch.clamp(
                    max_other - target_logits + confidence, min=0.0
                )
            else:
                # Want max_{i!=y} Z(x')_i - Z(x')_y >= confidence
                true_logits = logits.gather(1, labels_eval.unsqueeze(1)).squeeze(1)

                other_logits = logits.clone()
                other_logits.scatter_(1, labels_eval.unsqueeze(1), float("-inf"))
                max_other, _ = other_logits.max(dim=1)

                margin_loss = torch.clamp(true_logits - max_other + confidence, min=0.0)

            # Feature-weighted L2 distance penalty
            dist_penalty = (weights.unsqueeze(0) * (delta**2)).sum(dim=1)
            total_loss = (dist_penalty + c_penalty * margin_loss).sum()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Discrete projection evaluation check
            with torch.no_grad():
                x_projected = metadata.project(
                    x_adv=x_orig + delta,
                    x_orig=x_orig,
                    clip_min=clip_min,
                    clip_max=clip_max,
                )

                eval_out = self.raw_model(x_projected)
                eval_logits = (
                    eval_out.logits if hasattr(eval_out, "logits") else eval_out
                )
                preds = eval_logits.argmax(dim=-1)

                if targeted:
                    succ = preds == labels_eval
                else:
                    succ = preds != labels_eval

                curr_dist = (weights.unsqueeze(0) * (x_projected - x_orig) ** 2).sum(
                    dim=1
                )

                for b in range(batch_size):
                    if succ[b] and (
                        not sample_succeeded[b] or curr_dist[b] < best_dist[b]
                    ):
                        best_adv[b] = x_projected[b]
                        best_dist[b] = curr_dist[b]
                        sample_succeeded[b] = True
                    elif not sample_succeeded[b] and curr_dist[b] < best_dist[b]:
                        best_adv[b] = x_projected[b]

        self.raw_model.train(was_training)

        return metadata.project(
            best_adv, x_orig, clip_min=clip_min, clip_max=clip_max
        ).detach()


# Self-register
register_attack("carlini_wagner_tabular", TabularCW)
register_attack("cw_tabular", TabularCW)
register_attack("tabular_cw", TabularCW)
