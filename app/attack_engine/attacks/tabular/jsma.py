"""
Jacobian-based Saliency Map Attack (JSMA) adapted for tabular data.

Identifies the most influential feature(s) via the output logit Jacobian and perturbs
a minimal, sparse subset of features (L0 constraint) while respecting tabular constraints.

Reference:
    Papernot et al., "The Limitations of Deep Learning in Adversarial Settings" (IEEE EuroS&P 2016)
"""

from typing import Any, Optional, Set
import torch

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


class TabularJSMA(BaseAttack):
    """
    Tabular Jacobian-based Saliency Map Attack (TabularJSMA).
    """

    metadata = AttackMetadata(
        name="jsma_tabular",
        domain="tabular",
        category="gradient",
        attack_type="white_box",
        requires_gradient=True,
        requires_loss=False,
        requires_model_weights=True,
        iterative=True,
        targeted_supported=True,
        untargeted_supported=True,
        query_based=False,
        supported_norms=["L0"],
        computational_cost="medium",
        supported_model_types=["pytorch"],
        parameters={
            "sparse": True,
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
        Validate TabularJSMA configuration parameters.
        """
        if config is None:
            return

        params = config.params or {}

        max_features = params.get("max_features", params.get("max_modifications", 5))
        if not isinstance(max_features, int) or max_features <= 0:
            raise AttackConfigurationError(
                f"max_features must be a positive integer, got {max_features}"
            )

        step_size = params.get("step_size", params.get("theta", 0.1))
        if step_size <= 0:
            raise AttackConfigurationError(
                f"step_size must be positive, got {step_size}"
            )

    def generate(
        self, inputs: Any, labels: Any, config: Optional[AttackConfig] = None
    ) -> torch.Tensor:
        """
        Generate Tabular JSMA adversarial examples.

        Args:
            inputs: Tabular tensor of shape (batch_size, num_features).
            labels: Ground truth labels (or target labels if targeted).
            config: Optional AttackConfig.

        Returns:
            Sparse adversarial tensor satisfying tabular constraints.
        """
        if config is None:
            config = AttackConfig()

        self.validate_config(config)
        params = config.params or {}

        max_features = int(
            params.get("max_features", params.get("max_modifications", 5))
        )
        step_size = float(params.get("step_size", params.get("theta", 0.1)))
        targeted = bool(params.get("targeted", False))
        target_label = params.get("target_label", None)

        device = get_device(self.raw_model)
        x_orig, y = validate_tabular_inputs(inputs, labels, attack_name="TabularJSMA")
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
        mutable_mask = metadata.get_mutable_mask(device)

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

        x_adv = x_orig.clone()

        for b in range(batch_size):
            sample = x_adv[b : b + 1].clone()
            target_class = int(labels_eval[b].item())
            true_class = int(y[b].item())

            # Initial prediction check
            with torch.no_grad():
                init_out = self.raw_model(sample)
                init_logits = (
                    init_out.logits if hasattr(init_out, "logits") else init_out
                )
                init_pred = init_logits.argmax(dim=-1).item()

            if targeted and init_pred == target_class:
                continue
            elif not targeted and init_pred != true_class:
                continue

            modified_features: Set[int] = set()

            for step in range(max_features):
                sample_var = sample.clone().detach().requires_grad_(True)
                outputs = self.raw_model(sample_var)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                num_classes = logits.size(-1)

                # Compute Jacobian matrix: shape (num_classes, num_features)
                jacobian = torch.zeros((num_classes, num_features), device=device)
                for k in range(num_classes):
                    if sample_var.grad is not None:
                        sample_var.grad.zero_()
                    grad_k = torch.autograd.grad(
                        logits[0, k],
                        sample_var,
                        retain_graph=(k < num_classes - 1),
                        create_graph=False,
                    )[0]
                    if grad_k is not None:
                        jacobian[k] = grad_k[0]

                # Compute saliency score per feature
                saliency = torch.full((num_features,), float("-inf"), device=device)

                for j in range(num_features):
                    # Skip immutable features or already modified features
                    if not mutable_mask[j] or j in modified_features:
                        continue

                    feat = metadata[j]

                    if targeted:
                        # Increase target class, decrease rival classes
                        j_target = jacobian[target_class, j].item()
                        other_sum = jacobian[:, j].sum().item() - j_target

                        if j_target > 0 and other_sum < 0:
                            saliency[j] = j_target * abs(other_sum)
                        elif j_target > 0:
                            saliency[j] = j_target
                        else:
                            saliency[j] = float("-inf")
                    else:
                        # Decrease true class, increase rival classes
                        j_true = jacobian[true_class, j].item()
                        other_abs_sum = sum(
                            abs(jacobian[k, j].item())
                            for k in range(num_classes)
                            if k != true_class
                        )
                        saliency[j] = abs(j_true) * other_abs_sum

                # Fallback if no positive saliency found: pick highest magnitude gradient
                if torch.isneginf(saliency).all():
                    if targeted:
                        fallback_scores = jacobian[target_class]
                    else:
                        fallback_scores = -jacobian[true_class]
                    for j in range(num_features):
                        if not mutable_mask[j] or j in modified_features:
                            fallback_scores[j] = float("-inf")
                    best_feat = torch.argmax(fallback_scores).item()
                    if torch.isneginf(fallback_scores[best_feat]):
                        break
                else:
                    best_feat = torch.argmax(saliency).item()

                feat = metadata[best_feat]
                current_val = sample[0, best_feat].item()

                # Determine direction:
                if targeted:
                    direction = (
                        1.0 if jacobian[target_class, best_feat].item() >= 0 else -1.0
                    )
                else:
                    direction = (
                        -1.0 if jacobian[true_class, best_feat].item() >= 0 else 1.0
                    )

                # Apply feature perturbation
                if feat.type == "continuous":
                    new_val = current_val + direction * step_size
                elif feat.type == "integer":
                    delta_int = max(1.0, round(step_size)) * direction
                    new_val = current_val + delta_int
                elif feat.type == "binary":
                    new_val = 0.0 if current_val >= 0.5 else 1.0
                elif feat.type == "categorical":
                    # Pick category that maximizes objective
                    if feat.categories and len(feat.categories) > 1:
                        best_cat = current_val
                        best_cat_score = float("-inf")
                        for cat in feat.categories:
                            if cat == current_val:
                                continue
                            test_sample = sample.clone()
                            test_sample[0, best_feat] = cat
                            with torch.no_grad():
                                t_out = self.raw_model(test_sample)
                                t_logits = (
                                    t_out.logits if hasattr(t_out, "logits") else t_out
                                )
                                if targeted:
                                    cat_score = t_logits[0, target_class].item()
                                else:
                                    cat_score = -t_logits[0, true_class].item()
                            if cat_score > best_cat_score:
                                best_cat_score = cat_score
                                best_cat = cat
                        new_val = best_cat
                    else:
                        new_val = current_val
                else:
                    new_val = current_val + direction * step_size

                sample[0, best_feat] = new_val
                modified_features.add(best_feat)

                # Project sample onto constraints
                sample = metadata.project(
                    x_adv=sample,
                    x_orig=x_orig[b : b + 1],
                    clip_min=clip_min,
                    clip_max=clip_max,
                )

                # Re-evaluate prediction for early stopping
                with torch.no_grad():
                    curr_out = self.raw_model(sample)
                    curr_logits = (
                        curr_out.logits if hasattr(curr_out, "logits") else curr_out
                    )
                    curr_pred = curr_logits.argmax(dim=-1).item()

                if targeted and curr_pred == target_class:
                    break
                elif not targeted and curr_pred != true_class:
                    break

            x_adv[b] = sample[0]

        self.raw_model.train(was_training)

        return metadata.project(
            x_adv, x_orig, clip_min=clip_min, clip_max=clip_max
        ).detach()


# Self-register
register_attack("jsma_tabular", TabularJSMA)
register_attack("tabular_jsma", TabularJSMA)
