"""
Jacobian-based Saliency Map Attack (JSMA) for time-series models.
"""

from typing import Any, Optional
import torch

from app.attack_engine.attacks.base_attack import BaseAttack
from app.attack_engine.models import AttackMetadata
from app.attack_engine.config import AttackConfig
from app.attack_engine.exceptions import (
    AttackConfigurationError,
    AttackExecutionError,
)
from app.attack_engine.attack_registry import register_attack
from app.attack_engine.utils.tensor_utils import get_device
from app.attack_engine.utils.validation import check_model_compatibility
from app.attack_engine.attacks.time_series.pgd import _apply_time_series_bounds


class JSMA(BaseAttack):
    """
    Time-Series Jacobian-based Saliency Map Attack (JSMA).

    A sparse L0 perturbation attack adapted specifically for sequential time-series data.
    Computes input-output Jacobian matrices to determine saliency scores for individual
    (time_step, feature) coordinate pairs, iteratively perturbing the most influential
    time-series coordinates until target misclassification is achieved or the perturbation
    budget is exhausted.
    """

    metadata = AttackMetadata(
        name="jsma",
        domain="time_series",
        category="saliency",
        attack_type="white_box",
        requires_gradient=True,
        requires_loss=False,
        requires_model_weights=True,
        iterative=True,
        targeted_supported=True,
        untargeted_supported=True,
        query_based=False,
        supported_norms=["L0"],
        computational_cost="high",
        supported_model_types=["pytorch"],
    )

    def __init__(self, model: Any):
        """
        Initialize JSMA attack.

        Args:
            model: PyTorch model or BaseModelAdapter wrapping a PyTorch model.
        """
        super().__init__(model)
        self.raw_model = self._get_raw_model()
        check_model_compatibility(self.model, self.metadata)

    def validate_config(self, config: Optional[AttackConfig] = None) -> None:
        """
        Validate JSMA attack configuration parameters.
        """
        if config is None:
            return

        if (
            config.clip_min is not None
            and config.clip_max is not None
            and config.clip_min > config.clip_max
        ):
            raise AttackConfigurationError(
                f"clip_min ({config.clip_min}) cannot be greater than clip_max ({config.clip_max})"
            )

        params = config.params or {}
        max_iter = params.get(
            "max_iter",
            params.get(
                "max_iterations",
                params.get("max_locations", params.get("max_steps", 20)),
            ),
        )
        if max_iter <= 0:
            raise AttackExecutionError("JSMA max_iter must be greater than 0.")

        theta = params.get("theta", params.get("step_size", 0.1))
        if theta == 0:
            raise AttackConfigurationError("JSMA theta (step size) cannot be 0.")

        max_fraction = params.get("max_fraction", params.get("max_features_ratio", 1.0))
        if max_fraction <= 0 or max_fraction > 1.0:
            raise AttackConfigurationError(
                f"max_fraction must be in (0, 1], got {max_fraction}"
            )

    def generate(
        self,
        inputs: Any,
        labels: Any = None,
        config: Optional[AttackConfig] = None,
        target_labels: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Generate sparse time-series adversarial examples using Jacobian saliency.

        Args:
            inputs: Input time series (batch_size, sequence_length, num_features) or (batch, seq_len).
            labels: Ground truth class labels or target labels if targeted.
            config: AttackConfig containing theta (step size), max_iter, and bounds.
            target_labels: Explicit target class labels tensor.

        Returns:
            Adversarially perturbed time-series tensor.
        """
        if config is None:
            config = AttackConfig()

        self.validate_config(config)

        if not isinstance(inputs, torch.Tensor):
            raise AttackExecutionError(
                f"JSMA expects inputs to be torch.Tensor, got {type(inputs)}"
            )

        if inputs.dim() not in (2, 3):
            raise AttackExecutionError(
                f"JSMA expects inputs to have 2 or 3 dimensions (batch, seq_len[, features]), "
                f"got shape {inputs.shape} with {inputs.dim()} dimensions."
            )

        if inputs.size(0) == 0:
            raise AttackExecutionError("Input batch cannot be empty.")
        if inputs.size(1) == 0:
            raise AttackExecutionError("Sequence length must be greater than 0.")
        if not torch.isfinite(inputs).all():
            raise AttackExecutionError("JSMA inputs contain NaN or Inf values.")

        is_2d = inputs.dim() == 2
        if is_2d:
            inputs_3d = inputs.unsqueeze(-1)
        else:
            inputs_3d = inputs

        params = config.params or {}
        targeted = params.get("targeted", False)
        if target_labels is not None:
            targeted = True
            labels = target_labels

        device = get_device(self.raw_model)
        x_orig = inputs_3d.clone().detach().to(device)

        def forward_model(x_tensor: torch.Tensor) -> Any:
            m_in = x_tensor.squeeze(-1) if is_2d else x_tensor
            return self.raw_model(m_in)

        # Obtain initial predictions
        with torch.no_grad():
            initial_out = forward_model(x_orig)
            initial_logits = self._extract_logits(initial_out)
            initial_preds = initial_logits.argmax(dim=-1)
            num_classes = initial_logits.size(-1)

        if labels is None:
            if targeted:
                raise AttackExecutionError(
                    "Target labels must be provided for targeted JSMA."
                )
            true_labels = initial_preds
        else:
            if not isinstance(labels, torch.Tensor):
                labels = torch.as_tensor(labels, device=device)
            else:
                labels = labels.to(device)
            true_labels = labels

        if true_labels.size(0) != inputs.size(0):
            raise AttackExecutionError(
                f"Inputs batch size ({inputs.size(0)}) does not match labels batch size ({true_labels.size(0)})."
            )

        # Set up targets for each sample
        B, T, D = x_orig.shape
        if targeted:
            targets = true_labels
        else:
            # For untargeted attack: runner-up strategy (most likely non-true class)
            with torch.no_grad():
                logits_copy = initial_logits.clone()
                for b in range(B):
                    logits_copy[b, true_labels[b]] = -float("inf")
                targets = logits_copy.argmax(dim=-1)

        max_iter = int(
            params.get(
                "max_iter",
                params.get(
                    "max_iterations",
                    params.get("max_locations", params.get("max_steps", 20)),
                ),
            )
        )
        max_fraction = float(
            params.get("max_fraction", params.get("max_features_ratio", 1.0))
        )
        budget = min(max_iter, max(1, int(T * D * max_fraction)))

        theta = float(params.get("theta", params.get("step_size", 0.1)))

        f_min = params.get("feature_min")
        f_max = params.get("feature_max")
        bounds = params.get("feature_bounds", params.get("bounds"))
        if (
            bounds is not None
            and isinstance(bounds, (tuple, list))
            and len(bounds) == 2
        ):
            f_min = f_min if f_min is not None else bounds[0]
            f_max = f_max if f_max is not None else bounds[1]

        was_training = self.raw_model.training
        self.raw_model.eval()

        try:
            x_adv = x_orig.clone().detach()

            # Keep track of modified count per sample and finished status
            finished = torch.zeros(B, dtype=torch.bool, device=device)
            perturbation_counts = torch.zeros(B, dtype=torch.int32, device=device)
            modified_locations = [set() for _ in range(B)]

            for step in range(budget):
                # Check current status
                with torch.no_grad():
                    curr_out = forward_model(x_adv)
                    curr_logits = self._extract_logits(curr_out)
                    curr_preds = curr_logits.argmax(dim=-1)

                    if targeted:
                        finished = finished | (curr_preds == targets)
                    else:
                        finished = finished | (curr_preds != true_labels)

                if finished.all():
                    break

                # Compute Jacobian matrix: d(logits_c) / d(x_adv)
                with torch.backends.cudnn.flags(enabled=False):
                    x_adv.requires_grad_(True)
                    out = forward_model(x_adv)
                    logits = self._extract_logits(out)

                    jacobian_list = []
                    for c in range(num_classes):
                        grad_c = torch.autograd.grad(
                            logits[:, c].sum(),
                            x_adv,
                            retain_graph=(c < num_classes - 1),
                            create_graph=False,
                            allow_unused=True,
                        )[0]
                        if grad_c is None:
                            grad_c = torch.zeros_like(x_adv)
                        # Guard NaN/Inf in Jacobian
                        grad_c = torch.nan_to_num(grad_c, nan=0.0, posinf=0.0, neginf=0.0)
                        jacobian_list.append(grad_c)

                # Jacobian shape: (B, num_classes, T, D)
                jacobian = torch.stack(jacobian_list, dim=1).detach()
                x_adv = x_adv.detach()

                # Saliency calculation per sample for (time_step, feature)
                for b in range(B):
                    if finished[b]:
                        continue

                    t_star = targets[b].item()
                    j_b = jacobian[b]  # (num_classes, T, D)

                    # Alpha: target class derivative w.r.t (t, d)
                    alpha = j_b[t_star]  # (T, D)

                    # Beta: sum of other classes derivatives w.r.t (t, d)
                    beta = j_b.sum(dim=0) - alpha  # (T, D)

                    # Determine eligible coordinates that haven't been modified yet
                    eligible = torch.ones((T, D), dtype=torch.bool, device=device)
                    for t_prev, d_prev in modified_locations[b]:
                        eligible[t_prev, d_prev] = False

                    # Exclude coordinates that cannot be perturbed further due to clip bounds
                    if config.clip_max is not None and theta > 0:
                        eligible = eligible & (x_adv[b] < config.clip_max - 1e-6)
                    if config.clip_min is not None and theta < 0:
                        eligible = eligible & (x_adv[b] > config.clip_min + 1e-6)

                    if not eligible.any():
                        finished[b] = True
                        continue

                    # Saliency score computation
                    if theta > 0:
                        valid_mask = (alpha > 0) & (beta < 0) & eligible
                        saliency = torch.where(
                            valid_mask, alpha * beta.abs(), torch.zeros_like(alpha)
                        )
                    else:
                        valid_mask = (alpha < 0) & (beta > 0) & eligible
                        saliency = torch.where(
                            valid_mask, alpha.abs() * beta, torch.zeros_like(alpha)
                        )

                    # Mask out non-eligible coordinates with -inf
                    saliency = torch.where(
                        eligible, saliency, torch.full_like(saliency, -float("inf"))
                    )

                    # Fallback if no coordinate satisfies standard JSMA conditions
                    if saliency.max() <= 0:
                        saliency = torch.where(
                            eligible, alpha.abs(), torch.full_like(alpha, -float("inf"))
                        )

                    if not torch.isfinite(saliency).any() or saliency.max() <= -float(
                        "inf"
                    ):
                        finished[b] = True
                        continue

                    # Find best (time_step, feature) coordinate
                    flat_idx = saliency.view(-1).argmax().item()
                    t_idx = flat_idx // D
                    d_idx = flat_idx % D

                    # Apply perturbation to the single selected (t, d) location
                    x_adv[b, t_idx, d_idx] = x_adv[b, t_idx, d_idx] + theta
                    modified_locations[b].add((t_idx, d_idx))
                    perturbation_counts[b] += 1

                # Apply bounds to ensure perturbed locations stay in valid range
                x_adv = _apply_time_series_bounds(
                    x_adv,
                    clip_min=config.clip_min,
                    clip_max=config.clip_max,
                    feature_min=f_min,
                    feature_max=f_max,
                )

            # Record perturbation statistics into config params for reporting
            params["l0_perturbations"] = perturbation_counts.tolist()
            params["modified_counts"] = perturbation_counts.tolist()

            res = x_adv.detach()
            if is_2d:
                res = res.squeeze(-1)
            return res

        except AttackExecutionError:
            raise
        except Exception as e:
            raise AttackExecutionError(f"Error during JSMA generation: {str(e)}") from e
        finally:
            if hasattr(self.raw_model, "zero_grad"):
                self.raw_model.zero_grad(set_to_none=True)
            self.raw_model.train(was_training)

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Convenience alias for generate."""
        return self.generate(*args, **kwargs)


# Self-registration
register_attack("jsma", JSMA)
register_attack("ts_jsma", JSMA)
register_attack("time_series_jsma", JSMA)
