"""
Projected Gradient Descent (PGD) adversarial attack for time-series data.
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
from app.attack_engine.utils.tensor_utils import clip_tensor, get_device, project_lp
from app.attack_engine.utils.validation import check_model_compatibility


def _apply_time_series_bounds(
    x: torch.Tensor,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    feature_min: Optional[Any] = None,
    feature_max: Optional[Any] = None,
) -> torch.Tensor:
    """
    Apply scalar and/or feature-wise bounds to a time-series tensor.
    Supports both 2D (B, T) and 3D (B, T, D) inputs.
    """
    if clip_min is not None or clip_max is not None:
        x = clip_tensor(x, clip_min, clip_max)

    if feature_min is not None or feature_max is not None:
        device = x.device
        dtype = x.dtype
        f_min = None
        f_max = None

        if feature_min is not None:
            if not isinstance(feature_min, torch.Tensor):
                f_min = torch.as_tensor(feature_min, device=device, dtype=dtype)
            else:
                f_min = feature_min.to(device=device, dtype=dtype)

            # Match dimension of time series (B, T, D) or (B, T)
            if f_min.dim() == 1 and x.dim() == 3:
                if f_min.size(0) != x.size(-1):
                    raise AttackConfigurationError(
                        f"feature_min length ({f_min.size(0)}) does not match feature dimension ({x.size(-1)})."
                    )
                f_min = f_min.view(1, 1, -1)
            elif f_min.dim() == 1 and x.dim() == 2:
                if f_min.size(0) != x.size(1):
                    raise AttackConfigurationError(
                        f"feature_min length ({f_min.size(0)}) does not match sequence length ({x.size(1)})."
                    )
                f_min = f_min.unsqueeze(0)
            elif f_min.dim() == 2 and x.dim() == 3:
                f_min = f_min.unsqueeze(0)

            x = torch.max(x, f_min)

        if feature_max is not None:
            if not isinstance(feature_max, torch.Tensor):
                f_max = torch.as_tensor(feature_max, device=device, dtype=dtype)
            else:
                f_max = feature_max.to(device=device, dtype=dtype)

            if f_max.dim() == 1 and x.dim() == 3:
                if f_max.size(0) != x.size(-1):
                    raise AttackConfigurationError(
                        f"feature_max length ({f_max.size(0)}) does not match feature dimension ({x.size(-1)})."
                    )
                f_max = f_max.view(1, 1, -1)
            elif f_max.dim() == 1 and x.dim() == 2:
                if f_max.size(0) != x.size(1):
                    raise AttackConfigurationError(
                        f"feature_max length ({f_max.size(0)}) does not match sequence length ({x.size(1)})."
                    )
                f_max = f_max.unsqueeze(0)
            elif f_max.dim() == 2 and x.dim() == 3:
                f_max = f_max.unsqueeze(0)

            x = torch.min(x, f_max)

        if f_min is not None and f_max is not None:
            if (f_min > f_max).any():
                raise AttackConfigurationError(
                    "feature_min must be less than or equal to feature_max."
                )

    return x


class TimeSeriesPGD(BaseAttack):
    """
    Projected Gradient Descent (PGD) attack for time-series models.

    Supports multivariate (batch_size, sequence_length, num_features) and
    univariate time-series data with L-infinity and L2 norm constraints.
    """

    metadata = AttackMetadata(
        name="time_series_pgd",
        domain="time_series",
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
        Initialize TimeSeriesPGD attack.

        Args:
            model: PyTorch model or BaseModelAdapter wrapping a PyTorch model.
        """
        super().__init__(model)
        self.raw_model = self._get_raw_model()
        check_model_compatibility(self.model, self.metadata)

    def validate_config(self, config: Optional[AttackConfig] = None) -> None:
        """
        Validate attack configuration parameters.
        """
        if config is None:
            return

        if config.epsilon < 0:
            raise AttackConfigurationError(
                f"Epsilon cannot be negative, got {config.epsilon}"
            )

        if (
            config.clip_min is not None
            and config.clip_max is not None
            and config.clip_min > config.clip_max
        ):
            raise AttackConfigurationError(
                f"clip_min ({config.clip_min}) cannot be greater than clip_max ({config.clip_max})"
            )

        params = config.params or {}
        num_steps = params.get(
            "num_steps", params.get("steps", params.get("iters", 10))
        )
        if num_steps <= 0:
            raise AttackExecutionError("PGD num_steps must be greater than 0.")

        norm = params.get("norm", "Linf")
        if norm.upper() not in ("LINF", "INF", "L2", "2"):
            raise AttackConfigurationError(
                f"Unsupported norm '{norm}'. Supported norms: ['Linf', 'L2']"
            )

        if "alpha" in params and params["alpha"] <= 0:
            raise AttackConfigurationError(
                f"Step size (alpha) must be greater than 0, got {params['alpha']}"
            )
        if "step_size" in params and params["step_size"] <= 0:
            raise AttackConfigurationError(
                f"step_size must be greater than 0, got {params['step_size']}"
            )

        # Validate feature-wise bounds if provided
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

        if f_min is not None and f_max is not None:
            t_min = torch.as_tensor(f_min)
            t_max = torch.as_tensor(f_max)
            if (t_min > t_max).any():
                raise AttackConfigurationError(
                    "feature_min cannot be greater than feature_max."
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
        Generate time-series adversarial examples using PGD.

        Args:
            inputs: Input time-series tensor of shape (batch_size, sequence_length, num_features)
                    or (batch_size, sequence_length).
            labels: True class labels tensor (or target labels if targeted=True).
            config: Optional AttackConfig instance.
            target_labels: Explicit target labels for targeted attack.

        Returns:
            Adversarially perturbed time-series tensor of matching shape.
        """
        if config is None:
            config = AttackConfig()

        self.validate_config(config)

        if not isinstance(inputs, torch.Tensor):
            raise AttackExecutionError(
                f"TimeSeriesPGD expects inputs to be torch.Tensor, got {type(inputs)}"
            )

        if inputs.dim() not in (2, 3):
            raise AttackExecutionError(
                f"TimeSeriesPGD expects inputs to have 2 or 3 dimensions (batch, seq_len[, features]), "
                f"got shape {inputs.shape} with {inputs.dim()} dimensions."
            )

        if inputs.size(0) == 0:
            raise AttackExecutionError("Input batch cannot be empty.")
        if inputs.size(1) == 0:
            raise AttackExecutionError("Sequence length must be greater than 0.")
        if not torch.isfinite(inputs).all():
            raise AttackExecutionError(
                "TimeSeriesPGD inputs contain NaN or Inf values."
            )

        params = config.params or {}
        targeted = params.get("targeted", False)
        if target_labels is not None:
            targeted = True
            labels = target_labels

        device = get_device(self.raw_model)
        x_orig = inputs.clone().detach().to(device)

        if labels is None:
            if targeted:
                raise AttackExecutionError(
                    "Target labels must be provided for targeted TimeSeriesPGD attack."
                )
            with torch.no_grad():
                out = self.raw_model(x_orig)
                logits = out.logits if hasattr(out, "logits") else out
                labels = logits.argmax(dim=-1)
        else:
            if not isinstance(labels, torch.Tensor):
                labels = torch.as_tensor(labels, device=device)
            else:
                labels = labels.to(device)

        if labels.size(0) != inputs.size(0):
            raise AttackExecutionError(
                f"Inputs batch size ({inputs.size(0)}) does not match labels batch size ({labels.size(0)})."
            )

        epsilon = float(config.epsilon)
        num_steps = params.get(
            "num_steps", params.get("steps", params.get("iters", 10))
        )
        norm = params.get("norm", "Linf")
        random_start = params.get("random_start", True)

        # Feature-wise bounds
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

        # If epsilon <= 0, no perturbation is mathematically allowed
        if epsilon <= 0:
            return (
                _apply_time_series_bounds(
                    x_orig,
                    clip_min=config.clip_min,
                    clip_max=config.clip_max,
                    feature_min=f_min,
                    feature_max=f_max,
                )
                .detach()
                .contiguous()
            )

        if "alpha" in params:
            alpha = float(params["alpha"])
        elif "step_size" in params:
            alpha = float(params["step_size"])
        else:
            alpha = (2.0 * epsilon / num_steps) if num_steps > 0 else epsilon

        # Loss function
        if config.loss_fn is not None:
            loss_fn = config.loss_fn
        else:
            loss_fn = nn.CrossEntropyLoss()

        was_training = self.raw_model.training
        self.raw_model.eval()

        try:
            x_adv = x_orig.clone().detach()

            # Random initialization within epsilon-ball
            if random_start and epsilon > 0:
                if norm.upper() in ("LINF", "INF"):
                    noise = torch.zeros_like(x_adv).uniform_(-epsilon, epsilon)
                else:
                    noise = torch.randn_like(x_adv)
                    noise_flat = noise.reshape(noise.size(0), -1)
                    l2_norm = torch.norm(noise_flat, p=2, dim=1, keepdim=True).view(
                        -1, *([1] * (noise.dim() - 1))
                    )
                    radius = (
                        torch.rand(noise.size(0), device=device).view(
                            -1, *([1] * (noise.dim() - 1))
                        )
                        * epsilon
                    )
                    noise = (noise / (l2_norm + 1e-12)) * radius

                x_adv = x_adv + noise
                x_adv = project_lp(
                    x_adv.contiguous(), x_orig, epsilon=epsilon, norm=norm
                )
                x_adv = _apply_time_series_bounds(
                    x_adv,
                    clip_min=config.clip_min,
                    clip_max=config.clip_max,
                    feature_min=f_min,
                    feature_max=f_max,
                )

            # Iterative PGD optimization
            for _ in range(num_steps):
                x_adv.requires_grad_(True)

                outputs = self.raw_model(x_adv)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = loss_fn(logits, labels)

                grad = torch.autograd.grad(
                    loss,
                    x_adv,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )[0]

                if grad is None:
                    grad = torch.zeros_like(x_adv)

                # Numerical stability: protect against NaN/Inf in gradients
                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

                # Direction step
                if norm.upper() in ("LINF", "INF"):
                    step = grad.sign()
                else:
                    grad_flat = grad.reshape(grad.size(0), -1)
                    l2_norm = torch.norm(grad_flat, p=2, dim=1, keepdim=True).view(
                        -1, *([1] * (grad.dim() - 1))
                    )
                    step = grad / (l2_norm + 1e-12)

                if targeted:
                    x_adv = x_adv.detach() - alpha * step
                else:
                    x_adv = x_adv.detach() + alpha * step

                # Projection step back to Lp-ball around original time series
                x_adv = project_lp(
                    x_adv.contiguous(), x_orig, epsilon=epsilon, norm=norm
                )

                # Clipping to valid scalar/feature-wise range
                x_adv = _apply_time_series_bounds(
                    x_adv,
                    clip_min=config.clip_min,
                    clip_max=config.clip_max,
                    feature_min=f_min,
                    feature_max=f_max,
                )

            # Final projection and bounds guarantee
            x_adv = project_lp(x_adv.contiguous(), x_orig, epsilon=epsilon, norm=norm)
            x_adv = _apply_time_series_bounds(
                x_adv,
                clip_min=config.clip_min,
                clip_max=config.clip_max,
                feature_min=f_min,
                feature_max=f_max,
            )
            return x_adv.detach().contiguous()

        except AttackExecutionError:
            raise
        except Exception as e:
            raise AttackExecutionError(
                f"Error during TimeSeriesPGD generation: {str(e)}"
            ) from e
        finally:
            if hasattr(self.raw_model, "zero_grad"):
                self.raw_model.zero_grad(set_to_none=True)
            self.raw_model.train(was_training)

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Convenience alias for generate."""
        return self.generate(*args, **kwargs)


# Alias
PGD = TimeSeriesPGD

# Registration in Attack Engine Registry
register_attack("time_series_pgd", TimeSeriesPGD)
register_attack("ts_pgd", TimeSeriesPGD)
register_attack("ts-pgd", TimeSeriesPGD)
