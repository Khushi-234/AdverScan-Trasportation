"""
Soft-DTW and temporal warping adversarial attack for time-series models.
"""

from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def compute_soft_dtw(
    x: torch.Tensor,
    y: torch.Tensor,
    gamma: float = 0.5,
    downsample_threshold: Optional[int] = 64,
) -> torch.Tensor:
    """
    Compute differentiable Soft-DTW distance between two sequences of shape (B, T, D).

    Args:
        x: First sequence tensor of shape (B, T, D).
        y: Second sequence tensor of shape (B, T, D).
        gamma: Smoothing temperature parameter (> 0).
        downsample_threshold: Threshold above which long sequences are downsampled
            to preserve O(T_sub^2) computational efficiency.

    Returns:
        Differentiable scalar Soft-DTW distance averaged across the batch.
    """
    B, T, D = x.shape
    if T <= 1:
        return torch.mean(torch.sum(torch.square(x - y), dim=-1))

    # Downsample temporally if sequence length is very large for efficiency O(T_sub^2)
    if (
        downsample_threshold is not None
        and downsample_threshold > 0
        and T > downsample_threshold
    ):
        stride = max(1, T // downsample_threshold)
        x_sub = x[:, ::stride, :]
        y_sub = y[:, ::stride, :]
        T_sub = x_sub.size(1)
    else:
        x_sub = x
        y_sub = y
        T_sub = T

    dist = torch.cdist(x_sub, y_sub, p=2) ** 2
    device = x.device
    dtype = x.dtype

    # Dynamic programming table with differentiable operations
    R = torch.full((B, T_sub + 1, T_sub + 1), 1e6, device=device, dtype=dtype)
    R[:, 0, 0] = 0.0

    inv_gamma = 1.0 / max(gamma, 1e-6)

    for i in range(1, T_sub + 1):
        for j in range(1, T_sub + 1):
            r0 = R[:, i - 1, j - 1]
            r1 = R[:, i - 1, j]
            r2 = R[:, i, j - 1]
            stacked = torch.stack(
                [-r0 * inv_gamma, -r1 * inv_gamma, -r2 * inv_gamma], dim=1
            )
            softmin = -gamma * torch.logsumexp(stacked, dim=1)
            R[:, i, j] = dist[:, i - 1, j - 1] + softmin

    return torch.mean(R[:, T_sub, T_sub])


def _warp_time_series_1d(
    x: torch.Tensor,
    theta_warp: torch.Tensor,
    max_warp: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiably warp the temporal dimension of a time-series tensor.

    Args:
        x: Input tensor of shape (B, T, D).
        theta_warp: Warping parameters tensor of shape (B, T).
        max_warp: Maximum fraction of sequence length shift in normalized [-1, 1] grid.

    Returns:
        warped: Warped tensor of shape (B, T, D).
        warp_disp: Warp displacement field.
    """
    B, T, D = x.shape
    device = x.device
    dtype = x.dtype

    if T <= 1:
        return x.clone(), torch.zeros((B, T), device=device, dtype=dtype)

    # Reshape input to (B, D, 1, T) for torch.nn.functional.grid_sample
    x_in = x.permute(0, 2, 1).unsqueeze(2)

    # Base grid along time in [-1, 1]
    base_grid_1d = torch.linspace(-1.0, 1.0, steps=T, device=device, dtype=dtype)
    base_grid = base_grid_1d.view(1, 1, T, 1).expand(B, 1, T, 1)

    # Differentiable bounded displacement field
    warp_disp = max_warp * torch.tanh(theta_warp).unsqueeze(1).unsqueeze(-1)

    grid_x = torch.clamp(base_grid + warp_disp, min=-1.0, max=1.0)
    grid_y = torch.zeros_like(grid_x)
    grid = torch.cat([grid_x, grid_y], dim=-1)

    # Apply 1D linear temporal resampling
    warped = F.grid_sample(
        x_in, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    # Reshape back to (B, T, D)
    warped = warped.squeeze(2).permute(0, 2, 1).contiguous()
    return warped, warp_disp.squeeze(1).squeeze(-1).contiguous()


class SoftDTWAttack(BaseAttack):
    """
    Soft-DTW and Temporal Warping adversarial attack for time-series.

    Modifies the timing and phase structure of the time-series sequence via
    differentiable temporal resampling, constrained and regularized by Soft-DTW
    alignment distance to preserve temporal plausibility and feature relationships.
    """

    metadata = AttackMetadata(
        name="soft_dtw",
        domain="time_series",
        category="temporal_warping",
        attack_type="white_box",
        requires_gradient=True,
        requires_loss=True,
        requires_model_weights=True,
        iterative=True,
        targeted_supported=True,
        untargeted_supported=True,
        query_based=False,
        supported_norms=["Linf", "L2"],
        computational_cost="high",
        supported_model_types=["pytorch"],
    )

    def __init__(self, model: Any):
        """
        Initialize SoftDTWAttack.

        Args:
            model: PyTorch model or BaseModelAdapter wrapping a PyTorch model.
        """
        super().__init__(model)
        self.raw_model = self._get_raw_model()
        check_model_compatibility(self.model, self.metadata)

    def validate_config(self, config: Optional[AttackConfig] = None) -> None:
        """
        Validate SoftDTWAttack configuration parameters.
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
            raise AttackExecutionError(
                "SoftDTWAttack num_steps must be greater than 0."
            )

        max_warp = params.get("max_warp", params.get("epsilon_warp", config.epsilon))
        if max_warp < 0:
            raise AttackConfigurationError(
                f"max_warp cannot be negative, got {max_warp}"
            )

        gamma = params.get("gamma", 0.5)
        if gamma <= 0:
            raise AttackConfigurationError(
                f"Soft-DTW gamma must be positive, got {gamma}"
            )

        lambda_dtw = params.get("lambda_dtw", params.get("dtw_weight", 0.05))
        if lambda_dtw < 0:
            raise AttackConfigurationError(
                f"lambda_dtw cannot be negative, got {lambda_dtw}"
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
        Generate temporally warped adversarial time series using Soft-DTW optimization.

        Args:
            inputs: Time-series tensor of shape (batch_size, sequence_length, num_features) or (batch, seq_len).
            labels: True class labels tensor (or target labels if targeted).
            config: AttackConfig containing epsilon/warp magnitude, regularization, and step parameters.
            target_labels: Explicit target labels for targeted attack.

        Returns:
            Temporally warped adversarial time-series tensor.
        """
        if config is None:
            config = AttackConfig()

        self.validate_config(config)

        if not isinstance(inputs, torch.Tensor):
            raise AttackExecutionError(
                f"SoftDTWAttack expects inputs to be torch.Tensor, got {type(inputs)}"
            )

        if inputs.dim() not in (2, 3):
            raise AttackExecutionError(
                f"SoftDTWAttack expects inputs to have 2 or 3 dimensions (batch, seq_len[, features]), "
                f"got shape {inputs.shape} with {inputs.dim()} dimensions."
            )

        if inputs.size(0) == 0:
            raise AttackExecutionError("Input batch cannot be empty.")
        if inputs.size(1) == 0:
            raise AttackExecutionError("Sequence length must be greater than 0.")
        if not torch.isfinite(inputs).all():
            raise AttackExecutionError(
                "SoftDTWAttack inputs contain NaN or Inf values."
            )

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

        if labels is None:
            if targeted:
                raise AttackExecutionError(
                    "Target labels must be provided for targeted SoftDTWAttack."
                )
            with torch.no_grad():
                out = forward_model(x_orig)
                logits = self._extract_logits(out)
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

        B, T, D = x_orig.shape
        num_steps = params.get(
            "num_steps", params.get("steps", params.get("iters", 10))
        )
        max_warp = float(
            params.get(
                "max_warp",
                params.get(
                    "epsilon_warp", config.epsilon if config.epsilon > 0 else 0.2
                ),
            )
        )
        gamma = float(params.get("gamma", 0.5))
        lambda_dtw = float(params.get("lambda_dtw", params.get("dtw_weight", 0.05)))
        lambda_smooth = float(params.get("lambda_smooth", 0.01))
        lr = float(params.get("lr", params.get("alpha", 0.1)))
        downsample_threshold = int(params.get("downsample_threshold", 64))

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

        # If warping magnitude <= 0 or sequence length <= 1, return bounded original
        if max_warp <= 0 or T <= 1:
            clamped = (
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
            if is_2d:
                clamped = clamped.squeeze(-1).contiguous()
            return clamped

        loss_fn = (
            config.loss_fn if config.loss_fn is not None else nn.CrossEntropyLoss()
        )

        was_training = self.raw_model.training
        self.raw_model.eval()

        try:
            # Warping parameters initialized to zero (identity mapping)
            theta_warp = torch.zeros(
                B, T, device=device, dtype=x_orig.dtype, requires_grad=True
            )
            optimizer = torch.optim.Adam([theta_warp], lr=lr)

            for _ in range(num_steps):
                optimizer.zero_grad()

                x_warped, warp_disp = _warp_time_series_1d(
                    x_orig, theta_warp, max_warp=max_warp
                )

                with torch.backends.cudnn.flags(enabled=False):
                    outputs = forward_model(x_warped)
                    logits = self._extract_logits(outputs)
                    cls_loss = loss_fn(logits, labels)

                    # Differentiable Soft-DTW alignment loss to keep sequence temporally close
                    dtw_loss = compute_soft_dtw(
                        x_orig,
                        x_warped,
                        gamma=gamma,
                        downsample_threshold=downsample_threshold,
                    )

                    # Differentiable smoothness penalty on temporal displacement field
                    smooth_loss = torch.mean(
                        torch.square(warp_disp[:, 1:] - warp_disp[:, :-1])
                    )

                    if targeted:
                        # Minimize classification loss towards target + Soft-DTW distance
                        total_loss = (
                            cls_loss + lambda_dtw * dtw_loss + lambda_smooth * smooth_loss
                        )
                    else:
                        # Maximize classification loss away from true - minimize Soft-DTW distance
                        total_loss = (
                            -cls_loss + lambda_dtw * dtw_loss + lambda_smooth * smooth_loss
                        )

                    total_loss.backward()

                if theta_warp.grad is None:
                    raise AttackExecutionError(
                        "Gradients w.r.t temporal warping parameters were not computed."
                    )

                theta_warp.grad = torch.nan_to_num(
                    theta_warp.grad, nan=0.0, posinf=0.0, neginf=0.0
                )

                optimizer.step()

            # Final warped sequence with bounds clipping
            with torch.no_grad():
                final_warped, _ = _warp_time_series_1d(
                    x_orig, theta_warp, max_warp=max_warp
                )
                final_warped = _apply_time_series_bounds(
                    final_warped,
                    clip_min=config.clip_min,
                    clip_max=config.clip_max,
                    feature_min=f_min,
                    feature_max=f_max,
                )

            res = final_warped.detach().contiguous()
            if is_2d:
                res = res.squeeze(-1).contiguous()
            return res

        except AttackExecutionError:
            raise
        except Exception as e:
            raise AttackExecutionError(
                f"Error during SoftDTWAttack generation: {str(e)}"
            ) from e
        finally:
            if hasattr(self.raw_model, "zero_grad"):
                self.raw_model.zero_grad(set_to_none=True)
            self.raw_model.train(was_training)

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Convenience alias for generate."""
        return self.generate(*args, **kwargs)


# Self-registration
register_attack("soft_dtw", SoftDTWAttack)
register_attack("soft_dtw_attack", SoftDTWAttack)
register_attack("ts_soft_dtw", SoftDTWAttack)
register_attack("dtw", SoftDTWAttack)
