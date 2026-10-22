"""
Wavelet and Spectral adversarial attack for time-series models.
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
from app.attack_engine.utils.tensor_utils import get_device, project_lp
from app.attack_engine.utils.validation import check_model_compatibility
from app.attack_engine.attacks.time_series.pgd import _apply_time_series_bounds


def _dwt_haar_1d(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
    """
    Compute differentiable 1D Haar Discrete Wavelet Transform along the time dimension (dim=1).

    Args:
        x: Tensor of shape (B, T, D).

    Returns:
        cA: Approximation coefficients of shape (B*D, 1, T_half).
        cD: Detail coefficients of shape (B*D, 1, T_half).
        pad: Padding applied to sequence length.
        T: Original sequence length.
        B: Batch size.
        D: Number of features.
    """
    B, T, D = x.shape
    # Reshape to (B*D, 1, T) for 1D convolution
    signal = x.permute(0, 2, 1).reshape(B * D, 1, T)
    pad = 0
    if T % 2 != 0:
        signal = F.pad(signal, (0, 1), mode="replicate")
        pad = 1

    inv_sqrt2 = 1.0 / (2.0**0.5)
    h_low = torch.tensor([[[inv_sqrt2, inv_sqrt2]]], device=x.device, dtype=x.dtype)
    h_high = torch.tensor([[[inv_sqrt2, -inv_sqrt2]]], device=x.device, dtype=x.dtype)

    cA = F.conv1d(signal, h_low, stride=2)
    cD = F.conv1d(signal, h_high, stride=2)
    return cA, cD, pad, T, B, D


def _idwt_haar_1d(
    cA: torch.Tensor,
    cD: torch.Tensor,
    pad: int,
    orig_T: int,
    B: int,
    D: int,
) -> torch.Tensor:
    """
    Compute differentiable 1D Inverse Haar Discrete Wavelet Transform.

    Args:
        cA: Approximation coefficients of shape (B*D, 1, T_half).
        cD: Detail coefficients of shape (B*D, 1, T_half).
        pad: Padding applied during forward DWT.
        orig_T: Original sequence length before padding.
        B: Batch size.
        D: Number of features.

    Returns:
        Reconstructed tensor of shape (B, orig_T, D).
    """
    inv_sqrt2 = 1.0 / (2.0**0.5)
    g_low = torch.tensor([[[inv_sqrt2, inv_sqrt2]]], device=cA.device, dtype=cA.dtype)
    g_high = torch.tensor([[[inv_sqrt2, -inv_sqrt2]]], device=cA.device, dtype=cA.dtype)

    rec = F.conv_transpose1d(cA, g_low, stride=2) + F.conv_transpose1d(
        cD, g_high, stride=2
    )
    if pad > 0:
        rec = rec[..., :orig_T]

    return rec.reshape(B, D, orig_T).permute(0, 2, 1).contiguous()


class WaveletAttack(BaseAttack):
    """
    Spectral / Wavelet adversarial attack for time-series data.

    Generates adversarial time-series perturbations via gradient filtering in the
    spectral/wavelet domain (differentiable Haar DWT or real FFT) followed by
    reconstruction and projected gradient steps in input space.
    """

    metadata = AttackMetadata(
        name="wavelet",
        domain="time_series",
        category="spectral",
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
        Initialize WaveletAttack.

        Args:
            model: PyTorch model or BaseModelAdapter wrapping a PyTorch model.
        """
        super().__init__(model)
        self.raw_model = self._get_raw_model()
        check_model_compatibility(self.model, self.metadata)

    def validate_config(self, config: Optional[AttackConfig] = None) -> None:
        """
        Validate WaveletAttack configuration parameters.
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
                "WaveletAttack num_steps must be greater than 0."
            )

        norm = params.get("norm", "Linf")
        if norm.upper() not in ("LINF", "INF", "L2", "2"):
            raise AttackConfigurationError(
                f"Unsupported norm '{norm}'. Supported norms: ['Linf', 'L2']"
            )

        transform_type = params.get("transform_type", "dwt").lower()
        if transform_type not in ("dwt", "haar", "fft"):
            raise AttackConfigurationError(
                f"Unsupported transform_type '{transform_type}'. Supported: ['dwt', 'haar', 'fft']"
            )

        band = params.get("band", "all")
        if isinstance(band, str):
            band_lower = band.lower()
            if band_lower not in ("all", "low", "high", "mid", "approx", "detail"):
                raise AttackConfigurationError(
                    f"Unsupported frequency band '{band}'. Supported: ['all', 'low', 'high', 'mid', 'approx', 'detail']"
                )
            if transform_type in ("dwt", "haar") and band_lower == "mid":
                raise AttackConfigurationError(
                    "Band 'mid' is only supported for FFT transform; DWT supports 'low'/'approx', 'high'/'detail', or 'all'."
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
        Generate spectral/wavelet adversarial time-series examples.

        Args:
            inputs: Time-series input tensor (batch_size, sequence_length, num_features) or (batch, seq_len).
            labels: True class labels tensor (or target labels if targeted).
            config: AttackConfig containing epsilon, bounds, and spectral params.
            target_labels: Explicit target labels for targeted attack.

        Returns:
            Adversarially perturbed time-series tensor.
        """
        if config is None:
            config = AttackConfig()

        self.validate_config(config)

        if not isinstance(inputs, torch.Tensor):
            raise AttackExecutionError(
                f"WaveletAttack expects inputs to be torch.Tensor, got {type(inputs)}"
            )

        if inputs.dim() not in (2, 3):
            raise AttackExecutionError(
                f"WaveletAttack expects inputs to have 2 or 3 dimensions (batch, seq_len[, features]), "
                f"got shape {inputs.shape} with {inputs.dim()} dimensions."
            )

        if inputs.size(0) == 0:
            raise AttackExecutionError("Input batch cannot be empty.")
        if inputs.size(1) == 0:
            raise AttackExecutionError("Sequence length must be greater than 0.")
        if not torch.isfinite(inputs).all():
            raise AttackExecutionError(
                "WaveletAttack inputs contain NaN or Inf values."
            )

        # Ensure 3D internally: (B, T, D)
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

        # For model evaluation, handle 2D or 3D inputs based on original dimension
        def forward_model(x_tensor: torch.Tensor) -> Any:
            m_in = x_tensor.squeeze(-1) if is_2d else x_tensor
            return self.raw_model(m_in)

        if labels is None:
            if targeted:
                raise AttackExecutionError(
                    "Target labels must be provided for targeted WaveletAttack."
                )
            with torch.no_grad():
                out = forward_model(x_orig)
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
        transform_type = params.get("transform_type", "dwt").lower()
        band = params.get("band", "all")
        if isinstance(band, str):
            band = band.lower()

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

        # If epsilon <= 0, return bounded original input immediately
        if epsilon <= 0:
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

        if "alpha" in params:
            alpha = float(params["alpha"])
        elif "step_size" in params:
            alpha = float(params["step_size"])
        else:
            alpha = (2.0 * epsilon / num_steps) if num_steps > 0 else epsilon

        # Loss function
        loss_fn = (
            config.loss_fn if config.loss_fn is not None else nn.CrossEntropyLoss()
        )

        was_training = self.raw_model.training
        self.raw_model.eval()

        try:
            x_adv = x_orig.clone().detach()

            # Iterative optimization in transformed domain
            for _ in range(num_steps):
                x_adv.requires_grad_(True)

                outputs = forward_model(x_adv)
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

                # Differentiable transformation of gradients into spectral / wavelet domain
                if transform_type in ("dwt", "haar"):
                    cA_g, cD_g, pad_g, orig_T, B_g, D_g = _dwt_haar_1d(grad)

                    # Filter gradient coefficients based on selected band
                    if band in ("low", "approx"):
                        cD_g = torch.zeros_like(cD_g)
                    elif band in ("high", "detail"):
                        cA_g = torch.zeros_like(cA_g)

                    # Reconstruct filtered gradient back into signal space
                    filtered_grad = _idwt_haar_1d(cA_g, cD_g, pad_g, orig_T, B_g, D_g)

                elif transform_type == "fft":
                    T_len = grad.size(1)
                    fft_grad = torch.fft.rfft(grad, dim=1)
                    num_freqs = fft_grad.size(1)

                    mask = torch.ones_like(fft_grad.real)
                    if band == "low":
                        cutoff = max(1, num_freqs // 3)
                        mask[:, cutoff:, :] = 0.0
                    elif band == "high":
                        cutoff = max(1, (2 * num_freqs) // 3)
                        mask[:, :cutoff, :] = 0.0
                    elif band == "mid":
                        low_cutoff = max(1, num_freqs // 3)
                        high_cutoff = max(low_cutoff + 1, (2 * num_freqs) // 3)
                        mask[:, :low_cutoff, :] = 0.0
                        mask[:, high_cutoff:, :] = 0.0

                    filtered_fft = fft_grad * mask
                    filtered_grad = torch.fft.irfft(filtered_fft, n=T_len, dim=1)

                filtered_grad = torch.nan_to_num(
                    filtered_grad, nan=0.0, posinf=0.0, neginf=0.0
                )

                # Direction step
                if norm.upper() in ("LINF", "INF"):
                    step = filtered_grad.sign()
                else:
                    grad_flat = filtered_grad.view(filtered_grad.size(0), -1)
                    l2_norm = torch.norm(grad_flat, p=2, dim=1, keepdim=True).view(
                        -1, *([1] * (filtered_grad.dim() - 1))
                    )
                    step = filtered_grad / (l2_norm + 1e-12)

                if targeted:
                    x_adv = x_adv.detach() - alpha * step
                else:
                    x_adv = x_adv.detach() + alpha * step

                # Projection step back to Lp-ball around original time series
                x_adv = project_lp(x_adv, x_orig, epsilon=epsilon, norm=norm)

                # Clipping bounds
                x_adv = _apply_time_series_bounds(
                    x_adv,
                    clip_min=config.clip_min,
                    clip_max=config.clip_max,
                    feature_min=f_min,
                    feature_max=f_max,
                )

            # Final projection and bounds check
            x_adv = project_lp(x_adv, x_orig, epsilon=epsilon, norm=norm)
            x_adv = _apply_time_series_bounds(
                x_adv,
                clip_min=config.clip_min,
                clip_max=config.clip_max,
                feature_min=f_min,
                feature_max=f_max,
            )

            res = x_adv.detach().contiguous()
            if is_2d:
                res = res.squeeze(-1).contiguous()
            return res

        except AttackExecutionError:
            raise
        except Exception as e:
            raise AttackExecutionError(
                f"Error during WaveletAttack generation: {str(e)}"
            ) from e
        finally:
            if hasattr(self.raw_model, "zero_grad"):
                self.raw_model.zero_grad(set_to_none=True)
            self.raw_model.train(was_training)

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Convenience alias for generate."""
        return self.generate(*args, **kwargs)


# Self-registration
register_attack("wavelet", WaveletAttack)
register_attack("wavelet_attack", WaveletAttack)
register_attack("spectral", WaveletAttack)
register_attack("ts_wavelet", WaveletAttack)
