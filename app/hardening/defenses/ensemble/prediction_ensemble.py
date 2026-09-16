"""
Prediction Ensemble Defense implementation for Module 7 (Hardening).

Combines predictions across multiple defensive transformations and pipelines
(e.g., raw input, smoothed input, quantized input, filtered input) via voting or probability averaging.
"""

import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.hardening.defenses.base import BaseDefense
from app.hardening.hardening_result import HardeningMetadata, HardeningResult
from app.hardening.exceptions import DefenseExecutionError, HardeningConfigurationError


class PredictionEnsembleModelWrapper(nn.Module):
    """
    nn.Module wrapper evaluating input across multiple defensive transforms and aggregating predictions.
    """

    def __init__(
        self,
        base_model: nn.Module,
        transforms: Sequence[Callable[[torch.Tensor], torch.Tensor]],
        voting: str = "soft",
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.transforms = list(transforms)
        self.voting = str(voting).lower().strip()

        if not self.transforms:
            raise HardeningConfigurationError("PredictionEnsembleModelWrapper requires at least one transform view.")

        # Weight validation
        if weights is not None:
            weights_list = [float(w) for w in weights]
            if len(weights_list) != len(self.transforms):
                raise HardeningConfigurationError(
                    f"Number of weights ({len(weights_list)}) must match number of transforms ({len(self.transforms)})."
                )
            if any(w < 0.0 for w in weights_list):
                raise HardeningConfigurationError("Ensemble weights must be non-negative.")
            total_w = sum(weights_list)
            if total_w <= 0.0:
                raise HardeningConfigurationError("Sum of ensemble weights must be strictly positive.")
            self.weights = [w / total_w for w in weights_list]
        else:
            self.weights = [1.0 / len(self.transforms)] * len(self.transforms)

    def _aggregate_outputs(self, all_logits: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate view predictions according to configured voting strategy.
        Supports both multi-class and binary classification without tie-breaking bias.
        """
        if len(all_logits) == 1:
            return all_logits[0]

        # Multi-class classification outputs (B, C) with C > 1
        if all_logits[0].ndim > 1 and all_logits[0].size(1) > 1:
            probs = [F.softmax(lg, dim=-1) for lg in all_logits]
            num_classes = all_logits[0].size(-1)

            if self.voting in ("soft", "average", "mean"):
                weighted_p = torch.zeros_like(probs[0])
                for p, w in zip(probs, self.weights):
                    weighted_p += p * w
                return torch.log(weighted_p.clamp_min(1e-10))

            elif self.voting == "hard":
                # Compute vote proportions for each class without torch.mode tie bias
                preds = [torch.argmax(p, dim=-1) for p in probs]  # List of (B,)
                vote_shares = torch.zeros_like(probs[0])
                for pred, w in zip(preds, self.weights):
                    vote_shares += F.one_hot(pred, num_classes=num_classes).float() * w
                return torch.log(vote_shares.clamp_min(1e-10))

        # Binary classification outputs (B, 1) or (B,)
        if all_logits[0].ndim == 1 or (all_logits[0].ndim == 2 and all_logits[0].size(1) == 1):
            probs_pos = [torch.sigmoid(lg) for lg in all_logits]

            if self.voting in ("soft", "average", "mean"):
                weighted_pos = torch.zeros_like(probs_pos[0])
                for p, w in zip(probs_pos, self.weights):
                    weighted_pos += p * w
                weighted_pos = torch.clamp(weighted_pos, 1e-7, 1.0 - 1e-7)
                return torch.logit(weighted_pos)

            elif self.voting == "hard":
                vote_pos = torch.zeros_like(probs_pos[0])
                for p, w in zip(probs_pos, self.weights):
                    vote_pos += (p >= 0.5).float() * w
                vote_pos = torch.clamp(vote_pos, 1e-7, 1.0 - 1e-7)
                return torch.logit(vote_pos)

        # Fallback for regression / raw representations
        weighted = torch.zeros_like(all_logits[0])
        for lg, w in zip(all_logits, self.weights):
            weighted += lg * w
        return weighted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass running model on each transformed view and aggregating."""
        all_logits: List[torch.Tensor] = []
        for t_fn in self.transforms:
            transformed_x = t_fn(x)
            out = self.base_model(transformed_x)
            if isinstance(out, tuple):
                out = out[0]
            all_logits.append(out)

        if not all_logits:
            out = self.base_model(x)
            return out[0] if isinstance(out, tuple) else out

        return self._aggregate_outputs(all_logits)

    def predict_with_metadata(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Evaluate inputs and return consensus rate and prediction variance across views
        without duplicate forward passes.
        """
        with torch.no_grad():
            all_logits: List[torch.Tensor] = []
            for t_fn in self.transforms:
                transformed_x = t_fn(x)
                out = self.base_model(transformed_x)
                if isinstance(out, tuple):
                    out = out[0]
                all_logits.append(out)

            if not all_logits:
                out = self.base_model(x)
                if isinstance(out, tuple):
                    out = out[0]
                return {"aggregated_logits": out}

            # Directly reuse all_logits to avoid redundant forward computation
            aggregated = self._aggregate_outputs(all_logits)

            result: Dict[str, Any] = {
                "aggregated_logits": aggregated,
            }

            # Multi-class consensus rate without torch.mode tie-breaking bias
            if all_logits[0].ndim > 1 and all_logits[0].size(1) > 1:
                probs = [F.softmax(lg, dim=-1) for lg in all_logits]
                preds = [torch.argmax(p, dim=-1) for p in probs]
                num_classes = all_logits[0].size(-1)

                vote_shares = torch.zeros_like(probs[0])
                for pred, w in zip(preds, self.weights):
                    vote_shares += F.one_hot(pred, num_classes=num_classes).float() * w

                agreements = vote_shares.max(dim=-1).values
                result["consensus_rate"] = agreements
                result["mean_consensus"] = float(agreements.mean().item())

            # Binary consensus rate
            elif all_logits[0].ndim == 1 or (all_logits[0].ndim == 2 and all_logits[0].size(1) == 1):
                probs_pos = [torch.sigmoid(lg) for lg in all_logits]
                vote_pos = torch.zeros_like(probs_pos[0])
                for p, w in zip(probs_pos, self.weights):
                    vote_pos += (p >= 0.5).float() * w
                agreements = torch.maximum(vote_pos, 1.0 - vote_pos).squeeze(-1)
                result["consensus_rate"] = agreements
                result["mean_consensus"] = float(agreements.mean().item())

            return result


class PredictionEnsembleDefense(BaseDefense):
    """
    Prediction Ensemble Defense.

    Applies multiple diverse input transformations / defense filters to each sample,
    gathers predictions from the model, and combines them through soft or hard voting.
    """

    supported_domains: List[str] = ["image", "nlp", "tabular", "audio"]
    defense_family: str = "ensemble"

    def __init__(
        self,
        voting: str = "soft",
        noise_std: float = 0.03,
        num_views: int = 3,
        clip_min: Optional[float] = 0.0,
        clip_max: Optional[float] = 1.0,
        transforms: Optional[Sequence[Callable[[torch.Tensor], torch.Tensor]]] = None,
        weights: Optional[Sequence[float]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize PredictionEnsembleDefense.

        Args:
            voting: Voting mechanism ('soft', 'hard', 'mean').
            noise_std: Noise std for synthetic perturbation views if transforms not supplied.
            num_views: Number of diverse views evaluated per sample (must be at least 1).
            clip_min: Optional lower bound for clipping noise views (set None for standardized inputs).
            clip_max: Optional upper bound for clipping noise views.
            transforms: Optional custom sequence of transform callables.
            weights: Optional weight per view.
            config: Optional configuration dictionary.
        """
        cfg = config or {}
        super().__init__(
            name="prediction_ensemble",
            defense_type="ensemble",
            config=cfg,
            supported_domains=self.supported_domains,
            defense_family=self.defense_family,
        )
        self.voting = str(cfg.get("voting", voting)).lower().strip()
        self.noise_std = float(cfg.get("noise_std", noise_std))
        self.num_views = int(cfg.get("num_views", num_views))

        raw_clip_min = cfg.get("clip_min", clip_min)
        self.clip_min = float(raw_clip_min) if raw_clip_min is not None else None
        raw_clip_max = cfg.get("clip_max", clip_max)
        self.clip_max = float(raw_clip_max) if raw_clip_max is not None else None

        self.custom_transforms = list(transforms) if transforms is not None else None
        raw_weights = cfg.get("weights", weights)
        self.weights = [float(w) for w in raw_weights] if raw_weights is not None else None

        # Parameter validation
        if self.num_views < 1:
            raise HardeningConfigurationError(f"num_views must be at least 1, got {self.num_views}")

        if self.noise_std < 0.0:
            raise HardeningConfigurationError(f"noise_std must be non-negative, got {self.noise_std}")

        if self.clip_min is not None and self.clip_max is not None and self.clip_min >= self.clip_max:
            raise HardeningConfigurationError(
                f"clip_min ({self.clip_min}) must be strictly less than clip_max ({self.clip_max})"
            )

        if self.voting not in ("soft", "hard", "mean", "average"):
            raise HardeningConfigurationError(
                f"Unsupported voting '{self.voting}'. Supported: ['soft', 'hard', 'mean']"
            )

        if self.custom_transforms is not None and len(self.custom_transforms) < 1:
            raise HardeningConfigurationError("transforms must contain at least one callable.")

    def _build_default_transforms(self) -> List[Callable[[torch.Tensor], torch.Tensor]]:
        """
        Build a diverse set of defensive view transformations dynamically for `num_views`.
        """
        if self.num_views == 1:
            return [lambda x: x]

        def _apply_noise(x: torch.Tensor, std: float) -> torch.Tensor:
            noisy = x + torch.randn_like(x) * std
            if self.clip_min is not None and self.clip_max is not None:
                return torch.clamp(noisy, self.clip_min, self.clip_max)
            return noisy

        def _apply_quantization(x: torch.Tensor, levels: int) -> torch.Tensor:
            if self.clip_min is not None and self.clip_max is not None:
                rng = self.clip_max - self.clip_min
                if rng > 0:
                    norm = (torch.clamp(x, self.clip_min, self.clip_max) - self.clip_min) / rng
                    quantized = torch.round(norm * (levels - 1)) / (levels - 1)
                    return quantized * rng + self.clip_min
            return torch.round(x * levels) / levels

        transforms: List[Callable[[torch.Tensor], torch.Tensor]] = [
            lambda x: x,  # View 1: Clean identity view
            lambda x, s=self.noise_std: _apply_noise(x, s),  # View 2: Base light noise
        ]

        if self.num_views >= 3:
            transforms.append(lambda x, l=16: _apply_quantization(x, l))  # View 3: 16-level (4-bit) quantization

        # Generate additional diverse views dynamically if num_views > 3
        for i in range(len(transforms), self.num_views):
            if i % 2 == 1:
                # Stochastic noise view with scaled variance
                std_multiplier = 1.0 + 0.3 * (i // 2)
                cur_std = self.noise_std * std_multiplier
                transforms.append(lambda x, s=cur_std: _apply_noise(x, s))
            else:
                # Squeezed view with different quantization levels (8, 32, etc.)
                levels = 8 if (i // 2) % 2 == 1 else 32
                transforms.append(lambda x, l=levels: _apply_quantization(x, l))

        return transforms[: self.num_views]

    def apply(
        self,
        model: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> HardeningResult:
        start_time = time.time()
        try:
            transforms = self.custom_transforms or self._build_default_transforms()
            wrapped_model = PredictionEnsembleModelWrapper(
                base_model=model,
                transforms=transforms,
                voting=self.voting,
                weights=self.weights,
            )

            extra_meta: Dict[str, Any] = {
                "defense_family": self.defense_family,
                "num_views": len(transforms),
            }

            # Evaluate consensus metadata under torch.no_grad() to eliminate autograd graph overhead
            if inputs is not None:
                with torch.no_grad():
                    meta_res = wrapped_model.predict_with_metadata(inputs)
                    if "mean_consensus" in meta_res:
                        extra_meta["mean_consensus_agreement"] = meta_res["mean_consensus"]

            hardened_inputs = inputs.clone() if inputs is not None else None
            exec_time = time.time() - start_time

            meta = HardeningMetadata(
                defense_name=self.name,
                defense_type=self.defense_type,
                parameters={
                    "voting": self.voting,
                    "num_views": len(transforms),
                    "noise_std": self.noise_std,
                    "clip_min": self.clip_min,
                    "clip_max": self.clip_max,
                },
                execution_time_seconds=exec_time,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                extra_metadata=extra_meta,
            )
            return HardeningResult(
                hardened_model=wrapped_model,
                metadata=meta,
                hardened_inputs=hardened_inputs,
                success=True,
                recommendations=[
                    f"Applied Prediction Ensemble ({len(transforms)} views, voting='{self.voting}')."
                ],
            )
        except Exception as e:
            if isinstance(e, HardeningConfigurationError):
                raise
            raise DefenseExecutionError(f"PredictionEnsembleDefense failed: {str(e)}") from e


__all__ = [
    "PredictionEnsembleModelWrapper",
    "PredictionEnsembleDefense",
]
