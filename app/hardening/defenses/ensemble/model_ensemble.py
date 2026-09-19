"""
Model Ensemble Defense implementation for Module 7 (Hardening).

Executes multiple compatible models on the same input tensor and aggregates
their predictive outputs (averaging, soft voting, hard voting, median) without retraining.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.hardening.defenses.base import BaseDefense
from app.hardening.hardening_result import HardeningMetadata, HardeningResult
from app.hardening.exceptions import DefenseExecutionError, HardeningConfigurationError


class EnsembleModelWrapper(nn.Module):
    """
    nn.Module wrapper coordinating parallel forward passes across multiple models
    and aggregating output predictions via configurable strategies.
    """

    def __init__(
        self,
        models: Sequence[nn.Module],
        aggregation: str = "mean",
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        model_list = list(models)
        if not model_list:
            raise HardeningConfigurationError("Ensemble must contain at least one model.")

        self.models = nn.ModuleList(model_list)
        self.aggregation = str(aggregation).lower().strip()

        # Weight validation
        if weights is not None:
            weights_list = [float(w) for w in weights]
            if len(weights_list) != len(self.models):
                raise HardeningConfigurationError(
                    f"Number of weights ({len(weights_list)}) must match number of models ({len(self.models)})."
                )
            if any(w < 0.0 for w in weights_list):
                raise HardeningConfigurationError("Ensemble weights must be non-negative.")
            total_w = sum(weights_list)
            if total_w <= 0.0:
                raise HardeningConfigurationError("Sum of ensemble weights must be strictly positive.")
            self.weights = [w / total_w for w in weights_list]
        else:
            self.weights = [1.0 / len(self.models)] * len(self.models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute forward pass across all ensemble member models and aggregate predictions.

        Returns:
            Aggregated output tensor matching expected logit / log-probability shape.
        """
        outputs: List[torch.Tensor] = []
        for model in self.models:
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            outputs.append(out)

        if not outputs:
            raise HardeningConfigurationError("Ensemble contains no valid models.")

        if len(outputs) == 1:
            return outputs[0]

        # Case 1: Median aggregation (applicable across any output shape)
        if self.aggregation == "median":
            stacked_logits = torch.stack(outputs, dim=0)
            return torch.median(stacked_logits, dim=0).values

        # Case 2: Multi-class classification outputs (B, C) with C > 1
        if outputs[0].ndim > 1 and outputs[0].size(1) > 1:
            probs = [F.softmax(o, dim=-1) for o in outputs]
            num_classes = outputs[0].size(-1)

            if self.aggregation in ("mean", "average", "soft_voting"):
                weighted_sum = torch.zeros_like(probs[0])
                for p, w in zip(probs, self.weights):
                    weighted_sum += p * w
                return torch.log(weighted_sum.clamp_min(1e-10))

            elif self.aggregation == "hard_voting":
                # Compute vote proportions for each class
                preds = [torch.argmax(p, dim=-1) for p in probs]  # List of (B,)
                vote_shares = torch.zeros_like(probs[0])
                for pred, w in zip(preds, self.weights):
                    one_hot_vote = F.one_hot(pred, num_classes=num_classes).float()
                    vote_shares += one_hot_vote * w
                # Return log vote shares so argmax yields the majority vote and softmax recovers vote proportions
                return torch.log(vote_shares.clamp_min(1e-10))

        # Case 3: Binary classification outputs (B, 1) or (B,)
        if outputs[0].ndim == 1 or (outputs[0].ndim == 2 and outputs[0].size(1) == 1):
            probs_pos = [torch.sigmoid(o) for o in outputs]

            if self.aggregation in ("mean", "average", "soft_voting"):
                weighted_pos = torch.zeros_like(probs_pos[0])
                for p, w in zip(probs_pos, self.weights):
                    weighted_pos += p * w
                weighted_pos = torch.clamp(weighted_pos, 1e-7, 1.0 - 1e-7)
                return torch.logit(weighted_pos)

            elif self.aggregation == "hard_voting":
                # Vote positive if predicted probability >= 0.5
                vote_pos = torch.zeros_like(probs_pos[0])
                for p, w in zip(probs_pos, self.weights):
                    vote_pos += (p >= 0.5).float() * w
                vote_pos = torch.clamp(vote_pos, 1e-7, 1.0 - 1e-7)
                return torch.logit(vote_pos)

        # Fallback: weighted average of raw outputs directly
        weighted_out = torch.zeros_like(outputs[0])
        for o, w in zip(outputs, self.weights):
            weighted_out += o * w
        return weighted_out


class ModelEnsembleDefense(BaseDefense):
    """
    Model Ensemble Defense.

    Combines multiple diverse, compatible models without retraining to neutralize
    transferable and single-model adversarial perturbations.
    """

    supported_domains: List[str] = ["image", "nlp", "tabular", "audio"]
    defense_family: str = "ensemble"

    def __init__(
        self,
        models: Optional[Sequence[nn.Module]] = None,
        aggregation: str = "mean",
        weights: Optional[Sequence[float]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize ModelEnsembleDefense.

        Args:
            models: Optional sequence of pre-trained models to ensemble.
            aggregation: Aggregation strategy ('mean', 'soft_voting', 'hard_voting', 'median').
            weights: Optional weight per model.
            config: Optional configuration dictionary.
        """
        cfg = config or {}
        super().__init__(
            name="model_ensemble",
            defense_type="ensemble",
            config=cfg,
            supported_domains=self.supported_domains,
            defense_family=self.defense_family,
        )
        self.models = list(models) if models is not None else []
        self.aggregation = str(cfg.get("aggregation", aggregation)).lower().strip()
        raw_weights = cfg.get("weights", weights)
        self.weights = [float(w) for w in raw_weights] if raw_weights is not None else None

        if self.aggregation not in ("mean", "average", "soft_voting", "hard_voting", "median"):
            raise HardeningConfigurationError(
                f"Unsupported aggregation '{self.aggregation}'. Supported: ['mean', 'soft_voting', 'hard_voting', 'median']"
            )

        if self.weights is not None:
            if any(w < 0.0 for w in self.weights):
                raise HardeningConfigurationError("Ensemble weights must be non-negative.")
            if sum(self.weights) <= 0.0:
                raise HardeningConfigurationError("Sum of ensemble weights must be strictly positive.")

    def apply(
        self,
        model: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> HardeningResult:
        start_time = time.time()
        try:
            ensemble_members: List[nn.Module] = [model]
            if self.models:
                for m in self.models:
                    if m is not model:
                        ensemble_members.append(m)

            weights_to_use = self.weights
            if weights_to_use is not None and len(weights_to_use) != len(ensemble_members):
                raise HardeningConfigurationError(
                    f"Number of weights ({len(weights_to_use)}) must match number of ensemble models ({len(ensemble_members)})."
                )

            wrapped_model = EnsembleModelWrapper(
                models=ensemble_members,
                aggregation=self.aggregation,
                weights=weights_to_use,
            )

            hardened_inputs = inputs.clone() if inputs is not None else None
            exec_time = time.time() - start_time

            meta = HardeningMetadata(
                defense_name=self.name,
                defense_type=self.defense_type,
                parameters={
                    "aggregation": self.aggregation,
                    "num_models": len(ensemble_members),
                    "weights": wrapped_model.weights,
                },
                execution_time_seconds=exec_time,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                extra_metadata={"defense_family": self.defense_family},
            )
            return HardeningResult(
                hardened_model=wrapped_model,
                metadata=meta,
                hardened_inputs=hardened_inputs,
                success=True,
                recommendations=[
                    f"Combined {len(ensemble_members)} models into ensemble wrapper (aggregation='{self.aggregation}')."
                ],
            )
        except Exception as e:
            if isinstance(e, HardeningConfigurationError):
                raise
            raise DefenseExecutionError(f"ModelEnsembleDefense failed: {str(e)}") from e


__all__ = [
    "EnsembleModelWrapper",
    "ModelEnsembleDefense",
]
