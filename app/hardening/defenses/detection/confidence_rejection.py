"""
Confidence-Based Rejection Defense implementation for Module 7 (Hardening).

Evaluates model prediction confidence against a configurable threshold, abstaining/rejecting
predictions when confidence is insufficient. Explicitly tracks coverage and rejection rate
so rejection cannot be misconstrued as improved model accuracy.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.hardening.defenses.base import BaseDefense
from app.hardening.hardening_result import HardeningMetadata, HardeningResult
from app.hardening.exceptions import DefenseExecutionError, HardeningConfigurationError


class ConfidenceRejectionModelWrapper(nn.Module):
    """
    nn.Module wrapper that executes forward inference and enforces confidence rejection.
    """

    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.6,
        enforce_defense: bool = True,
        rejection_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.threshold = float(threshold)
        self.enforce_defense = enforce_defense
        self.rejection_value = float(rejection_value)
        self.last_rejection_result: Optional[Dict[str, Any]] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass actively routing through confidence rejection.

        If enforce_defense is True, inputs with maximum prediction confidence
        below threshold have their output logits suppressed (set to uniform/zero logits),
        preventing generic evaluation loops from silently bypassing rejection.
        """
        if not self.enforce_defense:
            return self.model(x)

        res = self.predict_with_rejection(x)
        self.last_rejection_result = res
        outputs = res["logits"].clone()
        is_rejected = res["is_rejected"]

        if is_rejected.any():
            outputs[is_rejected] = self.rejection_value

        return outputs

    def predict_with_rejection(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Evaluate inputs and return predictions alongside abstention / rejection flags.

        Args:
            x: Input tensor.

        Returns:
            Dict containing:
                - logits: Raw model output
                - probabilities: Softmax/Sigmoid probabilities
                - predictions: Argmax predicted classes
                - confidences: Max confidence per sample
                - is_rejected: BoolTensor indicating which samples are rejected
                - coverage: Proportion of accepted samples (1 - rejection_rate)
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            if outputs.ndim > 1 and outputs.size(1) > 1:
                probs = F.softmax(outputs, dim=1)
                max_probs, preds = torch.max(probs, dim=1)
            else:
                p = torch.sigmoid(outputs).squeeze(-1)
                probs = torch.stack([1.0 - p, p], dim=-1)
                max_probs = torch.max(p, 1.0 - p)
                preds = (p >= 0.5).long()

            is_rejected = max_probs < self.threshold
            coverage = float((~is_rejected).float().mean().item()) if len(is_rejected) > 0 else 0.0

            return {
                "logits": outputs,
                "probabilities": probs,
                "predictions": preds,
                "confidences": max_probs,
                "is_rejected": is_rejected,
                "coverage": coverage,
            }


class ConfidenceRejectionDefense(BaseDefense):
    """
    Confidence-Based Rejection Defense.

    Rejects or flags input samples whose peak prediction confidence is below a
    user-configured threshold. Tracks coverage and rejection metrics rigorously.
    """

    supported_domains: List[str] = ["image", "nlp", "tabular", "audio"]
    defense_family: str = "detection"

    def __init__(
        self,
        threshold: float = 0.6,
        enforce_defense: bool = True,
        rejection_value: float = 0.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize ConfidenceRejectionDefense.

        Args:
            threshold: Confidence threshold below which inputs are rejected (0.0 to 1.0).
            enforce_defense: Whether wrapper forward() suppresses logits for rejected inputs.
            rejection_value: Logit value assigned to rejected samples when enforce_defense is True.
            config: Optional configuration dictionary.
        """
        cfg = config or {}
        super().__init__(
            name="confidence_rejection",
            defense_type="detection",
            config=cfg,
            supported_domains=self.supported_domains,
            defense_family=self.defense_family,
        )
        self.threshold = float(cfg.get("threshold", threshold))
        self.enforce_defense = bool(cfg.get("enforce_defense", enforce_defense))
        self.rejection_value = float(cfg.get("rejection_value", rejection_value))
        if not (0.0 <= self.threshold <= 1.0):
            raise HardeningConfigurationError(f"threshold must be in [0.0, 1.0], got {self.threshold}")

    def apply(
        self,
        model: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> HardeningResult:
        start_time = time.time()
        try:
            model.eval()
            wrapped_model = ConfidenceRejectionModelWrapper(
                model=model,
                threshold=self.threshold,
                enforce_defense=self.enforce_defense,
                rejection_value=self.rejection_value,
            )
            metadata_dict: Dict[str, Any] = {
                "threshold": self.threshold,
                "enforce_defense": self.enforce_defense,
                "rejection_value": self.rejection_value,
                "defense_family": self.defense_family,
            }

            if inputs is None:
                exec_time = time.time() - start_time
                meta = HardeningMetadata(
                    defense_name=self.name,
                    defense_type=self.defense_type,
                    parameters={"threshold": self.threshold},
                    execution_time_seconds=exec_time,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    extra_metadata=metadata_dict,
                )
                return HardeningResult(
                    hardened_model=wrapped_model,
                    metadata=meta,
                    hardened_inputs=None,
                    hardened_labels=None,
                    success=True,
                    recommendations=["Inputs tensor not provided; registered wrapped model with confidence rejection."],
                )

            with torch.no_grad():
                res = wrapped_model.predict_with_rejection(inputs)
                max_probs = res["confidences"]
                is_rejected = res["is_rejected"]
                is_accepted = ~is_rejected
                rejected_count = int(is_rejected.sum().item())
                total_samples = int(inputs.size(0))
                rejection_rate = float(rejected_count / total_samples) if total_samples > 0 else 0.0
                coverage = float(1.0 - rejection_rate)

            # Filter inputs and labels synchronously to preserve dataset index alignment
            hardened_inputs = inputs[is_accepted] if is_accepted.any() else inputs[:0]
            hardened_labels = None
            if labels is not None:
                hardened_labels = labels[is_accepted] if is_accepted.any() else labels[:0]

            exec_time = time.time() - start_time
            metadata_dict.update(
                {
                    "total_samples": total_samples,
                    "rejected_count": rejected_count,
                    "accepted_count": total_samples - rejected_count,
                    "rejection_rate": rejection_rate,
                    "coverage": coverage,
                    "mean_confidence": float(max_probs.mean().item()) if total_samples > 0 else 0.0,
                    "rejection_mask": is_rejected.cpu().tolist(),
                    "is_accepted_mask": is_accepted.cpu().tolist(),
                    "confidences": max_probs.cpu().tolist(),
                }
            )
            if hardened_labels is not None:
                metadata_dict["has_filtered_labels"] = True

            meta = HardeningMetadata(
                defense_name=self.name,
                defense_type=self.defense_type,
                parameters={
                    "threshold": self.threshold,
                    "enforce_defense": self.enforce_defense,
                    "rejection_value": self.rejection_value,
                },
                execution_time_seconds=exec_time,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                extra_metadata=metadata_dict,
            )

            recommendations = [
                f"Evaluated {total_samples} samples; rejected {rejected_count} ({rejection_rate:.1%}) below threshold {self.threshold:.2f}.",
                f"Effective deployment coverage: {coverage:.1%}. Note: rejection rate must be considered alongside accuracy.",
            ]

            return HardeningResult(
                hardened_model=wrapped_model,
                metadata=meta,
                hardened_inputs=hardened_inputs,
                hardened_labels=hardened_labels,
                success=True,
                recommendations=recommendations,
            )
        except Exception as e:
            if isinstance(e, HardeningConfigurationError):
                raise
            raise DefenseExecutionError(f"ConfidenceRejectionDefense failed: {str(e)}") from e
