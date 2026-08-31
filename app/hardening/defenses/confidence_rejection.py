"""
Confidence-Based Rejection Defense implementation for Module 7 (Hardening).

Rejects input samples whose maximum prediction confidence falls below a configured threshold.
"""

import time
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.hardening.defenses.base import BaseDefense
from app.hardening.hardening_result import HardeningMetadata, HardeningResult


class ConfidenceRejectionDefense(BaseDefense):
    """
    Confidence-Based Rejection Defense.

    Flags or rejects input samples when the model's highest prediction confidence
    falls below a user-defined threshold.
    """

    def __init__(
        self,
        threshold: float = 0.6,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize ConfidenceRejectionDefense.

        Args:
            threshold: Confidence threshold below which inputs are rejected (0.0 to 1.0).
            config: Optional configuration dictionary.
        """
        cfg = config or {}
        super().__init__(
            name="confidence_rejection",
            defense_type="rejection",
            config=cfg,
        )
        self.threshold = float(cfg.get("threshold", threshold))

    def apply(
        self,
        model: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> HardeningResult:
        """
        Apply confidence-based rejection to input tensor using model predictions.

        Args:
            model: Target PyTorch neural network module.
            inputs: Tensor of input samples.
            labels: Optional ground truth target labels.
            **kwargs: Additional execution context.

        Returns:
            HardeningResult with rejection mask and metadata.
        """
        start_time = time.time()
        model.eval()

        metadata_dict: Dict[str, Any] = {"threshold": self.threshold}

        if inputs is None:
            exec_time = time.time() - start_time
            meta = HardeningMetadata(
                defense_name=self.name,
                defense_type=self.defense_type,
                parameters={"threshold": self.threshold},
                execution_time_seconds=exec_time,
                extra_metadata=metadata_dict,
            )
            return HardeningResult(
                hardened_model=model,
                metadata=meta,
                hardened_inputs=None,
                success=True,
                recommendations=["Inputs tensor not provided; defense registered model without batch evaluation."],
            )

        with torch.no_grad():
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            if outputs.ndim > 1 and outputs.size(1) > 1:
                probs = F.softmax(outputs, dim=1)
                max_probs, preds = torch.max(probs, dim=1)
            else:
                probs = torch.sigmoid(outputs).squeeze(-1)
                max_probs = torch.max(probs, 1 - probs)

            is_accepted = max_probs >= self.threshold
            is_rejected = ~is_accepted
            rejected_count = int(is_rejected.sum().item())
            total_samples = int(inputs.size(0))

        exec_time = time.time() - start_time

        metadata_dict.update(
            {
                "total_samples": total_samples,
                "rejected_count": rejected_count,
                "accepted_count": total_samples - rejected_count,
                "rejection_rate": rejected_count / total_samples if total_samples > 0 else 0.0,
                "mean_confidence": float(max_probs.mean().item()) if total_samples > 0 else 0.0,
                "rejection_mask": is_rejected.cpu().tolist(),
                "confidences": max_probs.cpu().tolist(),
            }
        )

        meta = HardeningMetadata(
            defense_name=self.name,
            defense_type=self.defense_type,
            parameters={"threshold": self.threshold},
            execution_time_seconds=exec_time,
            extra_metadata=metadata_dict,
        )

        recommendations = [
            f"Evaluated {total_samples} samples; {rejected_count} samples rejected below confidence threshold {self.threshold:.2f}."
        ]

        return HardeningResult(
            hardened_model=model,
            metadata=meta,
            hardened_inputs=inputs[is_accepted] if is_accepted.any() else inputs[:0],
            success=True,
            recommendations=recommendations,
        )
