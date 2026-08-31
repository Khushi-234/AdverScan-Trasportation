"""
Adversarial Example Detection Defense implementation for Module 7 (Hardening).

Provides a model-agnostic, attack-agnostic detector interface for identifying potential
adversarial examples via output variance, perturbation stability analysis, and confidence margin metrics.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.hardening.defenses.base import BaseDefense
from app.hardening.hardening_result import HardeningMetadata, HardeningResult


class AdversarialDetectionDefense(BaseDefense):
    """
    Adversarial Example Detection Defense.

    Provides generic detection capability to identify adversarial inputs
    by evaluating prediction sensitivity under small perturbations and output distribution metrics.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        noise_std: float = 0.05,
        num_samples: int = 5,
        method: str = "sensitivity",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize AdversarialDetectionDefense.

        Args:
            threshold: Anomaly/detection score threshold above which input is flagged as adversarial (0.0 to 1.0).
            noise_std: Standard deviation of Gaussian noise used for sensitivity probing.
            num_samples: Number of noisy probes evaluated per input.
            method: Detection heuristic ('sensitivity', 'margin', 'entropy').
            config: Optional configuration dictionary.
        """
        cfg = config or {}
        super().__init__(
            name="adversarial_detection",
            defense_type="detection",
            config=cfg,
        )
        self.threshold = float(cfg.get("threshold", threshold))
        self.noise_std = float(cfg.get("noise_std", noise_std))
        self.num_samples = int(cfg.get("num_samples", num_samples))
        self.method = str(cfg.get("method", method)).lower().strip()

    def detect(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generic detection method calculating adversarial anomaly scores and detection flags.

        Args:
            model: Target PyTorch neural network.
            inputs: Input tensor (batch_size, ...).

        Returns:
            Tuple of:
                - is_adversarial (torch.BoolTensor): Boolean flag per sample indicating adversarial status.
                - scores (torch.FloatTensor): Calculated anomaly/confidence detection score per sample [0, 1].
        """
        model.eval()
        device = inputs.device
        batch_size = inputs.size(0)

        with torch.no_grad():
            clean_logits = model(inputs)
            if isinstance(clean_logits, tuple):
                clean_logits = clean_logits[0]

            if clean_logits.ndim > 1 and clean_logits.size(1) > 1:
                clean_probs = F.softmax(clean_logits, dim=1)
                clean_preds = torch.argmax(clean_probs, dim=1)
            else:
                clean_probs = torch.sigmoid(clean_logits).squeeze(-1)
                clean_preds = (clean_probs >= 0.5).long()

            if self.method == "entropy":
                # High prediction entropy / uniform logits indicate perturbed or out-of-distribution sample
                if clean_logits.ndim > 1 and clean_logits.size(1) > 1:
                    log_probs = F.log_softmax(clean_logits, dim=1)
                    entropy = -torch.sum(clean_probs * log_probs, dim=1)
                    max_entropy = torch.log(torch.tensor(float(clean_logits.size(1)), device=device))
                    scores = entropy / max_entropy
                else:
                    p = clean_probs.clamp(1e-7, 1 - 1e-7)
                    entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
                    scores = entropy / torch.log(torch.tensor(2.0, device=device))

            elif self.method == "margin":
                # Small prediction margin between top-1 and top-2 logits indicates high vulnerability/adversarial state
                if clean_logits.ndim > 1 and clean_logits.size(1) > 1:
                    top_vals, _ = torch.topk(clean_probs, k=2, dim=1)
                    margin = top_vals[:, 0] - top_vals[:, 1]
                    scores = 1.0 - margin
                else:
                    scores = 1.0 - (2.0 * (clean_probs - 0.5).abs())

            else:
                # Default: Prediction sensitivity under input noise (adversarial examples show prediction instability)
                disagreements = torch.zeros(batch_size, device=device)
                prob_shifts = torch.zeros(batch_size, device=device)

                for _ in range(self.num_samples):
                    noisy_inputs = inputs + torch.randn_like(inputs) * self.noise_std
                    noisy_logits = model(noisy_inputs)
                    if isinstance(noisy_logits, tuple):
                        noisy_logits = noisy_logits[0]

                    if noisy_logits.ndim > 1 and noisy_logits.size(1) > 1:
                        noisy_probs = F.softmax(noisy_logits, dim=1)
                        noisy_preds = torch.argmax(noisy_probs, dim=1)
                        disagreements += (noisy_preds != clean_preds).float()
                        prob_shifts += torch.abs(noisy_probs - clean_probs).max(dim=1).values
                    else:
                        noisy_probs = torch.sigmoid(noisy_logits).squeeze(-1)
                        noisy_preds = (noisy_probs >= 0.5).long()
                        disagreements += (noisy_preds != clean_preds).float()
                        prob_shifts += torch.abs(noisy_probs - clean_probs)

                instability_rate = disagreements / float(self.num_samples)
                mean_shift = prob_shifts / float(self.num_samples)
                scores = 0.6 * instability_rate + 0.4 * mean_shift.clamp(0.0, 1.0)

            is_adversarial = scores >= self.threshold

        return is_adversarial, scores

    def apply(
        self,
        model: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> HardeningResult:
        """
        Apply adversarial example detector to inputs and return standardized HardeningResult.

        Args:
            model: Target PyTorch neural network module.
            inputs: Tensor of input samples.
            labels: Optional ground truth target labels.
            **kwargs: Additional runtime arguments.

        Returns:
            HardeningResult with detection status, detection score/confidence, threshold, and metadata.
        """
        start_time = time.time()
        metadata_dict: Dict[str, Any] = {
            "threshold": self.threshold,
            "noise_std": self.noise_std,
            "num_samples": self.num_samples,
            "method": self.method,
        }

        if inputs is None:
            exec_time = time.time() - start_time
            meta = HardeningMetadata(
                defense_name=self.name,
                defense_type=self.defense_type,
                parameters={
                    "threshold": self.threshold,
                    "noise_std": self.noise_std,
                    "num_samples": self.num_samples,
                    "method": self.method,
                },
                execution_time_seconds=exec_time,
                extra_metadata=metadata_dict,
            )
            return HardeningResult(
                hardened_model=model,
                metadata=meta,
                hardened_inputs=None,
                success=True,
                recommendations=["Inputs tensor not provided; detector registered model without evaluation."],
            )

        is_adv_mask, scores = self.detect(model, inputs)
        total_samples = int(inputs.size(0))
        detected_count = int(is_adv_mask.sum().item())
        detection_rate = float(detected_count / total_samples) if total_samples > 0 else 0.0
        avg_score = float(scores.mean().item()) if total_samples > 0 else 0.0

        metadata_dict.update(
            {
                "total_samples": total_samples,
                "adversarial_detected_count": detected_count,
                "clean_detected_count": total_samples - detected_count,
                "detection_rate": detection_rate,
                "average_detection_score": avg_score,
                "detection_mask": is_adv_mask.cpu().tolist(),
                "detection_scores": scores.cpu().tolist(),
            }
        )

        exec_time = time.time() - start_time
        meta = HardeningMetadata(
            defense_name=self.name,
            defense_type=self.defense_type,
            parameters={
                "threshold": self.threshold,
                "noise_std": self.noise_std,
                "num_samples": self.num_samples,
                "method": self.method,
            },
            execution_time_seconds=exec_time,
            extra_metadata=metadata_dict,
        )

        clean_inputs = inputs[~is_adv_mask] if (~is_adv_mask).any() else inputs[:0]

        recommendations = [
            f"Adversarial Detection ({self.method}): Scanned {total_samples} samples; detected {detected_count} adversarial inputs "
            f"({detection_rate * 100:.1f}%) with score threshold {self.threshold:.2f} (avg score: {avg_score:.3f})."
        ]

        return HardeningResult(
            hardened_model=model,
            metadata=meta,
            hardened_inputs=clean_inputs,
            success=True,
            recommendations=recommendations,
        )
