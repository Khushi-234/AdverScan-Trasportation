"""
Adversarial Example Detection Defense implementation for Module 7 (Hardening).

Provides a modular detector for identifying adversarial inputs using distinct detection
heuristics (sensitivity under perturbation, prediction margin, entropy, consistency).
Does not conflate low-confidence predictions with adversarial status.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.hardening.defenses.base import BaseDefense
from app.hardening.hardening_result import HardeningMetadata, HardeningResult
from app.hardening.exceptions import DefenseExecutionError, HardeningConfigurationError


class AdversarialDetectorModelWrapper(nn.Module):
    """
    nn.Module wrapper that executes forward inference and enforces adversarial detection.
    """

    def __init__(
        self,
        model: nn.Module,
        detector: Any,
        enforce_defense: bool = True,
        rejection_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.detector = detector
        self.enforce_defense = enforce_defense
        self.rejection_value = float(rejection_value)
        self.last_detection_result: Optional[Dict[str, Any]] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass actively routing through adversarial detection.

        If enforce_defense is True, inputs flagged as adversarial have their
        prediction logits suppressed (set to uniform/zero logits), preventing
        generic attack and evaluation loops (e.g., PGD, C&W, accuracy harnesses)
        from silently bypassing the defense by construction.
        """
        if not self.enforce_defense:
            return self.model(x)

        res = self.predict_with_detection(x)
        self.last_detection_result = res
        logits = res["logits"].clone()
        is_adversarial = res["is_adversarial"]

        if is_adversarial.any():
            logits[is_adversarial] = self.rejection_value

        return logits

    def predict_with_detection(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Evaluate inputs and return predictions alongside adversarial detection metrics.
        """
        return self.detector.detect_with_predictions(self.model, x)


class AdversarialDetectionDefense(BaseDefense):
    """
    Adversarial Example Detection Defense.

    Identifies suspicious / adversarial inputs using modular detection strategies:
    - 'sensitivity': Measures prediction divergence under small isotropic perturbations.
    - 'margin': Measures narrow logit gap between top-1 and top-2 classes.
    - 'entropy': Measures normalized predictive entropy across classes.
    """

    supported_domains: List[str] = ["image", "nlp", "tabular", "audio"]
    defense_family: str = "detection"

    def __init__(
        self,
        threshold: float = 0.5,
        noise_std: float = 0.05,
        num_samples: int = 5,
        method: str = "sensitivity",
        enforce_defense: bool = True,
        rejection_value: float = 0.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize AdversarialDetectionDefense.

        Args:
            threshold: Anomaly score threshold [0, 1] above which an input is classified as adversarial.
            noise_std: Standard deviation of probe noise for sensitivity analysis.
            num_samples: Number of perturbation probes evaluated per sample (must be >= 1).
            method: Detection heuristic ('sensitivity', 'margin', 'entropy').
            enforce_defense: Whether wrapper forward() suppresses logits for detected adversarial inputs.
            rejection_value: Logit value assigned to detected samples when enforce_defense is True.
            config: Optional configuration dictionary.
        """
        cfg = config or {}
        super().__init__(
            name="adversarial_detection",
            defense_type="detection",
            config=cfg,
            supported_domains=self.supported_domains,
            defense_family=self.defense_family,
        )
        self.threshold = float(cfg.get("threshold", threshold))
        self.noise_std = float(cfg.get("noise_std", noise_std))
        self.num_samples = int(cfg.get("num_samples", num_samples))
        self.method = str(cfg.get("method", method)).lower().strip()
        self.enforce_defense = bool(cfg.get("enforce_defense", enforce_defense))
        self.rejection_value = float(cfg.get("rejection_value", rejection_value))

        if not (0.0 <= self.threshold <= 1.0):
            raise HardeningConfigurationError(f"threshold must be in [0.0, 1.0], got {self.threshold}")
        if self.num_samples < 1:
            raise HardeningConfigurationError(f"num_samples must be at least 1, got {self.num_samples}")
        if self.noise_std < 0.0:
            raise HardeningConfigurationError(f"noise_std must be non-negative, got {self.noise_std}")
        if self.method not in ("sensitivity", "margin", "entropy"):
            raise HardeningConfigurationError(
                f"Unsupported detection method '{self.method}'. Supported: ['sensitivity', 'margin', 'entropy']"
            )

    def detect(self, model: nn.Module, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate adversarial anomaly scores and detection flags.

        Args:
            model: PyTorch model module.
            inputs: Input tensor (B, ...).

        Returns:
            Tuple of (is_adversarial: BoolTensor, anomaly_scores: FloatTensor).
        """
        res = self.detect_with_predictions(model, inputs)
        return res["is_adversarial"], res["scores"]

    def detect_with_predictions(self, model: nn.Module, inputs: torch.Tensor) -> Dict[str, Any]:
        """
        Evaluate inputs and return predictions alongside adversarial anomaly scores.
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
                p = torch.sigmoid(clean_logits).squeeze(-1)
                clean_probs = torch.stack([1.0 - p, p], dim=-1)
                clean_preds = (p >= 0.5).long()

            if self.method == "entropy":
                # High prediction entropy can indicate perturbed / out-of-distribution sample
                if clean_logits.ndim > 1 and clean_logits.size(1) > 1:
                    log_probs = F.log_softmax(clean_logits, dim=1)
                    entropy = -torch.sum(clean_probs * log_probs, dim=1)
                    max_entropy = torch.log(torch.tensor(float(clean_logits.size(1)), device=device))
                    scores = entropy / (max_entropy + 1e-8)
                else:
                    p = clean_probs[:, 1].clamp(1e-7, 1 - 1e-7)
                    entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
                    scores = entropy / torch.log(torch.tensor(2.0, device=device))

            elif self.method == "margin":
                # Adversarial inputs frequently have razor-thin margin between top-1 and top-2
                if clean_probs.size(1) > 2:
                    top2_vals, _ = torch.topk(clean_probs, k=2, dim=1)
                    margin = top2_vals[:, 0] - top2_vals[:, 1]
                    scores = 1.0 - margin  # Higher score = smaller margin = more suspicious
                else:
                    # Binary classification (shape B, 2): margin between class 1 and class 0 probabilities
                    # |p1 - p0| is mathematically equivalent to 2 * |p - 0.5|
                    margin = torch.abs(clean_probs[:, 1] - clean_probs[:, 0])
                    scores = 1.0 - margin

            elif self.method == "sensitivity":
                # Evaluate prediction instability under small random noise
                divergence_accum = torch.zeros(batch_size, device=device)
                for _ in range(self.num_samples):
                    noise = torch.randn_like(inputs) * self.noise_std
                    noisy_logits = model(inputs + noise)
                    if isinstance(noisy_logits, tuple):
                        noisy_logits = noisy_logits[0]

                    if noisy_logits.ndim > 1 and noisy_logits.size(1) > 1:
                        noisy_probs = F.softmax(noisy_logits, dim=1)
                        # Total Variation / L1 distance between probability vectors
                        div = torch.sum(torch.abs(noisy_probs - clean_probs), dim=1) / 2.0
                    else:
                        noisy_p = torch.sigmoid(noisy_logits).squeeze(-1)
                        div = torch.abs(noisy_p - clean_probs[:, 1])

                    divergence_accum += div

                scores = (divergence_accum / self.num_samples).clamp(0.0, 1.0)
            else:
                scores = torch.zeros(batch_size, device=device)

            scores = scores.clamp(0.0, 1.0)
            is_adversarial = scores >= self.threshold

        return {
            "logits": clean_logits,
            "probabilities": clean_probs,
            "predictions": clean_preds,
            "scores": scores,
            "is_adversarial": is_adversarial,
        }

    def apply(
        self,
        model: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> HardeningResult:
        start_time = time.time()
        try:
            wrapped_model = AdversarialDetectorModelWrapper(
                model=model,
                detector=self,
                enforce_defense=self.enforce_defense,
                rejection_value=self.rejection_value,
            )
            metadata_dict: Dict[str, Any] = {
                "threshold": self.threshold,
                "method": self.method,
                "noise_std": self.noise_std,
                "num_samples": self.num_samples,
                "enforce_defense": self.enforce_defense,
                "rejection_value": self.rejection_value,
                "defense_family": self.defense_family,
            }

            if inputs is None:
                exec_time = time.time() - start_time
                meta = HardeningMetadata(
                    defense_name=self.name,
                    defense_type=self.defense_type,
                    parameters={"threshold": self.threshold, "method": self.method},
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
                    recommendations=["Registered AdversarialDetection model wrapper."],
                )

            res = self.detect_with_predictions(model, inputs)
            is_adversarial = res["is_adversarial"]
            scores = res["scores"]
            detected_count = int(is_adversarial.sum().item())
            total_samples = int(inputs.size(0))
            detection_rate = float(detected_count / total_samples) if total_samples > 0 else 0.0

            # Filter inputs and labels synchronously to preserve dataset index alignment
            is_accepted = ~is_adversarial
            hardened_inputs = inputs[is_accepted] if is_accepted.any() else inputs[:0]
            hardened_labels = None
            if labels is not None:
                hardened_labels = labels[is_accepted] if is_accepted.any() else labels[:0]

            exec_time = time.time() - start_time
            metadata_dict.update(
                {
                    "total_samples": total_samples,
                    "adversarial_detected_count": detected_count,
                    "clean_accepted_count": total_samples - detected_count,
                    "detection_rate": detection_rate,
                    "mean_anomaly_score": float(scores.mean().item()) if total_samples > 0 else 0.0,
                    "detection_mask": is_adversarial.cpu().tolist(),
                    "is_accepted_mask": is_accepted.cpu().tolist(),
                    "detection_scores": scores.cpu().tolist(),
                }
            )
            if hardened_labels is not None:
                metadata_dict["has_filtered_labels"] = True

            meta = HardeningMetadata(
                defense_name=self.name,
                defense_type=self.defense_type,
                parameters={
                    "threshold": self.threshold,
                    "method": self.method,
                    "noise_std": self.noise_std,
                    "num_samples": self.num_samples,
                    "enforce_defense": self.enforce_defense,
                    "rejection_value": self.rejection_value,
                },
                execution_time_seconds=exec_time,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                extra_metadata=metadata_dict,
            )

            return HardeningResult(
                hardened_model=wrapped_model,
                metadata=meta,
                hardened_inputs=hardened_inputs,
                hardened_labels=hardened_labels,
                success=True,
                recommendations=[
                    f"Adversarial Detection ({self.method}): {detected_count}/{total_samples} samples flagged "
                    f"above anomaly threshold {self.threshold:.2f}."
                ],
            )
        except Exception as e:
            if isinstance(e, HardeningConfigurationError):
                raise
            raise DefenseExecutionError(f"AdversarialDetectionDefense failed: {str(e)}") from e
