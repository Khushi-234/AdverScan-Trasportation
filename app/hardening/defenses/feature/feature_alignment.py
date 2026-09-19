"""
Feature Alignment / Consistency Defense for Module 7 (Hardening).

Post-training, deployment-time defense that measures and enforces representation
consistency between inputs and reference smoothed/projected variants without retraining.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.hardening.defenses.base import BaseDefense
from app.hardening.hardening_result import HardeningMetadata, HardeningResult
from app.hardening.exceptions import DefenseExecutionError, HardeningConfigurationError


class FeatureAlignmentModelWrapper(nn.Module):
    """
    nn.Module wrapper evaluating input and reference probe consistency at deployment time.
    Blends representations to favor consistent, noise-invariant predictions without model retraining.
    """

    def __init__(
        self,
        base_model: nn.Module,
        noise_std: float = 0.05,
        alignment_weight: float = 0.5,
        metric: str = "cosine",
        clip_min: Optional[float] = 0.0,
        clip_max: Optional[float] = 1.0,
        consistency_threshold: float = 0.7,
        num_probes: int = 1,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.noise_std = float(noise_std)
        self.alignment_weight = float(alignment_weight)
        self.metric = metric
        self.clip_min = float(clip_min) if clip_min is not None else None
        self.clip_max = float(clip_max) if clip_max is not None else None
        self.consistency_threshold = float(consistency_threshold)
        self.num_probes = max(1, int(num_probes))

    def _generate_probe(self, x: torch.Tensor) -> torch.Tensor:
        """Generate Gaussian noise probe with optional pixel bounds clamping."""
        probe = x + torch.randn_like(x) * self.noise_std
        if self.clip_min is not None or self.clip_max is not None:
            c_min = float(self.clip_min) if self.clip_min is not None else float("-inf")
            c_max = float(self.clip_max) if self.clip_max is not None else float("inf")
            probe = torch.clamp(probe, min=c_min, max=c_max)
        return probe

    def _run_probes(self, x: torch.Tensor) -> torch.Tensor:
        """Run probe evaluation, supporting multi-probe Monte Carlo averaging to mitigate stochastic variance."""
        if self.num_probes > 1:
            probe_logits_list = []
            for _ in range(self.num_probes):
                p = self._generate_probe(x)
                out = self.base_model(p)
                if isinstance(out, tuple):
                    out = out[0]
                probe_logits_list.append(out)
            return torch.stack(probe_logits_list, dim=0).mean(dim=0)
        else:
            probe = self._generate_probe(x)
            out = self.base_model(probe)
            if isinstance(out, tuple):
                out = out[0]
            return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute aligned prediction combining original and stabilized reference evaluations.
        Ensures consistent evaluation mode between forward pass and probe passes.
        """
        # If wrapper is not in training mode, ensure base model runs consistently in eval mode
        # preventing stochastic discrepancy caused by BatchNorm or Dropout layers
        if not self.training and self.base_model.training:
            self.base_model.eval()

        logits_orig = self.base_model(x)
        if isinstance(logits_orig, tuple):
            logits_orig = logits_orig[0]

        logits_probe = self._run_probes(x)

        # Convert to probability space for alignment
        if logits_orig.ndim > 1 and logits_orig.size(1) > 1:
            p_orig = F.softmax(logits_orig, dim=-1)
            p_probe = F.softmax(logits_probe, dim=-1)

            # Measure consistency (cosine similarity)
            sim = F.cosine_similarity(p_orig, p_probe, dim=-1).unsqueeze(-1)  # (B, 1)

            # Predictable blending formula at the extremes:
            # alpha = alignment_weight + (1 - alignment_weight) * sim
            # When alignment_weight = 1.0 -> alpha = 1.0 (100% original prediction as documented)
            # When alignment_weight = 0.0 -> alpha = sim (full dependence on measured consistency)
            # When predictions are consistent (sim ~ 1.0) -> alpha = 1.0 (trust original)
            # When predictions diverge (sim is low) -> alpha drops towards alignment_weight
            alpha = torch.clamp(self.alignment_weight + (1.0 - self.alignment_weight) * sim, 0.0, 1.0)
            aligned_p = alpha * p_orig + (1.0 - alpha) * p_probe
            return torch.log(aligned_p + 1e-10)
        else:
            p_orig = torch.sigmoid(logits_orig)
            p_probe = torch.sigmoid(logits_probe)
            sim = 1.0 - torch.abs(p_orig - p_probe)
            if sim.ndim < p_orig.ndim:
                sim = sim.unsqueeze(-1)
            alpha = torch.clamp(self.alignment_weight + (1.0 - self.alignment_weight) * sim, 0.0, 1.0)
            aligned_p = alpha * p_orig + (1.0 - alpha) * p_probe
            return torch.logit(aligned_p.clamp(1e-6, 1.0 - 1e-6))

    def evaluate_consistency(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Evaluate feature and predictive consistency metrics for input batch.
        Saves and restores original model training state without permanent side effects.
        """
        was_training = self.base_model.training
        self.base_model.eval()
        try:
            with torch.no_grad():
                logits_orig = self.base_model(x)
                if isinstance(logits_orig, tuple):
                    logits_orig = logits_orig[0]

                logits_probe = self._run_probes(x)

                if logits_orig.ndim > 1 and logits_orig.size(1) > 1:
                    p_orig = F.softmax(logits_orig, dim=-1)
                    p_probe = F.softmax(logits_probe, dim=-1)
                    sim = F.cosine_similarity(p_orig, p_probe, dim=-1)
                else:
                    p_orig = torch.sigmoid(logits_orig)
                    p_probe = torch.sigmoid(logits_probe)
                    sim = 1.0 - torch.abs(p_orig - p_probe).squeeze(-1)

                return {
                    "consistency_scores": sim,
                    "mean_consistency": float(sim.mean().item()) if len(sim) > 0 else 0.0,
                    "is_consistent": (sim >= self.consistency_threshold),
                }
        finally:
            if was_training:
                self.base_model.train()


class FeatureAlignmentDefense(BaseDefense):
    """
    Feature Alignment / Feature Consistency Defense.

    Deployment-time post-training defense that measures prediction stability between the
    input and reference transformations and aligns output representations.
    Does NOT retrain the model.
    """

    supported_domains: List[str] = ["image", "tabular"]
    defense_family: str = "feature"

    def __init__(
        self,
        noise_std: float = 0.05,
        alignment_weight: float = 0.5,
        metric: str = "cosine",
        clip_min: Optional[float] = 0.0,
        clip_max: Optional[float] = 1.0,
        consistency_threshold: float = 0.7,
        num_probes: int = 1,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize FeatureAlignmentDefense.

        Args:
            noise_std: Probe noise standard deviation for generating reference feature views.
            alignment_weight: Base mixing weight between raw and reference predictions in [0.0, 1.0].
            metric: Consistency similarity metric ('cosine').
            clip_min: Optional minimum valid input value to prevent out-of-range probe artifacts.
            clip_max: Optional maximum valid input value to prevent out-of-range probe artifacts.
            consistency_threshold: Threshold above which a sample is classified as consistent.
            num_probes: Number of Monte Carlo probes to average for variance reduction.
            config: Optional configuration dictionary.
        """
        cfg = config or {}
        super().__init__(
            name="feature_alignment",
            defense_type="feature",
            config=cfg,
            supported_domains=self.supported_domains,
            defense_family=self.defense_family,
        )
        self.noise_std = float(cfg.get("noise_std", noise_std))
        self.alignment_weight = float(cfg.get("alignment_weight", alignment_weight))
        self.metric = str(cfg.get("metric", metric)).lower().strip()

        raw_cmin = cfg.get("clip_min", clip_min)
        self.clip_min = float(raw_cmin) if raw_cmin is not None else None
        raw_cmax = cfg.get("clip_max", clip_max)
        self.clip_max = float(raw_cmax) if raw_cmax is not None else None

        self.consistency_threshold = float(cfg.get("consistency_threshold", consistency_threshold))
        self.num_probes = max(1, int(cfg.get("num_probes", num_probes)))

        if self.noise_std < 0.0:
            raise HardeningConfigurationError(f"noise_std must be non-negative, got {self.noise_std}")
        if not (0.0 <= self.alignment_weight <= 1.0):
            raise HardeningConfigurationError(f"alignment_weight must be in [0.0, 1.0], got {self.alignment_weight}")
        if not (0.0 <= self.consistency_threshold <= 1.0):
            raise HardeningConfigurationError(
                f"consistency_threshold must be in [0.0, 1.0], got {self.consistency_threshold}"
            )
        if self.clip_min is not None and self.clip_max is not None and self.clip_min >= self.clip_max:
            raise HardeningConfigurationError(
                f"clip_min ({self.clip_min}) must be strictly less than clip_max ({self.clip_max})"
            )

    def apply(
        self,
        model: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> HardeningResult:
        start_time = time.time()
        try:
            wrapped_model = FeatureAlignmentModelWrapper(
                base_model=model,
                noise_std=self.noise_std,
                alignment_weight=self.alignment_weight,
                metric=self.metric,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
                consistency_threshold=self.consistency_threshold,
                num_probes=self.num_probes,
            )

            extra_meta: Dict[str, Any] = {"defense_family": self.defense_family}

            if inputs is not None:
                consistency_info = wrapped_model.evaluate_consistency(inputs)
                extra_meta.update({
                    "mean_consistency_score": consistency_info["mean_consistency"],
                    "consistency_scores": consistency_info["consistency_scores"].cpu().tolist(),
                })

            hardened_inputs = inputs.clone() if inputs is not None else None
            exec_time = time.time() - start_time

            meta = HardeningMetadata(
                defense_name=self.name,
                defense_type=self.defense_type,
                parameters={
                    "noise_std": self.noise_std,
                    "alignment_weight": self.alignment_weight,
                    "metric": self.metric,
                    "clip_min": self.clip_min,
                    "clip_max": self.clip_max,
                    "consistency_threshold": self.consistency_threshold,
                    "num_probes": self.num_probes,
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
                    f"Applied Feature Alignment wrapper (weight={self.alignment_weight:.2f}, noise_std={self.noise_std:.2f}, num_probes={self.num_probes})."
                ],
            )
        except Exception as e:
            if isinstance(e, HardeningConfigurationError):
                raise
            raise DefenseExecutionError(f"FeatureAlignmentDefense failed: {str(e)}") from e
