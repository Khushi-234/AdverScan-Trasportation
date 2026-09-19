"""
Configurable Acceptance Thresholds for Module 8 (Re-test) and Evaluation in AdverScan.

Enables context-specific, tunable thresholds for evaluating whether a defended model
provides practically meaningful improvement in vulnerability score, attack success rate (ASR),
clean accuracy preservation, and latency limits during before/after re-testing.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RetestThresholds:
    """
    Configurable acceptance thresholds for M8 re-test verification and iterative defense selection.

    Attributes:
        min_vuln_score_improvement: Minimum meaningful reduction in vulnerability score (default: 5.0 points).
        min_asr_reduction: Minimum meaningful reduction in Attack Success Rate (default: 0.10, i.e., 10 percentage points).
        max_clean_accuracy_drop: Maximum acceptable drop in clean accuracy (default: 0.02, i.e., 2 percentage points).
        max_latency_overhead_ms: Maximum acceptable inference latency overhead in milliseconds (default: 30.0 ms).
        respect_latency_sensitive: Whether to strictly enforce latency constraint when context is latency sensitive.
        severe_clean_drop_threshold: Clean accuracy degradation threshold considered harmful (default: 0.10, i.e., 10 percentage points).
        extra_thresholds: Additional custom threshold parameters for experiment tuning.
    """

    min_vuln_score_improvement: float = 5.0
    min_asr_reduction: float = 0.10
    max_clean_accuracy_drop: float = 0.02
    max_latency_overhead_ms: float = 30.0
    respect_latency_sensitive: bool = True
    severe_clean_drop_threshold: float = 0.10
    extra_thresholds: Dict[str, Any] = field(default_factory=dict)

    def evaluate(
        self,
        before_metrics: Dict[str, Any],
        after_metrics: Dict[str, Any],
        latency_sensitive: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate before vs after metrics against configured acceptance criteria.

        Args:
            before_metrics: Dictionary containing baseline metrics (vuln_score, asr, clean_accuracy, latency).
            after_metrics: Dictionary containing defended metrics.
            latency_sensitive: Flag indicating if operational context requires strict latency adherence.

        Returns:
            Dict containing:
                - status: "accepted" | "insufficient" | "harmful"
                - is_improved: bool (any measurable reduction in vuln score or ASR)
                - improvement_sufficient: bool (satisfies all configured acceptance thresholds)
                - vulnerability_score_improvement: Optional[float]
                - asr_reduction: Optional[float]
                - clean_accuracy_drop: Optional[float]
                - clean_accuracy_drop_pct_points: Optional[float]
                - latency_overhead_ms: Optional[float]
                - reason: str
        """
        def _extract(data: Dict[str, Any], *keys: str) -> Optional[float]:
            for k in keys:
                if k in data and data[k] is not None:
                    try:
                        return float(data[k])
                    except (ValueError, TypeError):
                        continue
            return None

        # Extract vulnerability score (0 - 100 scale)
        vuln_before = _extract(before_metrics, "vulnerability_score", "vuln_score")
        vuln_after = _extract(after_metrics, "vulnerability_score", "vuln_score")
        vuln_improvement: Optional[float] = None
        if vuln_before is not None and vuln_after is not None:
            vuln_improvement = round(vuln_before - vuln_after, 2)

        # Extract ASR (0.0 - 1.0 fraction)
        asr_before = _extract(before_metrics, "attack_success_rate", "asr")
        asr_after = _extract(after_metrics, "attack_success_rate", "asr")
        asr_reduction: Optional[float] = None
        if asr_before is not None and asr_after is not None:
            asr_reduction = round(asr_before - asr_after, 4)

        # Extract Clean Accuracy (0.0 - 1.0 fraction)
        clean_before = _extract(before_metrics, "clean_accuracy", "accuracy")
        clean_after = _extract(after_metrics, "clean_accuracy", "accuracy")
        clean_drop: Optional[float] = None
        clean_drop_pts: Optional[float] = None
        if clean_before is not None and clean_after is not None:
            clean_drop = round(clean_before - clean_after, 4)
            clean_drop_pts = round(clean_drop * 100.0, 2)

        # Extract Latency (ms)
        def _extract_lat(data: Dict[str, Any]) -> Optional[float]:
            val = _extract(data, "latency_ms", "latency")
            if val is not None:
                return val
            val_sec = _extract(data, "execution_time_seconds", "runtime_seconds", "latency_sec")
            if val_sec is not None:
                return round(val_sec * 1000.0, 4)
            return None

        lat_before = _extract_lat(before_metrics)
        lat_after = _extract_lat(after_metrics)
        lat_overhead_ms: Optional[float] = None
        if lat_after is not None and lat_before is not None:
            lat_overhead_ms = round(lat_after - lat_before, 4)
        elif lat_after is not None:
            lat_overhead_ms = 0.0

        # Check 1: Any measurable improvement (is_improved)
        has_vuln_improvement = vuln_improvement is not None and vuln_improvement > 0.0
        has_asr_reduction = asr_reduction is not None and asr_reduction > 0.0
        is_improved = bool(has_vuln_improvement or has_asr_reduction)

        # Check 2: Harmful check
        is_harmful = False
        harmful_reasons = []
        if clean_drop is not None and clean_drop > self.severe_clean_drop_threshold:
            is_harmful = True
            harmful_reasons.append(
                f"Severe clean accuracy drop: {clean_drop_pts:.1f} percentage points exceeds harmful threshold {self.severe_clean_drop_threshold * 100:.1f}%"
            )
        if vuln_improvement is not None and vuln_improvement < -5.0:
            is_harmful = True
            harmful_reasons.append(
                f"Vulnerability worsened by {abs(vuln_improvement):.1f} points"
            )
        if asr_reduction is not None and asr_reduction < -0.10:
            is_harmful = True
            harmful_reasons.append(
                f"ASR worsened by {abs(asr_reduction) * 100:.1f} percentage points"
            )

        # Check 3: Acceptance thresholds (improvement_sufficient)
        rejection_reasons = []

        robustness_passed = False
        if vuln_improvement is not None and vuln_improvement >= self.min_vuln_score_improvement:
            robustness_passed = True
        elif asr_reduction is not None and asr_reduction >= self.min_asr_reduction:
            robustness_passed = True
        else:
            if vuln_improvement is not None and asr_reduction is not None:
                rejection_reasons.append(
                    f"Insufficient robustness improvement: Vuln score improvement {vuln_improvement:.1f} < {self.min_vuln_score_improvement:.1f} points and ASR reduction {asr_reduction * 100:.1f}% < {self.min_asr_reduction * 100:.1f}%"
                )
            elif vuln_improvement is not None:
                rejection_reasons.append(
                    f"Insufficient vulnerability score reduction: {vuln_improvement:.1f} < {self.min_vuln_score_improvement:.1f} points"
                )
            elif asr_reduction is not None:
                rejection_reasons.append(
                    f"Insufficient ASR reduction: {asr_reduction * 100:.1f}% < {self.min_asr_reduction * 100:.1f}%"
                )
            else:
                rejection_reasons.append("No vulnerability score or ASR metrics available to verify improvement")

        clean_passed = True
        if clean_drop is not None and clean_drop > self.max_clean_accuracy_drop:
            clean_passed = False
            rejection_reasons.append(
                f"Clean accuracy drop {clean_drop_pts:.1f}% exceeds maximum acceptable drop {self.max_clean_accuracy_drop * 100:.1f}%"
            )

        latency_passed = True
        strict_latency = latency_sensitive and self.respect_latency_sensitive
        if strict_latency and lat_overhead_ms is not None and lat_overhead_ms > self.max_latency_overhead_ms:
            latency_passed = False
            rejection_reasons.append(
                f"Latency overhead {lat_overhead_ms:.1f} ms exceeds maximum acceptable limit {self.max_latency_overhead_ms:.1f} ms under latency-sensitive constraint"
            )

        improvement_sufficient = robustness_passed and clean_passed and latency_passed and not is_harmful

        if improvement_sufficient:
            status = "accepted"
            reason = "Defense satisfies all acceptance thresholds with meaningful improvement."
        elif is_harmful:
            status = "harmful"
            reason = "; ".join(harmful_reasons)
        else:
            status = "insufficient"
            reason = "; ".join(rejection_reasons) if rejection_reasons else "Improvement does not satisfy configured thresholds."

        return {
            "status": status,
            "is_improved": is_improved,
            "improvement_sufficient": improvement_sufficient,
            "vulnerability_score_before": vuln_before,
            "vulnerability_score_after": vuln_after,
            "vulnerability_score_improvement": vuln_improvement,
            "asr_before": asr_before,
            "asr_after": asr_after,
            "asr_reduction": asr_reduction,
            "clean_accuracy_before": clean_before,
            "clean_accuracy_after": clean_after,
            "clean_accuracy_drop": clean_drop,
            "clean_accuracy_drop_pct_points": clean_drop_pts,
            "defended_latency_ms": lat_after,
            "baseline_latency_ms": lat_before,
            "latency_overhead_ms": lat_overhead_ms,
            "reason": reason,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert thresholds to dictionary."""
        return asdict(self)


# Canonical alias
Thresholds = RetestThresholds
