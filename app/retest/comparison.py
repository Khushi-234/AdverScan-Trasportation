"""
Comparison module for Module 8 (Re-Test & Comparison) in AdverScan.

Compares before-hardening and after-hardening vulnerability metrics and scores,
calculating deltas (Δ = After - Before) and assessing overall robustness improvement.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from app.vulnerability_analysis.assessment_result import AssessmentResult
from app.vulnerability_analysis.scoring_result import ScoringResult
from app.retest.retest_result import ComparisonResult
from app.retest.threshold import RetestThresholds


class ComparisonEngine:
    """
    Engine for comparing before vs after hardening evaluation results.
    Does not perform attacks or hardening.
    """

    def __init__(self, thresholds: Optional[RetestThresholds] = None) -> None:
        """Initialize ComparisonEngine with optional acceptance thresholds."""
        self.thresholds = thresholds or RetestThresholds()

    def compare_attack(
        self,
        before_assessment: Union[AssessmentResult, Dict[str, Any]],
        after_assessment: Union[AssessmentResult, Dict[str, Any]],
        before_scoring: Union[ScoringResult, Dict[str, Any]],
        after_scoring: Union[ScoringResult, Dict[str, Any]],
        thresholds: Optional[RetestThresholds] = None,
        latency_sensitive: bool = False,
    ) -> ComparisonResult:
        """
        Compare before and after results for a single attack.

        Args:
            before_assessment: AssessmentResult or dict before hardening.
            after_assessment: AssessmentResult or dict after hardening.
            before_scoring: ScoringResult or dict before hardening.
            after_scoring: ScoringResult or dict after hardening.

        Returns:
            ComparisonResult: Detailed delta and comparative summary DTO.
        """
        def _to_dict(item: Any) -> Dict[str, Any]:
            if hasattr(item, "to_dict") and callable(getattr(item, "to_dict")):
                return item.to_dict()
            if isinstance(item, dict):
                return dict(item)
            return {}

        b_assess_dict = _to_dict(before_assessment)
        a_assess_dict = _to_dict(after_assessment)
        b_score_dict = _to_dict(before_scoring)
        a_score_dict = _to_dict(after_scoring)

        attack_name = a_assess_dict.get("attack_name", b_assess_dict.get("attack_name", "unknown_attack"))

        # Extract metric values
        b_asr_raw = b_assess_dict.get("attack_success_rate")
        a_asr_raw = a_assess_dict.get("attack_success_rate")
        b_asr: Optional[float] = float(b_asr_raw) if b_asr_raw is not None else None
        a_asr: Optional[float] = float(a_asr_raw) if a_asr_raw is not None else None

        delta_asr: Optional[float] = None
        asr_reduction: Optional[float] = None
        if a_asr is not None and b_asr is not None:
            delta_asr = round(float(a_asr) - float(b_asr), 4)
            asr_reduction = round(float(b_asr) - float(a_asr), 4)

        def _safe_float(val: Any, default: Optional[float] = 0.0) -> Optional[float]:
            if val is None:
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        b_acc_drop = _safe_float(b_assess_dict.get("accuracy_drop")) or 0.0
        a_acc_drop = _safe_float(a_assess_dict.get("accuracy_drop")) or 0.0
        delta_acc_drop = round(a_acc_drop - b_acc_drop, 4)

        b_f1_drop = _safe_float(b_assess_dict.get("f1_drop")) or 0.0
        a_f1_drop = _safe_float(a_assess_dict.get("f1_drop")) or 0.0
        delta_f1_drop = round(a_f1_drop - b_f1_drop, 4)

        b_conf_drop = _safe_float(b_assess_dict.get("confidence_drop")) or 0.0
        a_conf_drop = _safe_float(a_assess_dict.get("confidence_drop")) or 0.0
        delta_conf_drop = round(a_conf_drop - b_conf_drop, 4)

        b_degrad = _safe_float(b_assess_dict.get("model_degradation")) or 0.0
        a_degrad = _safe_float(a_assess_dict.get("model_degradation")) or 0.0
        delta_model_degrad = round(a_degrad - b_degrad, 4)

        b_vuln_score = _safe_float(b_score_dict.get("vulnerability_score")) or 0.0
        a_vuln_score = _safe_float(a_score_dict.get("vulnerability_score")) or 0.0
        delta_vuln_score = round(a_vuln_score - b_vuln_score, 4)

        b_clean_raw = _safe_float(b_assess_dict.get("clean_accuracy"), default=None)
        a_clean_raw = _safe_float(a_assess_dict.get("clean_accuracy"), default=None)
        b_clean_acc = b_clean_raw if b_clean_raw is not None else 0.0
        a_clean_acc = a_clean_raw if a_clean_raw is not None else 0.0
        delta_clean_acc = round(a_clean_acc - b_clean_acc, 4)
        clean_acc_drop = round(b_clean_acc - a_clean_acc, 4) if (b_clean_raw is not None and a_clean_raw is not None) else None

        b_adv_acc = _safe_float(b_assess_dict.get("adversarial_accuracy")) or 0.0
        a_adv_acc = _safe_float(a_assess_dict.get("adversarial_accuracy")) or 0.0
        delta_adv_acc = round(a_adv_acc - b_adv_acc, 4)

        # Perturbation deltas if present
        b_pert = b_assess_dict.get("perturbation", {})
        a_pert = a_assess_dict.get("perturbation", {})
        delta_pert: Dict[str, Any] = {}
        if isinstance(b_pert, dict) and isinstance(a_pert, dict):
            for k in set(b_pert.keys()).union(a_pert.keys()):
                val_b = b_pert.get(k)
                val_a = a_pert.get(k)
                if isinstance(val_b, (int, float)) and isinstance(val_a, (int, float)):
                    delta_pert[k] = round(float(val_a) - float(val_b), 4)

        # Risk level comparison
        b_risk = str(b_score_dict.get("risk_level", "UNKNOWN"))
        a_risk = str(a_score_dict.get("risk_level", "UNKNOWN"))
        risk_changed = (b_risk != a_risk)

        # Check explicit failure or unsupported status
        a_status = a_assess_dict.get("status")
        summary_notes: List[str] = []

        active_thresh = thresholds or self.thresholds or RetestThresholds()
        b_metrics = {**b_assess_dict, **b_score_dict}
        a_metrics = {**a_assess_dict, **a_score_dict}
        thresh_eval = active_thresh.evaluate(
            before_metrics=b_metrics,
            after_metrics=a_metrics,
            latency_sensitive=latency_sensitive,
        )

        if a_status == "unsupported":
            status = "unsupported"
            defense_eff = "unsupported"
            is_improved = False
            summary_notes.append(f"Attack '{attack_name}' is unsupported on the defended model.")
        elif a_status in ("failed", "attack_failed"):
            status = "attack_failed"
            defense_eff = "inconclusive"
            is_improved = False
            summary_notes.append(
                f"Attack '{attack_name}' execution failed during re-test: {a_assess_dict.get('error', 'Execution error')}"
            )
        elif b_asr is None or a_asr is None:
            status = "inconclusive"
            defense_eff = "inconclusive"
            is_improved = False
            summary_notes.append(f"Attack '{attack_name}' re-test is inconclusive: ASR is undefined.")
        elif b_asr == 0.0:
            summary_notes.append(
                f"Baseline ASR for '{attack_name}' was 0.0 (zero-success attack); robustness improvement cannot be verified."
            )
            defense_eff = "inconclusive"
            status = "inconclusive"
            is_improved = False
        else:
            thresh_status = thresh_eval.get("status")
            if thresh_status == "accepted":
                status = "successful"
                is_improved = True
                defense_eff = "high" if (asr_reduction is not None and asr_reduction >= 0.50) else "moderate"
            elif thresh_status == "harmful":
                status = "failed_defense"
                is_improved = False
                defense_eff = "harmful"
            elif thresh_status == "insufficient":
                status = "failed_defense"
                is_improved = bool(thresh_eval.get("is_improved", False))
                defense_eff = "low" if is_improved else "ineffective"
            else:
                status = "inconclusive"
                is_improved = False
                defense_eff = "inconclusive"

        # Summary notes for verified metrics
        if delta_vuln_score < 0:
            summary_notes.append(f"Vulnerability score decreased by {abs(delta_vuln_score):.2f} points.")
        elif delta_vuln_score > 0:
            summary_notes.append(f"Vulnerability score increased by {delta_vuln_score:.2f} points.")
        else:
            summary_notes.append("Vulnerability score remained unchanged.")

        if delta_asr is not None:
            if delta_asr < 0:
                summary_notes.append(f"Attack Success Rate (ASR) reduced by {abs(delta_asr)*100:.2f}%.")
            elif delta_asr > 0:
                summary_notes.append(f"Attack Success Rate (ASR) increased by {delta_asr*100:.2f}%.")

        if delta_adv_acc > 0:
            summary_notes.append(f"Adversarial accuracy improved by {delta_adv_acc*100:.2f}%.")
        elif delta_adv_acc < 0:
            summary_notes.append(f"Adversarial accuracy degraded by {abs(delta_adv_acc)*100:.2f}%.")

        if clean_acc_drop is not None and clean_acc_drop > 0:
            summary_notes.append(f"Clean accuracy dropped by {clean_acc_drop*100:.2f}%.")

        if risk_changed:
            summary_notes.append(f"Risk level changed from {b_risk} to {a_risk}.")

        if thresh_eval.get("reason"):
            summary_notes.append(f"Threshold verdict: {str(thresh_eval.get('status')).upper()} ({thresh_eval.get('reason')})")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return ComparisonResult(
            attack_name=attack_name,
            before_assessment=b_assess_dict,
            after_assessment=a_assess_dict,
            before_scoring=b_score_dict,
            after_scoring=a_score_dict,
            delta_attack_success_rate=delta_asr,
            delta_accuracy_drop=delta_acc_drop,
            delta_f1_drop=delta_f1_drop,
            delta_confidence_drop=delta_conf_drop,
            delta_model_degradation=delta_model_degrad,
            delta_vulnerability_score=delta_vuln_score,
            delta_clean_accuracy=delta_clean_acc,
            delta_adversarial_accuracy=delta_adv_acc,
            delta_perturbation=delta_pert,
            before_risk_level=b_risk,
            after_risk_level=a_risk,
            risk_level_changed=risk_changed,
            is_improved=is_improved,
            summary_notes=summary_notes,
            timestamp=timestamp,
            status=status,
            defense_effectiveness=defense_eff,
            asr_reduction=asr_reduction,
            clean_accuracy_drop=clean_acc_drop,
            threshold_evaluation=thresh_eval,
        )

    def compare_pipeline(
        self,
        before_vulnerability_analysis: Dict[str, Dict[str, Any]],
        after_vulnerability_analysis: Dict[str, Dict[str, Any]],
        thresholds: Optional[RetestThresholds] = None,
        latency_sensitive: bool = False,
    ) -> Dict[str, ComparisonResult]:
        """
        Compare pipeline outputs across multiple attacks.

        Args:
            before_vulnerability_analysis: Dict mapping attack_name -> {"assessment": ..., "scoring": ...}
            after_vulnerability_analysis: Dict mapping attack_name -> {"assessment": ..., "scoring": ...}
            thresholds: Optional RetestThresholds acceptance configuration.
            latency_sensitive: Whether operational context requires strict latency adherence.

        Returns:
            Dict[str, ComparisonResult]: Dict mapping attack_name -> ComparisonResult.
        """
        comparisons: Dict[str, ComparisonResult] = {}
        all_keys = set(before_vulnerability_analysis.keys()).union(after_vulnerability_analysis.keys())

        for atk_name in all_keys:
            b_info = before_vulnerability_analysis.get(atk_name, {})
            a_info = after_vulnerability_analysis.get(atk_name, {})

            b_assess = b_info.get("assessment", {})
            b_score = b_info.get("scoring", {})
            a_assess = a_info.get("assessment", {})
            a_score = a_info.get("scoring", {})

            comparison = self.compare_attack(
                before_assessment=b_assess,
                after_assessment=a_assess,
                before_scoring=b_score,
                after_scoring=a_score,
                thresholds=thresholds,
                latency_sensitive=latency_sensitive,
            )
            comparisons[atk_name] = comparison

        return comparisons


def compare_results(
    before_vulnerability_analysis: Dict[str, Dict[str, Any]],
    after_vulnerability_analysis: Dict[str, Dict[str, Any]],
    thresholds: Optional[RetestThresholds] = None,
    latency_sensitive: bool = False,
) -> Dict[str, ComparisonResult]:
    """
    Convenience function to compare before vs after vulnerability analysis.

    Args:
        before_vulnerability_analysis: Dict mapping attack_name -> {"assessment": ..., "scoring": ...}
        after_vulnerability_analysis: Dict mapping attack_name -> {"assessment": ..., "scoring": ...}
        thresholds: Optional RetestThresholds acceptance configuration.
        latency_sensitive: Whether operational context requires strict latency adherence.

    Returns:
        Dict[str, ComparisonResult] mapping attack_name -> ComparisonResult.
    """
    engine = ComparisonEngine(thresholds=thresholds)
    return engine.compare_pipeline(
        before_vulnerability_analysis,
        after_vulnerability_analysis,
        thresholds=thresholds,
        latency_sensitive=latency_sensitive,
    )
