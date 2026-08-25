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


class ComparisonEngine:
    """
    Engine for comparing before vs after hardening evaluation results.
    Does not perform attacks or hardening.
    """

    def compare_attack(
        self,
        before_assessment: Union[AssessmentResult, Dict[str, Any]],
        after_assessment: Union[AssessmentResult, Dict[str, Any]],
        before_scoring: Union[ScoringResult, Dict[str, Any]],
        after_scoring: Union[ScoringResult, Dict[str, Any]],
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
        b_assess_dict = before_assessment.to_dict() if hasattr(before_assessment, "to_dict") else dict(before_assessment)
        a_assess_dict = after_assessment.to_dict() if hasattr(after_assessment, "to_dict") else dict(after_assessment)
        b_score_dict = before_scoring.to_dict() if hasattr(before_scoring, "to_dict") else dict(before_scoring)
        a_score_dict = after_scoring.to_dict() if hasattr(after_scoring, "to_dict") else dict(after_scoring)

        attack_name = a_assess_dict.get("attack_name", b_assess_dict.get("attack_name", "unknown_attack"))

        # Extract metric values
        b_asr = b_assess_dict.get("attack_success_rate")
        a_asr = a_assess_dict.get("attack_success_rate")
        delta_asr: Optional[float] = None
        if a_asr is not None and b_asr is not None:
            delta_asr = round(float(a_asr) - float(b_asr), 4)

        b_acc_drop = float(b_assess_dict.get("accuracy_drop", 0.0))
        a_acc_drop = float(a_assess_dict.get("accuracy_drop", 0.0))
        delta_acc_drop = round(a_acc_drop - b_acc_drop, 4)

        b_f1_drop = float(b_assess_dict.get("f1_drop", 0.0))
        a_f1_drop = float(a_assess_dict.get("f1_drop", 0.0))
        delta_f1_drop = round(a_f1_drop - b_f1_drop, 4)

        b_conf_drop = float(b_assess_dict.get("confidence_drop", 0.0))
        a_conf_drop = float(a_assess_dict.get("confidence_drop", 0.0))
        delta_conf_drop = round(a_conf_drop - b_conf_drop, 4)

        b_degrad = float(b_assess_dict.get("model_degradation", 0.0))
        a_degrad = float(a_assess_dict.get("model_degradation", 0.0))
        delta_model_degrad = round(a_degrad - b_degrad, 4)

        b_vuln_score = float(b_score_dict.get("vulnerability_score", 0.0))
        a_vuln_score = float(a_score_dict.get("vulnerability_score", 0.0))
        delta_vuln_score = round(a_vuln_score - b_vuln_score, 4)

        b_clean_acc = float(b_assess_dict.get("clean_accuracy", 0.0))
        a_clean_acc = float(a_assess_dict.get("clean_accuracy", 0.0))
        delta_clean_acc = round(a_clean_acc - b_clean_acc, 4)

        b_adv_acc = float(b_assess_dict.get("adversarial_accuracy", 0.0))
        a_adv_acc = float(a_assess_dict.get("adversarial_accuracy", 0.0))
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

        # Robustness Improvement criteria:
        # A decrease in vulnerability score, ASR, accuracy drop, or degradation indicates improvement.
        improved_score = delta_vuln_score < 0
        improved_asr = delta_asr < 0 if delta_asr is not None else False
        improved_acc_drop = delta_acc_drop < 0
        improved_adv_acc = delta_adv_acc > 0

        is_improved = (improved_score or improved_asr or improved_acc_drop or improved_adv_acc)

        # Construct summary notes
        summary_notes: List[str] = []
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

        if risk_changed:
            summary_notes.append(f"Risk level changed from {b_risk} to {a_risk}.")

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
        )

    def compare_pipeline(
        self,
        before_vulnerability_analysis: Dict[str, Dict[str, Any]],
        after_vulnerability_analysis: Dict[str, Dict[str, Any]],
    ) -> Dict[str, ComparisonResult]:
        """
        Compare pipeline outputs across multiple attacks.

        Args:
            before_vulnerability_analysis: Dict mapping attack_name -> {"assessment": ..., "scoring": ...}
            after_vulnerability_analysis: Dict mapping attack_name -> {"assessment": ..., "scoring": ...}

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
            )
            comparisons[atk_name] = comparison

        return comparisons


def compare_results(
    before_vulnerability_analysis: Dict[str, Dict[str, Any]],
    after_vulnerability_analysis: Dict[str, Dict[str, Any]],
) -> Dict[str, ComparisonResult]:
    """
    Convenience function to compare before vs after vulnerability analysis.

    Args:
        before_vulnerability_analysis: Dict mapping attack_name -> {"assessment": ..., "scoring": ...}
        after_vulnerability_analysis: Dict mapping attack_name -> {"assessment": ..., "scoring": ...}

    Returns:
        Dict[str, ComparisonResult] mapping attack_name -> ComparisonResult.
    """
    engine = ComparisonEngine()
    return engine.compare_pipeline(before_vulnerability_analysis, after_vulnerability_analysis)
