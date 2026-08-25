"""
Input data contract DTO for Module 9 (Report Generator) in AdverScan.

Defines the structured data required to generate security reports by aggregating
outputs from previous modules (M1 - M8).
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class ReportData:
    """
    Input data contract for the AdverScan Report Generator.

    Aggregates:
    - M1: Model Information
    - M2: Baseline Performance
    - M3 / M2: Attack Results (FGSM, PGD, DeepFool, etc.)
    - M4 / M5: Vulnerability Metrics, Vulnerability Score, Risk Level
    - M6: XAI Findings
    - M7: Hardening Results
    - M8: Before vs After Comparison
    """

    model_info: Dict[str, Any] = field(default_factory=dict)
    baseline_performance: Dict[str, Any] = field(default_factory=dict)
    attack_results: Dict[str, Any] = field(default_factory=dict)
    vulnerability_metrics: Dict[str, Any] = field(default_factory=dict)
    vulnerability_score: Optional[float] = None
    risk_level: Optional[str] = None
    xai_findings: Dict[str, Any] = field(default_factory=dict)
    hardening_results: Dict[str, Any] = field(default_factory=dict)
    before_vs_after: Dict[str, Any] = field(default_factory=dict)
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ReportData object to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReportData":
        """
        Create a ReportData instance from a dictionary.

        Args:
            data: Dictionary containing report input fields.

        Returns:
            ReportData instance.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict for ReportData.from_dict, got {type(data).__name__}")

        # Extract risk level and score if nested inside vulnerability_analysis
        vuln_metrics = data.get("vulnerability_metrics") or data.get("vulnerability_analysis") or {}
        vuln_score = data.get("vulnerability_score")
        risk_lvl = data.get("risk_level")

        if (vuln_score is None or risk_lvl is None) and isinstance(vuln_metrics, dict):
            for _, val in vuln_metrics.items():
                if isinstance(val, dict):
                    scoring = val.get("scoring") or val.get("vulnerability_scoring") or {}
                    if isinstance(scoring, dict):
                        if vuln_score is None and "vulnerability_score" in scoring:
                            vuln_score = scoring["vulnerability_score"]
                        if risk_lvl is None and "risk_level" in scoring:
                            risk_lvl = scoring["risk_level"]

        return cls(
            model_info=data.get("model_info") or data.get("model_metadata") or {},
            baseline_performance=data.get("baseline_performance") or data.get("baseline_evaluation") or {},
            attack_results=data.get("attack_results") or {},
            vulnerability_metrics=vuln_metrics,
            vulnerability_score=vuln_score,
            risk_level=risk_lvl,
            xai_findings=data.get("xai_findings") or data.get("xai_results") or {},
            hardening_results=data.get("hardening_results") or {},
            before_vs_after=data.get("before_vs_after") or data.get("comparisons") or data.get("comparison_results") or {},
            extra_metadata=data.get("extra_metadata") or {},
        )

    @classmethod
    def from_orchestration_and_retest(
        cls,
        orchestration_result: Any,
        retest_result: Optional[Any] = None,
    ) -> "ReportData":
        """
        Convenience factory to build ReportData from OrchestrationResult and RetestResult DTOs or dicts.

        Args:
            orchestration_result: OrchestrationResult DTO or dict from M8 Orchestrator.
            retest_result: RetestResult DTO or dict from M8 Retest Engine (optional).

        Returns:
            ReportData instance.
        """
        orch_dict = (
            orchestration_result.to_dict()
            if hasattr(orchestration_result, "to_dict")
            else (orchestration_result if isinstance(orchestration_result, dict) else {})
        )

        retest_dict = (
            retest_result.to_dict()
            if hasattr(retest_result, "to_dict")
            else (retest_result if isinstance(retest_result, dict) else {})
        )

        # Merge attack evaluation info if available
        attack_res = orch_dict.get("attack_results", {})
        adv_evals = orch_dict.get("adversarial_evaluations", {})
        merged_attacks = {}
        for atk_name, atk_info in attack_res.items():
            atk_dict = dict(atk_info) if isinstance(atk_info, dict) else {}
            if atk_name in adv_evals:
                atk_dict["evaluation"] = adv_evals[atk_name]
            merged_attacks[atk_name] = atk_dict

        # Extract vulnerability score & risk level from orchestration result if available
        vuln_analysis = orch_dict.get("vulnerability_analysis", {})
        vuln_score = None
        risk_level = None
        if isinstance(vuln_analysis, dict):
            for _, val in vuln_analysis.items():
                if isinstance(val, dict) and "scoring" in val:
                    scoring = val["scoring"]
                    if isinstance(scoring, dict):
                        vuln_score = scoring.get("vulnerability_score", vuln_score)
                        risk_level = scoring.get("risk_level", risk_level)

        # Before vs After comparison from re-test
        before_vs_after = retest_dict.get("comparisons", {})

        return cls(
            model_info=orch_dict.get("model_metadata", {}),
            baseline_performance=orch_dict.get("baseline_evaluation", {}),
            attack_results=merged_attacks,
            vulnerability_metrics=vuln_analysis,
            vulnerability_score=vuln_score,
            risk_level=risk_level,
            xai_findings=orch_dict.get("xai_results", {}),
            hardening_results=orch_dict.get("hardening_results", {}),
            before_vs_after=before_vs_after,
            extra_metadata={
                "status": orch_dict.get("status"),
                "execution_mode": orch_dict.get("execution_mode"),
                "timestamp": orch_dict.get("timestamp"),
                "execution_time_seconds": orch_dict.get("execution_time_seconds"),
            },
        )
