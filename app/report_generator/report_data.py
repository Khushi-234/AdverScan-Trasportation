"""
ReportData — Structured input data contract for the AdverScan Report Generator (Module 9).

Aggregates all outputs from M1–M8 plus execution performance into a single,
validated container that feeds the ReportGenerator coordinator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .execution_summary import ExecutionSummary


@dataclass
class ReportData:
    """
    Complete input data contract for generating an AdverScan Security Report.

    Aggregates:
    - M1  : Model Information
    - M2  : Dataset / Evaluation Configuration + Baseline Performance
    - M3  : Adversarial Attack Results (FGSM, PGD, DeepFool, …)
    - M5  : Vulnerability Assessment (ASR, accuracy drop, F1 drop, confidence drop, …)
    - M6  : XAI Findings (feature attribution, attack impact, failure analysis)
    - M7  : Hardening (selected defense, applied defense, hardened model)
    - M8  : Re-Test Results + Before vs After Comparison
    - M9  : Execution Performance (from ResultTracker / ExecutionSummary)
    """

    # ── M1: Model Information ─────────────────────────────────────────────────
    model_info: Dict[str, Any] = field(default_factory=dict)

    # ── M2: Dataset Configuration + Baseline Performance ─────────────────────
    dataset_config: Dict[str, Any] = field(default_factory=dict)
    baseline_performance: Dict[str, Any] = field(default_factory=dict)

    # ── M3: Attack Results ────────────────────────────────────────────────────
    attack_results: Dict[str, Any] = field(default_factory=dict)

    # ── M5: Vulnerability Assessment ─────────────────────────────────────────
    vulnerability_metrics: Dict[str, Any] = field(default_factory=dict)
    vulnerability_score: Optional[float] = None
    risk_level: Optional[str] = None

    # ── M6: XAI Findings ─────────────────────────────────────────────────────
    xai_findings: Dict[str, Any] = field(default_factory=dict)

    # ── M7: Hardening ─────────────────────────────────────────────────────────
    hardening_results: Dict[str, Any] = field(default_factory=dict)

    # ── M8: Re-Test + Before vs After ────────────────────────────────────────
    retest_results: Dict[str, Any] = field(default_factory=dict)
    before_vs_after: Dict[str, Any] = field(default_factory=dict)

    # ── M9 / Execution Performance ────────────────────────────────────────────
    execution_summary: Optional[ExecutionSummary] = None

    # ── Extras ────────────────────────────────────────────────────────────────
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Convert ReportData to a plain dictionary."""
        d = asdict(self)
        # execution_summary is not a standard dataclass — serialize manually
        d["execution_summary"] = self.execution_summary.to_dict() if self.execution_summary else None
        return d

    # ── Factory Methods ───────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReportData":
        """
        Create a ReportData from a plain dictionary (e.g. from JSON).
        Handles both the new key schema and the legacy OrchestrationResult schema.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict for ReportData.from_dict, got {type(data).__name__}")

        # ── Vulnerability score / risk level extraction ────────────────────────
        vuln_metrics = (
            data.get("vulnerability_metrics")
            or data.get("vulnerability_analysis")
            or {}
        )
        vuln_score = data.get("vulnerability_score")
        risk_lvl = data.get("risk_level")

        if (vuln_score is None or risk_lvl is None) and isinstance(vuln_metrics, dict):
            for _, val in vuln_metrics.items():
                if isinstance(val, dict):
                    scoring = val.get("scoring") or val.get("vulnerability_scoring") or {}
                    if isinstance(scoring, dict):
                        if vuln_score is None:
                            vuln_score = scoring.get("vulnerability_score")
                        if risk_lvl is None:
                            risk_lvl = scoring.get("risk_level")

        # ── Execution summary ─────────────────────────────────────────────────
        exec_sum_raw = data.get("execution_summary")
        exec_summary = None
        if isinstance(exec_sum_raw, dict):
            exec_summary = ExecutionSummary.from_tracker_dict(exec_sum_raw)
        elif isinstance(exec_sum_raw, ExecutionSummary):
            exec_summary = exec_sum_raw
        # Also try to extract from metadata.tracker (OrchestrationResult format)
        if exec_summary is None:
            tracker = (data.get("metadata") or {}).get("tracker")
            if isinstance(tracker, dict):
                exec_summary = ExecutionSummary.from_tracker_dict(tracker)

        return cls(
            model_info=data.get("model_info") or data.get("model_metadata") or {},
            dataset_config=data.get("dataset_config") or {},
            baseline_performance=(
                data.get("baseline_performance") or data.get("baseline_evaluation") or {}
            ),
            attack_results=data.get("attack_results") or {},
            vulnerability_metrics=vuln_metrics,
            vulnerability_score=vuln_score,
            risk_level=risk_lvl,
            xai_findings=data.get("xai_findings") or data.get("xai_results") or {},
            hardening_results=data.get("hardening_results") or {},
            retest_results=data.get("retest_results") or {},
            before_vs_after=(
                data.get("before_vs_after")
                or data.get("comparisons")
                or data.get("comparison_results")
                or {}
            ),
            execution_summary=exec_summary,
            extra_metadata=data.get("extra_metadata") or {},
        )

    @classmethod
    def from_orchestration_and_retest(
        cls,
        orchestration_result: Any,
        retest_result: Optional[Any] = None,
    ) -> "ReportData":
        """
        Convenience factory: build ReportData from OrchestrationResult and
        optional RetestResult DTOs or plain dicts.

        Args:
            orchestration_result: OrchestrationResult DTO or dict from M8 Orchestrator.
            retest_result: RetestResult DTO or dict from M8 Retest Engine (optional).

        Returns:
            ReportData instance with all sections populated.
        """
        orch_dict: Dict[str, Any] = (
            orchestration_result.to_dict()
            if hasattr(orchestration_result, "to_dict")
            else (orchestration_result if isinstance(orchestration_result, dict) else {})
        )
        retest_dict: Dict[str, Any] = (
            retest_result.to_dict()
            if hasattr(retest_result, "to_dict")
            else (retest_result if isinstance(retest_result, dict) else {})
        )

        # Merge per-attack info (M3 metadata + M2-adversarial evaluation)
        attack_res = orch_dict.get("attack_results") or {}
        adv_evals = orch_dict.get("adversarial_evaluations") or {}
        merged_attacks: Dict[str, Any] = {}
        for atk_name, atk_info in attack_res.items():
            atk_dict = dict(atk_info) if isinstance(atk_info, dict) else {}
            if atk_name in adv_evals:
                atk_dict["evaluation"] = adv_evals[atk_name]
            merged_attacks[atk_name] = atk_dict

        # Extract vulnerability score & risk level
        vuln_analysis = orch_dict.get("vulnerability_analysis") or {}
        vuln_score = None
        risk_level = None
        if isinstance(vuln_analysis, dict):
            for _, val in vuln_analysis.items():
                if isinstance(val, dict) and "scoring" in val:
                    scoring = val["scoring"]
                    if isinstance(scoring, dict):
                        vuln_score = scoring.get("vulnerability_score", vuln_score)
                        risk_level = scoring.get("risk_level", risk_level)

        # Build execution summary
        exec_summary = ExecutionSummary.from_orchestration_result(orch_dict)

        return cls(
            model_info=orch_dict.get("model_metadata") or {},
            dataset_config={},
            baseline_performance=orch_dict.get("baseline_evaluation") or {},
            attack_results=merged_attacks,
            vulnerability_metrics=vuln_analysis,
            vulnerability_score=vuln_score,
            risk_level=risk_level,
            xai_findings=orch_dict.get("xai_results") or {},
            hardening_results=orch_dict.get("hardening_results") or {},
            retest_results=retest_dict,
            before_vs_after=retest_dict.get("comparisons") or {},
            execution_summary=exec_summary,
            extra_metadata={
                "status": orch_dict.get("status"),
                "execution_mode": orch_dict.get("execution_mode"),
                "timestamp": orch_dict.get("timestamp"),
                "execution_time_seconds": orch_dict.get("execution_time_seconds"),
            },
        )
