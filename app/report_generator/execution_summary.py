"""
ExecutionSummary — Module 13 of the AdverScan Security Report.

Captures per-module execution timing, status, and performance metrics
as recorded by the pipeline's ResultTracker. This is the data source
for the "Execution Performance" section of the final report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModuleExecutionRecord:
    """
    Single module's execution record within the pipeline run.
    """

    module_id: str
    module_name: str
    status: str                       # "SUCCESS" | "FAILED" | "SKIPPED"
    elapsed_seconds: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def status_icon(self) -> str:
        return {"SUCCESS": "✅", "FAILED": "❌", "SKIPPED": "⏭"}.get(self.status, "❓")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_id": self.module_id,
            "module_name": self.module_name,
            "status": self.status,
            "elapsed_seconds": self.elapsed_seconds,
            "metrics": self.metrics,
            "error": self.error,
        }


@dataclass
class ExecutionSummary:
    """
    Complete execution performance summary across all pipeline modules.

    Populated from ResultTracker.to_dict() or an OrchestrationResult's
    module_timings / metadata['tracker'] fields.
    """

    run_label: str = "AdverScan Pipeline"
    run_timestamp: str = ""
    total_elapsed_seconds: float = 0.0
    modules: List[ModuleExecutionRecord] = field(default_factory=list)

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_tracker_dict(cls, tracker_dict: Dict[str, Any]) -> "ExecutionSummary":
        """
        Build an ExecutionSummary from a ResultTracker.to_dict() payload.

        Args:
            tracker_dict: Output of ResultTracker.to_dict().

        Returns:
            ExecutionSummary instance.
        """
        modules_raw = tracker_dict.get("modules", {})
        records = []
        for mid, mod in modules_raw.items():
            records.append(
                ModuleExecutionRecord(
                    module_id=mid,
                    module_name=mod.get("module_name", mid),
                    status=mod.get("status", "UNKNOWN"),
                    elapsed_seconds=float(mod.get("elapsed_seconds", 0.0)),
                    metrics=mod.get("metrics") or {},
                    error=mod.get("error"),
                )
            )
        return cls(
            run_label=tracker_dict.get("run_label", "AdverScan Pipeline"),
            run_timestamp=tracker_dict.get("run_timestamp", ""),
            total_elapsed_seconds=float(tracker_dict.get("total_elapsed_seconds", 0.0)),
            modules=records,
        )

    @classmethod
    def from_orchestration_result(cls, orch_dict: Dict[str, Any]) -> "ExecutionSummary":
        """
        Build an ExecutionSummary from an OrchestrationResult.to_dict() payload.
        Falls back to module_timings if full tracker data is unavailable.

        Args:
            orch_dict: Output of OrchestrationResult.to_dict().

        Returns:
            ExecutionSummary instance.
        """
        # Try full tracker payload first (populated by the updated orchestrator)
        tracker = (orch_dict.get("metadata") or {}).get("tracker")
        if isinstance(tracker, dict) and "modules" in tracker:
            return cls.from_tracker_dict(tracker)

        # Fallback — reconstruct from module_timings flat dict
        module_timings = orch_dict.get("module_timings") or {}
        records = [
            ModuleExecutionRecord(
                module_id=mid,
                module_name=mid.replace("_", " ").upper(),
                status="SUCCESS",
                elapsed_seconds=float(elapsed),
            )
            for mid, elapsed in module_timings.items()
        ]
        return cls(
            run_label=f"AdverScan [{orch_dict.get('execution_mode', 'pipeline')}]",
            run_timestamp=orch_dict.get("timestamp", ""),
            total_elapsed_seconds=float(orch_dict.get("execution_time_seconds", 0.0)),
            modules=records,
        )

    # ── Accessors ──────────────────────────────────────────────────────────────

    @property
    def failed_modules(self) -> List[ModuleExecutionRecord]:
        return [r for r in self.modules if r.status == "FAILED"]

    @property
    def succeeded_modules(self) -> List[ModuleExecutionRecord]:
        return [r for r in self.modules if r.status == "SUCCESS"]

    @property
    def overall_status(self) -> str:
        if not self.modules:
            return "UNKNOWN"
        if any(r.status == "FAILED" for r in self.modules):
            return "PARTIAL_SUCCESS" if any(r.status == "SUCCESS" for r in self.modules) else "FAILED"
        return "SUCCESS"

    # ── Serialization ──────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_label": self.run_label,
            "run_timestamp": self.run_timestamp,
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "overall_status": self.overall_status,
            "modules": [r.to_dict() for r in self.modules],
        }
