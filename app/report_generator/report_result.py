"""
ReportResult — Output DTO for the AdverScan Report Generator (Module 9).

Holds the fully generated, 15-section security report and provides
serialization helpers (JSON, Markdown text).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .execution_summary import ExecutionSummary


@dataclass
class ReportResult:
    """
    Generated AdverScan Security Report DTO.

    Contains all 15 report sections and provides serialization to
    JSON and formatted text (Markdown).
    """

    # Report identity
    report_id: str
    timestamp: str
    status: str = "SUCCESS"
    scan_id: Optional[str] = None

    # Section 1: Executive Summary
    executive_summary: Dict[str, Any] = field(default_factory=dict)

    # Section 2: Model Information
    model_info: Dict[str, Any] = field(default_factory=dict)

    # Section 3: Dataset / Evaluation Configuration
    dataset_config: Dict[str, Any] = field(default_factory=dict)

    # Section 4: Baseline Performance
    baseline_performance: Dict[str, Any] = field(default_factory=dict)

    # Section 5: Adversarial Attack Results
    attack_results: Dict[str, Any] = field(default_factory=dict)

    # Section 6: Vulnerability Assessment
    vulnerability_metrics: Dict[str, Any] = field(default_factory=dict)

    # Section 7: Vulnerability Score + Risk Level
    vulnerability_score: Optional[float] = None
    risk_level: str = "UNKNOWN"

    # Section 8: MITRE ATLAS Mapping
    mitre_atlas_mapping: Dict[str, Any] = field(default_factory=dict)

    # Section 9: XAI Findings
    xai_findings: Dict[str, Any] = field(default_factory=dict)

    # Section 10: Hardening
    hardening_results: Dict[str, Any] = field(default_factory=dict)

    # Section 11: Re-Test Results
    retest_results: Dict[str, Any] = field(default_factory=dict)

    # Section 12: Before vs After Comparison
    before_vs_after: Dict[str, Any] = field(default_factory=dict)

    # Section 13: Execution Performance
    execution_summary: Optional[ExecutionSummary] = None

    # Section 14: Recommendations
    recommendations: List[str] = field(default_factory=list)

    # Section 15: Final Security Summary
    final_security_summary: Dict[str, Any] = field(default_factory=dict)

    # Formatted report string (Markdown)
    formatted_report: str = ""

    # Arbitrary metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Serialization ─────────────────────────────────────────────────────────

    @staticmethod
    def _sanitize(obj: Any) -> Any:
        """Recursively convert non-JSON-serializable objects to primitives."""
        if obj is None or isinstance(obj, (int, float, str, bool)):
            return obj
        if hasattr(obj, "item") and callable(getattr(obj, "item")):
            try:
                return obj.item()
            except Exception:
                pass
        if hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
            try:
                return obj.tolist()
            except Exception:
                pass
        if isinstance(obj, dict):
            return {str(k): ReportResult._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [ReportResult._sanitize(v) for v in obj]
        if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
            return ReportResult._sanitize(obj.to_dict())
        return str(obj)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ReportResult to a plain, JSON-safe dictionary."""
        d = {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "status": self.status,
            "scan_id": self.scan_id,
            "executive_summary": self.executive_summary,
            "model_info": self.model_info,
            "dataset_config": self.dataset_config,
            "baseline_performance": self.baseline_performance,
            "attack_results": self.attack_results,
            "vulnerability_metrics": self.vulnerability_metrics,
            "vulnerability_score": self.vulnerability_score,
            "risk_level": self.risk_level,
            "mitre_atlas_mapping": self.mitre_atlas_mapping,
            "xai_findings": self.xai_findings,
            "hardening_results": self.hardening_results,
            "retest_results": self.retest_results,
            "before_vs_after": self.before_vs_after,
            "execution_summary": (
                self.execution_summary.to_dict() if self.execution_summary else None
            ),
            "recommendations": self.recommendations,
            "final_security_summary": self.final_security_summary,
            "metadata": self.metadata,
        }
        return self._sanitize(d)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save_json(self, output_path: Union[str, Path]) -> None:
        """Save as JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    def save_text(self, output_path: Union[str, Path]) -> None:
        """Save formatted Markdown report as text file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.formatted_report)
