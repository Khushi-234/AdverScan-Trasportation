"""
ReportResult — Output DTO for the AdverScan Report Generator (Module 9).

Holds the fully generated, 15-section security report and provides
serialization helpers (JSON, Markdown text).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import io
import json
import math
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
        if obj is None or isinstance(obj, (str, bool)):
            return obj
        if isinstance(obj, (int, float)):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        if hasattr(obj, "item") and callable(getattr(obj, "item")):
            try:
                val = obj.item()
                if isinstance(val, (int, float)):
                    if math.isnan(val) or math.isinf(val):
                        return None
                    return val
                return val
            except Exception:
                pass
        if hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
            try:
                return ReportResult._sanitize(obj.tolist())
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

    def to_csv(self) -> str:
        """Serialize execution performance and module timings to CSV string."""
        buf = io.StringIO()
        es = self.execution_summary
        if es is None or not es.modules:
            writer = csv.writer(buf)
            writer.writerow(["module_id", "module_name", "status", "elapsed_seconds"])
            writer.writerow(["N/A", "No execution data captured", "UNKNOWN", 0.0])
            return buf.getvalue()

        metric_keys: List[str] = []
        for rec in es.modules:
            for k in (rec.metrics or {}):
                col = f"metric_{k}"
                if col not in metric_keys:
                    metric_keys.append(col)

        header = ["module_id", "module_name", "status", "elapsed_seconds"] + metric_keys
        dict_writer = csv.DictWriter(buf, fieldnames=header, extrasaction="ignore")
        dict_writer.writeheader()
        for rec in es.modules:
            row: Dict[str, Any] = {
                "module_id": rec.module_id,
                "module_name": rec.module_name,
                "status": rec.status,
                "elapsed_seconds": rec.elapsed_seconds,
            }
            for mk, mv in (rec.metrics or {}).items():
                if mv is None or (isinstance(mv, float) and (math.isnan(mv) or math.isinf(mv))):
                    row[f"metric_{mk}"] = "N/A"
                else:
                    row[f"metric_{mk}"] = mv
            dict_writer.writerow(row)
        writer = csv.writer(buf)
        writer.writerow([])
        writer.writerow(["TOTAL", es.run_label, es.overall_status, es.total_elapsed_seconds])
        return buf.getvalue()

    def save_csv(self, output_path: Union[str, Path]) -> None:
        """Save execution performance and module timings as CSV file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write(self.to_csv())
