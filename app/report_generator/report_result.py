"""
DTO dataclass for Module 9 (Report Generator) in AdverScan.

Stores the generated security report content, aggregated sections, recommendations,
formatted text output, and metadata.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class ReportResult:
    """
    Data Transfer Object for the generated AdverScan Security Report.
    """

    report_id: str
    timestamp: str
    status: str = "SUCCESS"
    model_info: Dict[str, Any] = field(default_factory=dict)
    baseline_performance: Dict[str, Any] = field(default_factory=dict)
    attack_results: Dict[str, Any] = field(default_factory=dict)
    vulnerability_metrics: Dict[str, Any] = field(default_factory=dict)
    vulnerability_score: Optional[float] = None
    risk_level: str = "UNKNOWN"
    xai_findings: Dict[str, Any] = field(default_factory=dict)
    hardening_results: Dict[str, Any] = field(default_factory=dict)
    before_vs_after: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    formatted_report: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _sanitize(obj: Any) -> Any:
        """Recursively convert PyTorch tensors, NumPy types, and non-serializable objects to JSON primitives."""
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
        """Convert ReportResult object to a dictionary representation."""
        raw_dict = asdict(self)
        return self._sanitize(raw_dict)

    def to_json(self, indent: int = 2) -> str:
        """Serialize ReportResult dictionary to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, output_path: Union[str, Path]) -> None:
        """
        Save report result dictionary as a JSON file.

        Args:
            output_path: Target file path to write JSON.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_text(self, output_path: Union[str, Path]) -> None:
        """
        Save formatted report string as a text/markdown file.

        Args:
            output_path: Target file path to write text/markdown report.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.formatted_report)
