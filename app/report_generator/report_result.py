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

    def to_dict(self) -> Dict[str, Any]:
        """Convert ReportResult object to a dictionary representation."""
        return asdict(self)

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
