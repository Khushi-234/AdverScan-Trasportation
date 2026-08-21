"""
Orchestration result DTO dataclass for Module 8 (Orchestration).
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class OrchestrationResult:
    """
    Unified result container returned by AdverScanOrchestrator.
    Composes serialized DTOs from existing modules M1–M7.
    """

    status: str  # "SUCCESS", "PARTIAL_SUCCESS", "FAILED"
    execution_mode: str
    execution_time_seconds: float = 0.0
    timestamp: Optional[str] = None

    # Composed Module Results
    model_metadata: Optional[Dict[str, Any]] = None
    baseline_evaluation: Optional[Dict[str, Any]] = None
    attack_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    adversarial_evaluations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    vulnerability_analysis: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    xai_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    hardening_results: Optional[Dict[str, Any]] = None

    # Error Tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert OrchestrationResult to dictionary."""
        return asdict(self)

    def save_json(self, output_path: Union[str, Path]) -> None:
        """Save orchestration result summary to JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
