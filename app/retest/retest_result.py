"""
DTO dataclasses for Module 8 (Re-Test & Comparison) in AdverScan.

Provides structured results for before vs after hardening metric comparisons and re-test execution.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class ComparisonResult:
    """
    DTO holding comparative evaluation metrics (BEFORE vs AFTER hardening) for a specific attack.

    Calculates changes (Δ = After - Before) across:
    - Attack Success Rate (ASR)
    - Perturbation
    - Accuracy Drop
    - F1 Drop
    - Confidence Drop
    - Model Degradation
    - Vulnerability Score
    - Risk Level
    """

    attack_name: str
    before_assessment: Dict[str, Any]
    after_assessment: Dict[str, Any]
    before_scoring: Dict[str, Any]
    after_scoring: Dict[str, Any]
    delta_attack_success_rate: Optional[float]
    delta_accuracy_drop: float
    delta_f1_drop: float
    delta_confidence_drop: float
    delta_model_degradation: float
    delta_vulnerability_score: float
    delta_clean_accuracy: float
    delta_adversarial_accuracy: float
    delta_perturbation: Dict[str, Any] = field(default_factory=dict)
    before_risk_level: str = "UNKNOWN"
    after_risk_level: str = "UNKNOWN"
    risk_level_changed: bool = False
    is_improved: bool = True
    summary_notes: List[str] = field(default_factory=list)
    timestamp: Optional[str] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ComparisonResult object to dictionary."""
        return asdict(self)

    def save_json(self, output_path: Union[str, Path]) -> None:
        """Save comparison result to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class RetestResult:
    """
    Main DTO containing complete re-test outputs, before results, after results,
    and comparison results.
    """

    hardened_model_name: Optional[str] = None
    dataset_name: Optional[str] = None
    num_samples: int = 0
    before_baseline_evaluation: Dict[str, Any] = field(default_factory=dict)
    after_baseline_evaluation: Dict[str, Any] = field(default_factory=dict)
    before_attack_results: Dict[str, Any] = field(default_factory=dict)
    after_attack_results: Dict[str, Any] = field(default_factory=dict)
    before_vulnerability_analysis: Dict[str, Any] = field(default_factory=dict)
    after_vulnerability_analysis: Dict[str, Any] = field(default_factory=dict)
    comparisons: Dict[str, ComparisonResult] = field(default_factory=dict)
    overall_improved: bool = True
    timestamp: Optional[str] = None
    execution_time_seconds: float = 0.0
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert RetestResult object to dictionary representation."""
        res_dict = {
            "hardened_model_name": self.hardened_model_name,
            "dataset_name": self.dataset_name,
            "num_samples": self.num_samples,
            "before_baseline_evaluation": self.before_baseline_evaluation,
            "after_baseline_evaluation": self.after_baseline_evaluation,
            "before_attack_results": self.before_attack_results,
            "after_attack_results": self.after_attack_results,
            "before_vulnerability_analysis": self.before_vulnerability_analysis,
            "after_vulnerability_analysis": self.after_vulnerability_analysis,
            "comparisons": {k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in self.comparisons.items()},
            "overall_improved": self.overall_improved,
            "timestamp": self.timestamp,
            "execution_time_seconds": self.execution_time_seconds,
            "extra_metadata": self.extra_metadata,
        }
        return res_dict

    def save_json(self, output_path: Union[str, Path]) -> None:
        """Save re-test result summary to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
