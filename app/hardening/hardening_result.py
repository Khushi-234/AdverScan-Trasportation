"""
DTO dataclass containing hardening execution outputs and metadata for Module 7.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class HardeningMetadata:
    """
    Metadata describing an executed model hardening process.

    Attributes:
        defense_name: Registered identifier of the defense applied.
        defense_type: Category of defense (e.g. 'preprocessing', 'smoothing', 'adversarial_training').
        parameters: Configuration parameters used for the defense.
        execution_time_seconds: Total time spent applying defense or training.
        timestamp: ISO/formatted timestamp of when hardening was completed.
        extra_metadata: Optional dictionary for additional arbitrary execution context.
    """

    defense_name: str
    defense_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    execution_time_seconds: float = 0.0
    timestamp: Optional[str] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DefenseAttemptResult:
    """
    Evaluation result for an individual defense candidate attempt during iterative defense selection.

    Attributes:
        defense_name: Name/identifier of the attempted defense candidate.
        status: Evaluation status ("accepted", "insufficient", "harmful").
        is_improved: True if any measurable improvement occurred (vuln score or ASR reduced).
        improvement_sufficient: True if improvement satisfies all configured acceptance thresholds.
        vuln_score_before: Baseline vulnerability score before defense.
        vuln_score_after: Vulnerability score after defense.
        vuln_score_improvement: Vulnerability score reduction (before - after).
        asr_before: Baseline Attack Success Rate before defense.
        asr_after: Attack Success Rate after defense.
        asr_reduction: ASR reduction (before - after).
        clean_accuracy_before: Baseline clean accuracy before defense.
        clean_accuracy_after: Clean accuracy after defense.
        clean_accuracy_drop: Clean accuracy drop (before - after).
        clean_accuracy_drop_pct_points: Clean accuracy drop in percentage points.
        latency_ms: Inference latency after defense (ms).
        latency_overhead_ms: Latency overhead compared to baseline (ms).
        parameters: Defense configuration parameters used for this attempt.
        reason: Clear explanation of acceptance, insufficiency, or harm.
    """

    defense_name: str
    status: str
    is_improved: bool
    improvement_sufficient: bool
    vuln_score_before: Optional[float] = None
    vuln_score_after: Optional[float] = None
    vuln_score_improvement: Optional[float] = None
    asr_before: Optional[float] = None
    asr_after: Optional[float] = None
    asr_reduction: Optional[float] = None
    clean_accuracy_before: Optional[float] = None
    clean_accuracy_after: Optional[float] = None
    clean_accuracy_drop: Optional[float] = None
    clean_accuracy_drop_pct_points: Optional[float] = None
    latency_ms: Optional[float] = None
    latency_overhead_ms: Optional[float] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert attempt result to dictionary."""
        return asdict(self)


@dataclass
class HardeningResult:
    """
    Standardized result contract returned by the Hardening Engine.

    Attributes:
        hardened_model: Hardened or wrapped model object (PyTorch module or defense wrapper).
        metadata: HardeningMetadata instance.
        hardened_inputs: Optional preprocessed input tensor or batch (for input-preprocessing defenses).
        hardened_labels: Optional filtered or transformed ground truth labels aligned with hardened_inputs.
        success: Boolean flag indicating if defense execution completed without errors.
        metrics_before: Optional metric mapping before hardening (e.g., baseline/adversarial accuracy).
        metrics_after: Optional metric mapping after hardening (e.g., hardened adversarial accuracy).
        recommendations: Recommended follow-up hardening steps or evaluation notes.
        selected_defense: Name of the defense that was finally selected (or None if none accepted).
        defense_attempts: List of DefenseAttemptResult records evaluated during iterative selection.
        num_attempts: Total count of defense attempts made.
        is_improved: True if any measurable improvement was achieved.
        improvement_sufficient: True if selected defense met configured acceptance thresholds.
        status: Overall result status ("accepted", "insufficient", "harmful").
        thresholds: Optional serialized dictionary of configured acceptance thresholds.
    """

    hardened_model: Any
    metadata: HardeningMetadata
    hardened_inputs: Optional[Any] = None
    hardened_labels: Optional[Any] = None
    success: bool = True
    metrics_before: Dict[str, Any] = field(default_factory=dict)
    metrics_after: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    # Iterative defense selection attributes
    selected_defense: Optional[str] = None
    defense_attempts: List[DefenseAttemptResult] = field(default_factory=list)
    num_attempts: int = 1
    is_improved: bool = False
    improvement_sufficient: bool = False
    status: str = "accepted"
    thresholds: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert HardeningResult metadata, evaluation metrics, and attempts to dictionary format.
        Note: Model tensors and callables are excluded or converted to descriptive strings.
        """
        attempts_dicts = [
            att.to_dict() if hasattr(att, "to_dict") else att
            for att in self.defense_attempts
        ]
        res_dict = {
            "metadata": asdict(self.metadata),
            "success": self.success,
            "status": self.status,
            "selected_defense": self.selected_defense or self.metadata.defense_name,
            "num_attempts": self.num_attempts,
            "is_improved": self.is_improved,
            "improvement_sufficient": self.improvement_sufficient,
            "thresholds": self.thresholds,
            "defense_attempts": attempts_dicts,
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
            "recommendations": self.recommendations,
            "hardened_model_class": self.hardened_model.__class__.__name__ if self.hardened_model is not None else None,
            "has_hardened_inputs": self.hardened_inputs is not None,
        }
        return res_dict

    def save_json(self, output_path: Union[str, Path]) -> None:
        """
        Save non-tensor summary of hardening result to a JSON file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
