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
class HardeningResult:
    """
    Standardized result contract returned by the Hardening Engine.

    Attributes:
        hardened_model: Hardened or wrapped model object (PyTorch module or defense wrapper).
        metadata: HardeningMetadata instance.
        hardened_inputs: Optional preprocessed input tensor or batch (for input-preprocessing defenses).
        success: Boolean flag indicating if defense execution completed without errors.
        metrics_before: Optional metric mapping before hardening (e.g., baseline/adversarial accuracy).
        metrics_after: Optional metric mapping after hardening (e.g., hardened adversarial accuracy).
        recommendations: Recommended follow-up hardening steps or evaluation notes.
    """

    hardened_model: Any
    metadata: HardeningMetadata
    hardened_inputs: Optional[Any] = None
    success: bool = True
    metrics_before: Dict[str, Any] = field(default_factory=dict)
    metrics_after: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert HardeningResult metadata and evaluation metrics to dictionary format.
        Note: Model tensors and callables are excluded or converted to descriptive strings.
        """
        res_dict = {
            "metadata": asdict(self.metadata),
            "success": self.success,
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
