"""
Evaluation result container for storing baseline and evaluation metrics in AdverScan.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EvaluationResult:
    """
    Structured container for machine learning evaluation metrics.
    """

    dataset_name: str
    model_name: str
    num_samples: int
    num_classes: int
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    average_confidence: float
    average_entropy: float
    per_class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion_matrix: List[List[int]] = field(default_factory=list)
    batch_size: int = 32
    device: str = "cpu"
    timestamp: Optional[str] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation result object to dictionary."""
        return asdict(self)

    def save_json(self, output_path: str | Path) -> None:
        """Save evaluation metrics to JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def log_to_mlflow(self, run_name: Optional[str] = None) -> bool:
        """
        Log evaluation metrics to an active MLflow run if present.
        Does NOT independently start or manage MLflow run lifecycle.
        Centralized lifecycle ownership belongs to MLflowTracker in app.utils.mlflow_tracker.
        Returns True if logged to active run, False otherwise.
        """
        try:
            import mlflow
            if mlflow.active_run() is None:
                return False

            mlflow.log_params({
                "dataset_name": self.dataset_name,
                "model_name": self.model_name,
                "num_samples": self.num_samples,
                "num_classes": self.num_classes,
                "batch_size": self.batch_size,
                "device": self.device,
            })
            mlflow.log_metrics({
                "accuracy": self.accuracy,
                "precision_macro": self.precision_macro,
                "recall_macro": self.recall_macro,
                "f1_macro": self.f1_macro,
                "precision_weighted": self.precision_weighted,
                "recall_weighted": self.recall_weighted,
                "f1_weighted": self.f1_weighted,
                "average_confidence": self.average_confidence,
                "average_entropy": self.average_entropy,
            })
            return True
        except Exception:
            return False

