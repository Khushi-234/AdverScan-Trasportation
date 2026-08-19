"""
ExplanationResult DTO for the explainability module in AdverScan.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass
class ExplanationResult:
    """
    Structured DTO holding XAI explanation outputs, clean vs. adversarial comparison,
    and prediction failure analysis metrics.
    """

    attack_name: str
    technique: str
    clean_prediction: Any
    adversarial_prediction: Any
    clean_confidence: float
    adversarial_confidence: float
    prediction_changed: bool
    true_label: Optional[Any] = None
    attack_caused_failure: Optional[bool] = None
    attribution: Dict[str, Any] = field(default_factory=dict)
    comparison: Dict[str, Any] = field(default_factory=dict)
    failure_analysis: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ExplanationResult object to dictionary."""
        return asdict(self)

    def save_json(self, output_path: Union[str, Path]) -> None:
        """Save explanation result metrics to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
