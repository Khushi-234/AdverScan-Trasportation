"""
Configuration dataclass and presets for AdverScan XAI module (Module 5 Step 2).
Defines evaluation modes (fast, standard, thorough), sample budgets, technique parameters,
and batching configurations.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class XAIConfig:
    """
    Structured configuration governing XAI explanation generation and cost control.

    Attributes:
        mode: Operational mode preset ('fast' | 'standard' | 'thorough').
        technique: Optional manual override for XAI technique.
        budget: Maximum samples to explain during evaluation runs.
        failed_attack_samples: Maximum failed adversarial attack samples retained for contrast.
        per_sample_timeout_s: Maximum execution timeout in seconds per sample.
        ig_steps: Step count along interpolation path for Integrated Gradients.
        shap_max_evals: Maximum model evaluations for SHAP explainers.
        lime_num_samples: Number of perturbation samples generated for LIME.
        batch_size: Batch size for chunked model inference in predict_batched.
        max_text_length: Maximum token length for text truncation in NLP models.
        seed: Random seed for reproducible perturbations and sampling.
    """

    mode: str = "standard"  # fast | standard | thorough
    technique: Optional[str] = None
    budget: int = 100
    failed_attack_samples: int = 10
    per_sample_timeout_s: float = 30.0
    ig_steps: int = 25
    shap_max_evals: int = 100
    lime_num_samples: int = 200
    batch_size: int = 64
    max_text_length: int = 128
    seed: int = 0

    def __post_init__(self) -> None:
        """Apply mode presets when specific parameters are not explicitly modified."""
        norm_mode = str(self.mode).lower()
        if norm_mode == "fast":
            if self.ig_steps == 25:
                self.ig_steps = 10
            if self.shap_max_evals == 100:
                self.shap_max_evals = 50
            if self.lime_num_samples == 200:
                self.lime_num_samples = 50
            if self.budget == 100:
                self.budget = 20
        elif norm_mode == "thorough":
            if self.ig_steps == 25:
                self.ig_steps = 50
            if self.shap_max_evals == 100:
                self.shap_max_evals = 250
            if self.lime_num_samples == 200:
                self.lime_num_samples = 500
            if self.budget == 100:
                self.budget = 250

    def to_dict(self) -> Dict[str, Any]:
        """Serialize XAIConfig to a standard dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "XAIConfig":
        """Instantiate XAIConfig from a dictionary, ignoring extraneous fields."""
        valid_keys = {
            "mode",
            "technique",
            "budget",
            "failed_attack_samples",
            "per_sample_timeout_s",
            "ig_steps",
            "shap_max_evals",
            "lime_num_samples",
            "batch_size",
            "max_text_length",
            "seed",
        }
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def fast(cls, **kwargs: Any) -> "XAIConfig":
        """Factory for lightweight, high-speed XAI evaluation configuration."""
        cfg = cls(mode="fast", **kwargs)
        return cfg

    @classmethod
    def standard(cls, **kwargs: Any) -> "XAIConfig":
        """Factory for balanced, standard XAI evaluation configuration."""
        cfg = cls(mode="standard", **kwargs)
        return cfg

    @classmethod
    def thorough(cls, **kwargs: Any) -> "XAIConfig":
        """Factory for thorough, high-precision XAI evaluation configuration."""
        cfg = cls(mode="thorough", **kwargs)
        return cfg
