"""
M5 - XAI Explainability Module for AdverScan.

Provides multi-modal model explainability (image, text, tabular, time-series),
feature attribution, clean vs. adversarial comparison, failure analysis,
and structured result DTOs.
"""

from app.explainability.aggregate import aggregate_explanations
from app.explainability.comparison import compare_attributions, compare_explanations
from app.explainability.domain_profiles import DomainProfile, get_domain_profile, is_valid_adversarial
from app.explainability.explainer import XAIExplainer
from app.explainability.explanation_result import ExplanationResult
from app.explainability.failure_analysis import analyze_failure, is_prediction_correct
from app.explainability.metrics import (
    compute_faithfulness,
    compute_perturbation_alignment,
    compute_stability,
)
from app.explainability.modalities import (
    BaseModalityHandler,
    ImageModalityHandler,
    TabularModalityHandler,
    TextModalityHandler,
    TimeSeriesModalityHandler,
    get_modality_handler,
)
from app.explainability.modality import Modality, detect_modality, is_valid_modality
from app.explainability.techniques import (
    BaseExplainer,
    GradientExplainer,
    LIMEExplainer,
    SHAPExplainer,
    TreeSHAPExplainer,
    TechniqueRegistry,
    technique_registry,
)

__all__ = [
    # Core contracts & coordinator
    "ExplanationResult",
    "XAIExplainer",
    "compare_explanations",
    "compare_attributions",
    "analyze_failure",
    "is_prediction_correct",
    "aggregate_explanations",
    # Modalities
    "Modality",
    "detect_modality",
    "is_valid_modality",
    "BaseModalityHandler",
    "ImageModalityHandler",
    "TextModalityHandler",
    "TabularModalityHandler",
    "TimeSeriesModalityHandler",
    "get_modality_handler",
    # Domain configuration
    "DomainProfile",
    "get_domain_profile",
    "is_valid_adversarial",
    # Techniques
    "BaseExplainer",
    "SHAPExplainer",
    "TreeSHAPExplainer",
    "LIMEExplainer",
    "GradientExplainer",
    "TechniqueRegistry",
    "technique_registry",
    # Metrics
    "compute_faithfulness",
    "compute_perturbation_alignment",
    "compute_stability",
]
