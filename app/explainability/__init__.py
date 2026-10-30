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
from app.explainability.config import XAIConfig
from app.explainability.utils import predict_batched, predict_proba_np
from app.explainability.profiling import (
    ProfileContext,
    ProfileStats,
    QueryCounter,
    profile_explainer_call,
)
from app.explainability.cache import ExplanationCache, cache_key



def __getattr__(name: str):
    if name in ("XAIPipeline", "run_xai_pipeline"):
        from app.explainability import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # Core contracts & coordinator
    "ExplanationResult",
    "XAIExplainer",
    "XAIConfig",
    "XAIPipeline",
    "run_xai_pipeline",
    "ExplanationCache",
    "cache_key",
    "predict_batched",
    "predict_proba_np",
    "compare_explanations",
    "compare_attributions",
    "analyze_failure",
    "is_prediction_correct",
    "aggregate_explanations",
    # Profiling & measurement
    "ProfileStats",
    "ProfileContext",
    "QueryCounter",
    "profile_explainer_call",
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
