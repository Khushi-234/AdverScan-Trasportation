"""
M6 - XAI Explainability Module for AdverScan.

Provides model explainability, feature attribution, clean vs. adversarial
comparison, failure analysis, and structured result DTOs.
"""

from app.explainability.explanation_result import ExplanationResult
from app.explainability.comparison import compare_explanations, compare_attributions
from app.explainability.failure_analysis import analyze_failure, is_prediction_correct
from app.explainability.explainer import XAIExplainer
from app.explainability.techniques import SHAPExplainer, LIMEExplainer

__all__ = [
    "ExplanationResult",
    "compare_explanations",
    "compare_attributions",
    "analyze_failure",
    "is_prediction_correct",
    "XAIExplainer",
    "SHAPExplainer",
    "LIMEExplainer",
]
