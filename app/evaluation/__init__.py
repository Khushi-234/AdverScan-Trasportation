"""
Module 2 — Baseline Evaluation for AdverScan framework.
"""

from app.evaluation.dataset_loader import BaseDatasetLoader, GTSRBDatasetLoader
from app.evaluation.evaluator import BaselineEvaluator, evaluate_baseline
from app.evaluation.metrics import MetricsCalculator
from app.evaluation.results import EvaluationResult

__all__ = [
    "EvaluationResult",
    "MetricsCalculator",
    "BaseDatasetLoader",
    "GTSRBDatasetLoader",
    "BaselineEvaluator",
    "evaluate_baseline",
]
