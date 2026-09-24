"""
Module 2 — Baseline Evaluation for AdverScan framework.
"""

from app.evaluation.dataset_loader import (
    BaseDatasetLoader,
    GTSRBDatasetLoader,
    GenericDatasetLoader,
    HFTextDatasetLoader,
    HFVisionDatasetLoader,
    TabularDatasetLoader,
    TimeSeriesDatasetLoader,
    get_dataset_loader,
)
from app.evaluation.evaluator import BaselineEvaluator, evaluate_baseline
from app.evaluation.metrics import MetricsCalculator
from app.evaluation.results import EvaluationResult

__all__ = [
    "EvaluationResult",
    "MetricsCalculator",
    "BaseDatasetLoader",
    "GTSRBDatasetLoader",
    "HFVisionDatasetLoader",
    "HFTextDatasetLoader",
    "TimeSeriesDatasetLoader",
    "TabularDatasetLoader",
    "GenericDatasetLoader",
    "get_dataset_loader",
    "BaselineEvaluator",
    "evaluate_baseline",
]
