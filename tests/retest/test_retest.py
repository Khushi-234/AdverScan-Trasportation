"""
Unit and Integration Tests for Module 8 (Re-Test & Comparison) in AdverScan.
"""

import pytest
import torch
import torch.nn as nn

from app.ingestion.adapters.pytorch_adapter import PyTorchAdapter
from app.orchestration.dataset_adapter import InMemoryDatasetLoader
from app.evaluation.results import EvaluationResult
from app.vulnerability_analysis.assessment_result import AssessmentResult
from app.vulnerability_analysis.scoring_result import ScoringResult
from app.retest.comparison import ComparisonEngine, compare_results
from app.retest.retest_result import ComparisonResult, RetestResult
from app.retest.retest_engine import RetestEngine


class SimpleToyModel(nn.Module):
    """Simple linear PyTorch model for testing."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.fc(x)


@pytest.fixture
def mock_assessments_and_scores():
    before_assessment = {
        "attack_name": "fgsm",
        "dataset_name": "test_ds",
        "num_samples": 10,
        "attack_success_rate": 0.8,
        "perturbation": {"l2_mean": 0.5},
        "accuracy_drop": 0.7,
        "f1_drop": 0.7,
        "confidence_drop": 0.5,
        "model_degradation": 0.6333,
        "clean_accuracy": 0.9,
        "adversarial_accuracy": 0.2,
        "clean_f1": 0.9,
        "adversarial_f1": 0.2,
        "clean_confidence": 0.9,
        "adversarial_confidence": 0.4,
    }
    after_assessment = {
        "attack_name": "fgsm",
        "dataset_name": "test_ds",
        "num_samples": 10,
        "attack_success_rate": 0.3,
        "perturbation": {"l2_mean": 0.5},
        "accuracy_drop": 0.2,
        "f1_drop": 0.2,
        "confidence_drop": 0.1,
        "model_degradation": 0.1667,
        "clean_accuracy": 0.88,
        "adversarial_accuracy": 0.68,
        "clean_f1": 0.88,
        "adversarial_f1": 0.68,
        "clean_confidence": 0.88,
        "adversarial_confidence": 0.78,
    }
    before_scoring = {
        "attack_name": "fgsm",
        "vulnerability_score": 75.0,
        "risk_level": "HIGH",
    }
    after_scoring = {
        "attack_name": "fgsm",
        "vulnerability_score": 25.0,
        "risk_level": "LOW",
    }

    return {
        "before_vuln": {"fgsm": {"assessment": before_assessment, "scoring": before_scoring}},
        "after_vuln": {"fgsm": {"assessment": after_assessment, "scoring": after_scoring}},
        "before_assessment": before_assessment,
        "after_assessment": after_assessment,
        "before_scoring": before_scoring,
        "after_scoring": after_scoring,
    }


def test_comparison_engine(mock_assessments_and_scores):
    engine = ComparisonEngine()
    res: ComparisonResult = engine.compare_attack(
        before_assessment=mock_assessments_and_scores["before_assessment"],
        after_assessment=mock_assessments_and_scores["after_assessment"],
        before_scoring=mock_assessments_and_scores["before_scoring"],
        after_scoring=mock_assessments_and_scores["after_scoring"],
    )

    assert res.attack_name == "fgsm"
    assert res.delta_attack_success_rate == -0.5
    assert res.delta_vulnerability_score == -50.0
    assert res.delta_accuracy_drop == -0.5
    assert res.before_risk_level == "HIGH"
    assert res.after_risk_level == "LOW"
    assert res.risk_level_changed is True
    assert res.is_improved is True
    assert len(res.summary_notes) > 0


def test_retest_engine_execution(mock_assessments_and_scores):
    torch.manual_seed(42)
    inputs = torch.randn(10, 3, 32, 32)
    labels = torch.randint(0, 10, (10,))

    toy_model = SimpleToyModel(num_classes=10)
    hardened_adapter = PyTorchAdapter(toy_model)

    dataset_loader = InMemoryDatasetLoader(inputs=inputs, targets=labels, dataset_name="toy_dataset")

    before_eval_result = EvaluationResult(
        dataset_name="toy_dataset",
        model_name="toy_model",
        num_samples=10,
        num_classes=10,
        accuracy=0.9,
        precision_macro=0.9,
        recall_macro=0.9,
        f1_macro=0.9,
        precision_weighted=0.9,
        recall_weighted=0.9,
        f1_weighted=0.9,
        average_confidence=0.9,
        average_entropy=0.1,
    )

    retest_engine = RetestEngine()
    retest_res: RetestResult = retest_engine.retest(
        hardened_model=hardened_adapter,
        dataset_loader=dataset_loader,
        attacks=["fgsm"],
        before_baseline_result=before_eval_result,
        before_vulnerability_analysis=mock_assessments_and_scores["before_vuln"],
        num_classes=10,
        model_name="toy_model",
    )

    assert isinstance(retest_res, RetestResult)
    assert "fgsm" in retest_res.comparisons
    assert retest_res.num_samples == 10
    assert retest_res.after_baseline_evaluation["accuracy"] is not None
    assert retest_res.to_dict() is not None
