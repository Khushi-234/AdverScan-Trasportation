"""
Integration unit tests for AdverScanOrchestrator in Module 8 (Orchestration).
"""

from typing import Generator, List, Tuple
import pytest
import torch
import torch.nn as nn

from app.evaluation.dataset_loader import BaseDatasetLoader
from app.orchestration import AdverScanOrchestrator, PipelineConfig, OrchestrationResult


class DummyModel(nn.Module):
    """Synthetic linear PyTorch model for fast CPU unit testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DummyDatasetLoader(BaseDatasetLoader):
    """Synthetic dataset loader for fast CPU unit testing."""

    def __init__(self, num_samples: int = 16, batch_size: int = 4):
        self._num_samples = num_samples
        self.batch_size = batch_size
        self.inputs = torch.randn(num_samples, 10)
        self.targets = torch.randint(0, 3, (num_samples,))

    @property
    def dataset_name(self) -> str:
        return "DummyDataset"

    def __len__(self) -> int:
        return self._num_samples

    def iterate_batches(
        self,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, List[int]], None, None]:
        for i in range(0, self._num_samples, self.batch_size):
            yield (
                self.inputs[i : i + self.batch_size],
                self.targets[i : i + self.batch_size],
                self.targets[i : i + self.batch_size].tolist(),
            )


@pytest.fixture
def dummy_setup():
    model = DummyModel()
    dataset_loader = DummyDatasetLoader()
    sample_input = torch.randn(1, 10)
    return model, dataset_loader, sample_input


def test_orchestrator_baseline_only(dummy_setup):
    model, dataset_loader, sample_input = dummy_setup
    config = PipelineConfig(
        model_path=model,
        sample_input=sample_input,
        num_classes=3,
        mode="baseline_only",
        custom_dataset_loader=dataset_loader,
    )

    orchestrator = AdverScanOrchestrator()
    res: OrchestrationResult = orchestrator.run(config)

    assert res.status == "SUCCESS"
    assert res.execution_mode == "baseline_only"
    assert res.model_metadata is not None
    assert res.baseline_evaluation is not None
    assert len(res.attack_results) == 0


def test_orchestrator_attack_assessment(dummy_setup):
    model, dataset_loader, sample_input = dummy_setup
    config = PipelineConfig(
        model_path=model,
        sample_input=sample_input,
        num_classes=3,
        mode="attack_assessment",
        attacks=["fgsm"],
        custom_dataset_loader=dataset_loader,
    )

    orchestrator = AdverScanOrchestrator()
    res: OrchestrationResult = orchestrator.run(config)

    assert res.status == "SUCCESS"
    assert res.execution_mode == "attack_assessment"
    assert "fgsm" in res.attack_results
    assert "fgsm" in res.adversarial_evaluations
    assert "fgsm" in res.vulnerability_analysis


def test_orchestrator_full_pipeline(dummy_setup):
    model, dataset_loader, sample_input = dummy_setup
    config = PipelineConfig(
        model_path=model,
        sample_input=sample_input,
        num_classes=3,
        mode="full",
        attacks=["fgsm"],
        enable_xai=False,
        enable_hardening=True,
        defense="spatial_smoothing",
        custom_dataset_loader=dataset_loader,
    )

    orchestrator = AdverScanOrchestrator()
    res: OrchestrationResult = orchestrator.run(config)

    assert res.status == "SUCCESS"
    assert res.execution_mode == "full"
    assert "fgsm" in res.vulnerability_analysis
    assert res.hardening_results is not None
    assert res.hardening_results["success"] is True
    assert res.retest_results is not None
    assert res.report_result is not None
    assert res.report_result["status"] == "SUCCESS"


def test_orchestrator_fatal_m1_failure():
    config = PipelineConfig(
        model_path="/non_existent_file_path_12345.pt",
        mode="baseline_only",
    )
    orchestrator = AdverScanOrchestrator()
    res = orchestrator.run(config)

    assert res.status == "FAILED"
    assert len(res.errors) > 0
    assert res.errors[0]["module"] == "M1_ingestion"
