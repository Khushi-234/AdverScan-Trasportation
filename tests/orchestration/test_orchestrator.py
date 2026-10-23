"""
Integration unit tests for AdverScanOrchestrator in Module 8 (Orchestration).
"""

from typing import Any, Generator, List, Tuple
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


def test_orchestrator_multi_batch_evaluation():
    """Verify that multi-batch datasets (5 samples, batch size 2 -> 3 batches) process all 5 samples across all pipeline steps."""
    model = DummyModel()
    dataset_loader = DummyDatasetLoader(num_samples=5, batch_size=2)  # 3 batches: [2, 2, 1]
    sample_input = torch.randn(1, 10)

    config = PipelineConfig(
        model_path=model,
        sample_input=sample_input,
        num_classes=3,
        mode="full",
        attacks=["fgsm"],
        enable_xai=False,
        enable_hardening=False,
        custom_dataset_loader=dataset_loader,
    )

    orchestrator = AdverScanOrchestrator()
    res: OrchestrationResult = orchestrator.run(config)

    assert res.status == "SUCCESS"
    # Baseline evaluation must represent all 5 samples
    assert res.baseline_evaluation["num_samples"] == 5

    # Adversarial evaluation must represent all 5 samples
    assert res.adversarial_evaluations["fgsm"]["num_samples"] == 5

    # Vulnerability analysis assessment must represent all 5 samples
    assess_res = res.vulnerability_analysis["fgsm"]["assessment"]
    assert assess_res["num_samples"] == 5


def test_orchestrator_module_timings(dummy_setup):
    """Verify that orchestrator records module timings in metadata and ExecutionSummary populates modules."""
    from app.report_generator.execution_summary import ExecutionSummary

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
    assert "module_timings" in res.metadata
    timings = res.metadata["module_timings"]
    assert "M1_ingestion" in timings
    assert "M2_baseline" in timings
    assert "M3_attack_engine" in timings
    assert "M5_vulnerability_analysis" in timings
    assert "M7_hardening" in timings
    assert "M8_retest" in timings
    assert "M9_report_generator" in timings

    # Verify ExecutionSummary consumes these timings and populates modules
    exec_summary = ExecutionSummary.from_orchestration_result(res.to_dict())
    assert len(exec_summary.modules) >= 7
    module_ids = [m.module_id for m in exec_summary.modules]
    assert "M1_ingestion" in module_ids
    assert "M2_baseline" in module_ids
    assert all(m.status == "SUCCESS" for m in exec_summary.modules)
    assert exec_summary.total_elapsed_seconds == res.execution_time_seconds


class DummyTextModel(nn.Module):
    """Synthetic text classification model for fast unit testing."""

    def __init__(self, vocab_size: int = 100, embed_dim: int = 16, num_classes: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids: Any = None, attention_mask: Any = None, **kwargs: Any) -> Any:
        if isinstance(input_ids, dict):
            attention_mask = input_ids.get("attention_mask")
            input_ids = input_ids.get("input_ids")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        emb = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (emb * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        return self.fc(pooled)


class DummyTextDatasetLoader(BaseDatasetLoader):
    """Synthetic text dataset loader yielding batches of variable sequence lengths."""

    def __init__(self):
        self.batch_size = 4
        # Batch 0 has sequence length 12, Batch 1 has sequence length 9
        self.batches = [
            (
                {
                    "input_ids": torch.randint(1, 90, (4, 12)),
                    "attention_mask": torch.ones(4, 12, dtype=torch.long),
                },
                torch.tensor([0, 1, 0, 1], dtype=torch.long),
                [0, 1, 0, 1],
            ),
            (
                {
                    "input_ids": torch.randint(1, 90, (4, 9)),
                    "attention_mask": torch.ones(4, 9, dtype=torch.long),
                },
                torch.tensor([1, 1, 0, 0], dtype=torch.long),
                [1, 1, 0, 0],
            ),
        ]

    @property
    def dataset_name(self) -> str:
        return "DummyVariableLengthTextDataset"

    def __len__(self) -> int:
        return 8

    def iterate_batches(self):
        for b in self.batches:
            yield b


def test_orchestrator_variable_length_text_multibatch():
    """Verify orchestrator successfully evaluates and attacks multi-batch text with variable sequence lengths."""
    model = DummyTextModel()
    dataset_loader = DummyTextDatasetLoader()
    sample_input = {
        "input_ids": torch.randint(1, 90, (1, 8)),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
    }

    config = PipelineConfig(
        model_path=model,
        sample_input=sample_input,
        num_classes=2,
        mode="attack_assessment",
        attacks=["embedding_pgd"],
        enable_xai=False,
        enable_hardening=False,
        custom_dataset_loader=dataset_loader,
    )

    orchestrator = AdverScanOrchestrator()
    res: OrchestrationResult = orchestrator.run(config)

    assert res.status == "SUCCESS"
    assert "embedding_pgd" in res.attack_results
    assert res.baseline_evaluation["num_samples"] == 8
    assert res.adversarial_evaluations["embedding_pgd"]["num_samples"] == 8
    assert "embedding_pgd" in res.vulnerability_analysis
    assert res.vulnerability_analysis["embedding_pgd"]["assessment"]["num_samples"] == 8



