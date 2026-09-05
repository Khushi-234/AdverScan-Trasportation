"""
Unit tests for A3 — MLflow Experiment Tracking Integration in AdverScan.
Verifies failure isolation, experiment lifecycle management, parameter/metric/tag logging,
artifact persistence, and Orchestrator integration.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import pytest

from app.orchestration.orchestration_result import OrchestrationResult
from app.orchestration.orchestrator import AdverScanOrchestrator
from app.orchestration.pipeline_config import PipelineConfig
from app.utils.mlflow_tracker import MLflowTracker, is_mlflow_available


def test_1_is_mlflow_available():
    """Verify that is_mlflow_available returns a boolean."""
    avail = is_mlflow_available()
    assert isinstance(avail, bool)


def test_2_pipeline_config_mlflow_defaults():
    """Verify enable_mlflow is False by default for backward compatibility."""
    cfg = PipelineConfig()
    assert cfg.enable_mlflow is False
    assert cfg.mlflow_tracking_uri is None
    assert cfg.mlflow_experiment_name is None


def test_3_orchestration_result_mlflow_defaults():
    """Verify mlflow fields on OrchestrationResult default to None."""
    res = OrchestrationResult(status="SUCCESS", execution_mode="full")
    assert res.mlflow_run_id is None
    assert res.mlflow_experiment_name is None


def test_4_mlflow_tracker_start_and_end_run():
    """Verify MLflowTracker start_run and end_run lifecycle."""
    if not is_mlflow_available():
        pytest.skip("MLflow not installed")

    with tempfile.TemporaryDirectory() as tmpdir:
        tracking_uri = f"file:{tmpdir}"
        tracker = MLflowTracker()

        run_id = tracker.start_run(
            experiment_name="TestExperiment",
            tracking_uri=tracking_uri,
            run_name="test_run",
        )

        assert run_id is not None
        assert tracker.active_run_id == run_id
        assert tracker.experiment_name == "TestExperiment"

        ended = tracker.end_run()
        assert ended is True
        assert tracker.active_run_id is None


def test_5_log_orchestration_run_data():
    """Verify parameter, tag, metric, and artifact logging."""
    if not is_mlflow_available():
        pytest.skip("MLflow not installed")

    with tempfile.TemporaryDirectory() as tmpdir:
        tracking_uri = f"file:{tmpdir}"
        tracker = MLflowTracker()
        run_id = tracker.start_run(
            experiment_name="TestLoggingExp",
            tracking_uri=tracking_uri,
            run_name="log_run",
        )
        assert run_id is not None

        cfg = PipelineConfig(
            experiment_id="EXP-101",
            experiment_name="TestLoggingExp",
            model_name="TestModel",
            attacks=["fgsm"],
            enable_mlflow=True,
        )

        res = OrchestrationResult(
            status="SUCCESS",
            execution_mode="full",
            execution_time_seconds=1.5,
            baseline_evaluation={"accuracy": 0.95, "f1_macro": 0.94},
            reproducibility_metadata={"seed": 42, "python_version": "3.12"},
            resource_summary={"cpu_percent": 12.5, "ram_used_gb": 4.2},
            failure_records=[],
            mlflow_run_id=run_id,
            mlflow_experiment_name="TestLoggingExp",
        )

        success = tracker.log_orchestration_run(result=res, config=cfg, output_dir=tmpdir)
        assert success is True

        ended = tracker.end_run()
        assert ended is True


def test_6_failure_safe_tracking():
    """Verify that tracking errors do not crash caller."""
    tracker = MLflowTracker()

    # Calling log_orchestration_run without active run return False gracefully
    res = OrchestrationResult(status="SUCCESS", execution_mode="full")
    cfg = PipelineConfig()
    logged = tracker.log_orchestration_run(result=res, config=cfg)
    assert logged is False

    # Calling end_run without active run returns False gracefully
    ended = tracker.end_run()
    assert ended is False


def test_7_orchestrator_integration_with_mlflow_enabled():
    """Verify full orchestrator execution with MLflow tracking enabled."""
    if not is_mlflow_available():
        pytest.skip("MLflow not installed")

    with tempfile.TemporaryDirectory() as tmpdir:
        import torch
        from tests.orchestration.test_orchestrator import DummyModel, DummyDatasetLoader

        model_path = os.path.join(tmpdir, "mock_model.pt")
        sample_input = torch.randn(1, 10)
        torch.save(DummyModel(), model_path)

        cfg = PipelineConfig(
            model_path=model_path,
            sample_input=sample_input,
            model_class=DummyModel,
            custom_dataset_loader=DummyDatasetLoader(),
            mode="baseline_only",
            output_dir=os.path.join(tmpdir, "results"),
            enable_mlflow=True,
            mlflow_tracking_uri=f"file:{tmpdir}/mlruns",
            mlflow_experiment_name="OrchestratorTestExp",
        )

        orchestrator = AdverScanOrchestrator()
        result = orchestrator.run(cfg)

        assert result.status == "SUCCESS"
        assert result.mlflow_run_id is not None
        assert result.mlflow_experiment_name == "OrchestratorTestExp"


def test_8_orchestrator_integration_default_mlflow_disabled():
    """Verify orchestrator execution when MLflow tracking is disabled by default."""
    with tempfile.TemporaryDirectory() as tmpdir:
        import torch
        from tests.orchestration.test_orchestrator import DummyModel, DummyDatasetLoader

        model_path = os.path.join(tmpdir, "mock_model.pt")
        sample_input = torch.randn(1, 10)
        torch.save(DummyModel(), model_path)

        cfg = PipelineConfig(
            model_path=model_path,
            sample_input=sample_input,
            model_class=DummyModel,
            custom_dataset_loader=DummyDatasetLoader(),
            mode="baseline_only",
            output_dir=os.path.join(tmpdir, "results"),
            enable_mlflow=False,
        )

        orchestrator = AdverScanOrchestrator()
        result = orchestrator.run(cfg)

        assert result.status == "SUCCESS"
        assert result.mlflow_run_id is None
