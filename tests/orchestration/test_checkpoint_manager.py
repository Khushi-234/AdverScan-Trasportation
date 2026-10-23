"""
Unit tests for EvaluationCheckpointManager in Module 8 (Orchestration).

These tests are 100% CPU-only and run in isolation using small mock datasets and temporary files.
"""

import json
import os
from pathlib import Path
import pytest

from app.orchestration.checkpoint_manager import (
    EvaluationCheckpointManager,
    EvaluationCheckpointState,
    CheckpointError,
    CheckpointCorruptedError,
    CheckpointIncompatibleError,
    compute_configuration_hash,
)


@pytest.fixture
def manager(tmp_path):
    """Fixture providing an EvaluationCheckpointManager operating inside a temp directory."""
    return EvaluationCheckpointManager(checkpoint_dir=tmp_path, filename_prefix="test_ckpt")


# 1. Initial Checkpoint Creation
def test_1_create_checkpoint(manager):
    state = manager.create_checkpoint(
        dataset_name="bazyl/GTSRB",
        total_samples=10,
        batch_size=2,
        attacks=["fgsm", "pgd"],
        run_id="test_run_01",
    )
    assert isinstance(state, EvaluationCheckpointState)
    assert state.run_id == "test_run_01"
    assert state.dataset_name == "bazyl/GTSRB"
    assert state.total_samples == 10
    assert state.batch_size == 2
    assert state.attacks == ["fgsm", "pgd"]
    assert len(state.configuration_hash) == 32


# 2. Initial State Values
def test_2_initial_state_values(manager):
    state = manager.create_checkpoint(
        dataset_name="bazyl/GTSRB",
        total_samples=10,
        batch_size=2,
        attacks=["fgsm"],
    )
    assert state.completed_batch_indices == []
    assert state.processed_samples_count == 0
    assert state.last_completed_batch_index is None
    assert state.status == "in_progress"
    assert state.schema_version == 1
    assert isinstance(state.accumulated_results, dict)


# 3. Save Checkpoint
def test_3_save_checkpoint(manager, tmp_path):
    state = manager.create_checkpoint(
        dataset_name="bazyl/GTSRB",
        total_samples=10,
        batch_size=2,
        attacks=["fgsm"],
        run_id="save_run",
    )
    saved_path = manager.save(state)
    assert saved_path.exists()
    assert saved_path.is_file()
    assert saved_path.parent == tmp_path


# 4. Load Checkpoint
def test_4_load_checkpoint(manager):
    state = manager.create_checkpoint(
        dataset_name="bazyl/GTSRB",
        total_samples=10,
        batch_size=2,
        attacks=["fgsm"],
        run_id="load_run",
    )
    saved_path = manager.save(state)

    loaded_state = manager.load(saved_path)
    assert loaded_state.run_id == "load_run"
    assert loaded_state.dataset_name == "bazyl/GTSRB"
    assert loaded_state.total_samples == 10
    assert loaded_state.batch_size == 2


# 5. Save -> Load Round Trip
def test_5_save_load_round_trip(manager):
    state = manager.create_checkpoint(
        dataset_name="bazyl/GTSRB",
        total_samples=10,
        batch_size=2,
        attacks=["fgsm", "deepfool"],
        run_id="round_trip",
        metadata={"experiment": "unit_test"},
    )
    manager.mark_batch_completed(state, batch_index=0, samples_processed=2, batch_results={"acc": 1.0})
    saved_path = manager.save(state)

    loaded_state = manager.load(saved_path)
    assert loaded_state.to_dict() == state.to_dict()


# 6. Idempotent Batch Completion
def test_6_idempotent_batch_completion(manager):
    state = manager.create_checkpoint("dummy", 10, 2, ["fgsm"])
    added_1 = manager.mark_batch_completed(state, batch_index=0, samples_processed=2)
    added_2 = manager.mark_batch_completed(state, batch_index=0, samples_processed=2)

    assert added_1 is True
    assert added_2 is False
    assert state.completed_batch_indices == [0]


# 7. Duplicate Batch Does Not Double-Count Samples
def test_7_duplicate_batch_sample_count(manager):
    state = manager.create_checkpoint("dummy", 10, 2, ["fgsm"])
    manager.mark_batch_completed(state, batch_index=0, samples_processed=2)
    manager.mark_batch_completed(state, batch_index=0, samples_processed=2)

    assert state.processed_samples_count == 2


# 8. Correct Processed Sample Count
def test_8_correct_processed_sample_count(manager):
    state = manager.create_checkpoint("dummy", 10, 2, ["fgsm"])
    manager.mark_batch_completed(state, batch_index=0, samples_processed=2)
    manager.mark_batch_completed(state, batch_index=1, samples_processed=2)
    manager.mark_batch_completed(state, batch_index=2, samples_processed=2)

    assert state.processed_samples_count == 6


# 9. Correct Last Completed Batch Index
def test_9_correct_last_completed_batch(manager):
    state = manager.create_checkpoint("dummy", 10, 2, ["fgsm"])
    manager.mark_batch_completed(state, batch_index=0, samples_processed=2)
    assert state.last_completed_batch_index == 0

    manager.mark_batch_completed(state, batch_index=1, samples_processed=2)
    assert state.last_completed_batch_index == 1


# 10. get_next_incomplete_batch_index()
def test_10_next_incomplete_batch_index(manager):
    state = manager.create_checkpoint("dummy", 10, 2, ["fgsm"])
    assert manager.get_next_incomplete_batch_index(state, total_batches=5) == 0

    manager.mark_batch_completed(state, batch_index=0, samples_processed=2)
    assert manager.get_next_incomplete_batch_index(state, total_batches=5) == 1


# 11. Resume Scenario
def test_11_resume_scenario(manager):
    # Simulate first run: process batches 0, 1, 2 out of 5
    state = manager.create_checkpoint("dummy", 10, 2, ["fgsm"], run_id="resume_run")
    manager.mark_batch_completed(state, 0, 2)
    manager.mark_batch_completed(state, 1, 2)
    manager.mark_batch_completed(state, 2, 2)
    saved_path = manager.save(state)

    # Process interruption & restart: load state
    resumed_state = manager.load(saved_path)
    next_batch = manager.get_next_incomplete_batch_index(resumed_state, total_batches=5)

    assert resumed_state.completed_batch_indices == [0, 1, 2]
    assert next_batch == 3


# 12. Non-Sequential Completed Batches
def test_12_non_sequential_completed_batches(manager):
    state = manager.create_checkpoint("dummy", 10, 2, ["fgsm"])
    manager.mark_batch_completed(state, 0, 2)
    manager.mark_batch_completed(state, 2, 2)  # batch 1 missing

    # Next smallest incomplete batch should be 1
    next_batch = manager.get_next_incomplete_batch_index(state, total_batches=5)
    assert next_batch == 1


# 13. Atomic Write Behavior
def test_13_atomic_write_behavior(manager, tmp_path):
    state = manager.create_checkpoint("dummy", 10, 2, ["fgsm"], run_id="atomic_run")
    target_file = manager.get_checkpoint_path("atomic_run")

    # Ensure no leftover temporary files remain in checkpoint directory after save
    saved_path = manager.save(state)
    assert saved_path == target_file
    
    tmp_files = list(tmp_path.glob("*.tmp_*"))
    assert len(tmp_files) == 0


# 14. Temporary File Cleanup on Save Failure
def test_14_temp_file_cleanup_on_failure(manager, tmp_path, monkeypatch):
    state = manager.create_checkpoint("dummy", 10, 2, ["fgsm"], run_id="fail_run")
    
    # Mock os.replace to raise an OSError to simulate atomic replace failure
    def mock_replace(src, dst):
        raise OSError("Simulated replace disk error")

    monkeypatch.setattr(os, "replace", mock_replace)

    with pytest.raises(CheckpointError, match="Failed to save checkpoint file atomically"):
        manager.save(state)

    # Confirm temporary file was cleaned up
    tmp_files = list(tmp_path.glob("*.tmp_*"))
    assert len(tmp_files) == 0


# 15. Empty Checkpoint File Handling
def test_15_empty_checkpoint_file(manager, tmp_path):
    empty_file = tmp_path / "empty_ckpt.json"
    empty_file.write_text("")

    with pytest.raises(CheckpointCorruptedError, match="Checkpoint file is empty"):
        manager.load(empty_file)


# 16. Malformed JSON Handling
def test_16_malformed_json_handling(manager, tmp_path):
    malformed_file = tmp_path / "bad_ckpt.json"
    malformed_file.write_text("{ incomplete_json: 123 ")

    with pytest.raises(CheckpointCorruptedError, match="malformed JSON"):
        manager.load(malformed_file)


# 17. Missing Required Fields Handling
def test_17_missing_required_fields(manager, tmp_path):
    incomplete_file = tmp_path / "incomplete_ckpt.json"
    incomplete_file.write_text(json.dumps({"run_id": "test"}))

    with pytest.raises(CheckpointCorruptedError, match="missing required schema keys"):
        manager.load(incomplete_file)


# 18. Unsupported Schema Version
def test_18_unsupported_schema_version(manager, tmp_path):
    future_file = tmp_path / "future_ckpt.json"
    future_data = {
        "run_id": "future",
        "dataset_name": "dummy",
        "total_samples": 10,
        "batch_size": 2,
        "attacks": ["fgsm"],
        "completed_batch_indices": [],
        "schema_version": 99,  # Unsupported version
    }
    future_file.write_text(json.dumps(future_data))

    with pytest.raises(CheckpointCorruptedError, match="Unsupported schema version 99"):
        manager.load(future_file)


# 19. Configuration Mismatch Handling
def test_19_configuration_mismatch(manager):
    state = manager.create_checkpoint("dataset_A", 10, 2, ["fgsm"], run_id="config_mismatch")
    saved_path = manager.save(state)

    mismatched_config = {
        "dataset_name": "dataset_B",  # Incompatible dataset!
        "total_samples": 10,
        "batch_size": 2,
        "attacks": ["fgsm"],
    }

    with pytest.raises(CheckpointIncompatibleError, match="configuration hash mismatch"):
        manager.load(saved_path, expected_config=mismatched_config)


# 20. Finalization
def test_20_finalization(manager):
    state = manager.create_checkpoint("dummy", 10, 2, ["fgsm"], run_id="final_run")
    saved_path = manager.finalize(state, status="completed")

    loaded_state = manager.load(saved_path)
    assert loaded_state.status == "completed"
    assert manager.get_next_incomplete_batch_index(loaded_state, total_batches=5) is None


# 21. Cleanup Helper
def test_21_cleanup_helper(manager):
    state = manager.create_checkpoint("dummy", 10, 2, ["fgsm"], run_id="clean_run")
    saved_path = manager.save(state)
    assert saved_path.exists()

    deleted = manager.cleanup(saved_path)
    assert deleted is True
    assert not saved_path.exists()

    # Second cleanup attempt returns False
    assert manager.cleanup(saved_path) is False


# 22. Missing Checkpoint Handling
def test_22_missing_checkpoint_handling(manager, tmp_path):
    missing_file = tmp_path / "non_existent.json"

    with pytest.raises(CheckpointCorruptedError, match="Checkpoint file does not exist"):
        manager.load(missing_file)


# 23. Deterministic Interrupted Multi-Batch Run Resumption Matching Uninterrupted Run
def test_23_orchestrator_interrupted_run_resumes_matching_uninterrupted_run(tmp_path):
    """
    Sprint 1.6 Integration Test:
    Interrupt a small multi-batch run, resume it from the saved checkpoint,
    and verify that the final results and predictions match the uninterrupted run exactly.
    """
    import torch
    import torch.nn as nn
    from app.orchestration.orchestrator import AdverScanOrchestrator
    from app.orchestration.pipeline_config import PipelineConfig
    from app.orchestration.dataset_adapter import InMemoryDatasetLoader

    # Fixed deterministic weights and inputs
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(10, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
    )
    # Freeze model weights for determinism
    for p in model.parameters():
        p.requires_grad = False

    inputs = torch.randn(8, 10)
    targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.long)

    loader_1 = InMemoryDatasetLoader(inputs=inputs.clone(), targets=targets.clone(), batch_size=4)
    loader_2 = InMemoryDatasetLoader(inputs=inputs.clone(), targets=targets.clone(), batch_size=4)
    loader_3 = InMemoryDatasetLoader(inputs=inputs.clone(), targets=targets.clone(), batch_size=4)

    # 1. Uninterrupted run
    uninterrupted_dir = tmp_path / "uninterrupted"
    cfg_uninterrupted = PipelineConfig(
        model_path=model,
        sample_input=inputs[:1],
        num_classes=3,
        mode="attack_assessment",
        attacks=["fgsm"],
        batch_size=4,
        enable_xai=False,
        enable_hardening=False,
        enable_report=False,
        enable_checkpoint=True,
        checkpoint_dir=str(uninterrupted_dir),
        custom_dataset_loader=loader_1,
        seed=42,
        deterministic=True,
    )
    orch_1 = AdverScanOrchestrator()
    res_uninterrupted = orch_1.run(cfg_uninterrupted)
    assert res_uninterrupted.status == "SUCCESS"

    # 2. Simulated interrupted run (interrupted after batch 0 completes)
    resumed_dir = tmp_path / "resumed"
    resumed_dir.mkdir(parents=True, exist_ok=True)
    cfg_interrupted = PipelineConfig(
        model_path=model,
        sample_input=inputs[:1],
        num_classes=3,
        mode="attack_assessment",
        attacks=["fgsm"],
        batch_size=4,
        enable_xai=False,
        enable_hardening=False,
        enable_report=False,
        enable_checkpoint=True,
        checkpoint_dir=str(resumed_dir),
        custom_dataset_loader=loader_2,
        seed=42,
        deterministic=True,
    )

    from app.attack_engine.attack_engine import AttackEngine
    orig_run_attack = AttackEngine.run_attack
    call_count = 0

    def fail_after_first_batch(self, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        res = orig_run_attack(self, *args, **kwargs)
        if call_count >= 2:
            raise RuntimeError("Simulated interruption during batch 1!")
        return res

    AttackEngine.run_attack = fail_after_first_batch
    try:
        orch_2 = AdverScanOrchestrator()
        orch_2.run(cfg_interrupted)
    except Exception:
        pass
    finally:
        AttackEngine.run_attack = orig_run_attack

    # Verify that a checkpoint file was created containing batch 0 completed
    ckpt_files = list(resumed_dir.glob("*.json"))
    assert len(ckpt_files) == 1, "Expected one checkpoint file from interrupted run"
    ckpt_file = ckpt_files[0]

    with open(ckpt_file, "r") as f:
        saved_state = json.load(f)
    assert 0 in saved_state["completed_batch_indices"]
    assert 1 not in saved_state["completed_batch_indices"]

    # 3. Resume the run from the saved checkpoint
    cfg_resume = PipelineConfig(
        model_path=model,
        sample_input=inputs[:1],
        num_classes=3,
        mode="attack_assessment",
        attacks=["fgsm"],
        batch_size=4,
        enable_xai=False,
        enable_hardening=False,
        enable_report=False,
        enable_checkpoint=True,
        checkpoint_dir=str(resumed_dir),
        resume_checkpoint_path=str(ckpt_file),
        custom_dataset_loader=loader_3,
        seed=42,
        deterministic=True,
    )
    orch_3 = AdverScanOrchestrator()
    res_resumed = orch_3.run(cfg_resume)
    assert res_resumed.status == "SUCCESS"

    # 4. Verify numerical and metric parity between uninterrupted and resumed runs
    assert res_resumed.metadata["checkpoint"]["completed_batch_indices"] == [0, 1]
    unint_acc = res_uninterrupted.adversarial_evaluations["fgsm"]["accuracy"]
    resum_acc = res_resumed.adversarial_evaluations["fgsm"]["accuracy"]
    assert unint_acc == resum_acc, f"Accuracy mismatch: {unint_acc} vs {resum_acc}"
    assert res_uninterrupted.adversarial_evaluations["fgsm"]["num_samples"] == res_resumed.adversarial_evaluations["fgsm"]["num_samples"]
    assert res_uninterrupted.adversarial_evaluations["fgsm"]["average_confidence"] == res_resumed.adversarial_evaluations["fgsm"]["average_confidence"]
    assert res_uninterrupted.adversarial_evaluations["fgsm"]["average_entropy"] == res_resumed.adversarial_evaluations["fgsm"]["average_entropy"]

