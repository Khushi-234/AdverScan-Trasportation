"""
Tests for AdverScan StorageRepository and ArtifactManager.
Covers:
- Filesystem artifact offloading (tensors, dicts, text, bytes)
- Assessment creation, updates, and cascading deletion
- Attack, defense, XAI, vulnerability, recommendation, and report persistence
- Historical querying, filtering, and storage diagnostics
"""

from pathlib import Path
import numpy as np
import pytest

from app.storage.storage import ArtifactManager, StorageRepository


@pytest.fixture
def temp_storage_env(tmp_path):
    """Provide isolated paths for SQLite database and artifact directory."""
    db_file = tmp_path / "test_adverscan.db"
    artifacts_dir = tmp_path / "artifacts"
    repo = StorageRepository(db_path=str(db_file), artifact_dir=artifacts_dir)
    return repo, str(db_file), artifacts_dir


# ==============================================================================
# ArtifactManager Tests
# ==============================================================================

def test_artifact_manager_numpy_and_tensor(tmp_path):
    """Verify that numpy arrays and tensor-like objects are saved as compressed .npz and retrievable."""
    mgr = ArtifactManager(base_dir=tmp_path / "artifacts")
    data_arr = np.array([[1.0, 2.5], [3.1, 4.8]], dtype=np.float32)

    rel_path = mgr.save_artifact(
        data=data_arr,
        scan_id="SCAN-001",
        category="attacks",
        filename_prefix="test_tensor",
    )
    assert rel_path is not None
    assert rel_path.endswith(".npz")
    assert "attacks/SCAN-001/test_tensor.npz" in rel_path

    # Verify loading
    loaded_data = mgr.load_artifact(rel_path)
    np.testing.assert_allclose(loaded_data, data_arr)


def test_artifact_manager_json_dict(tmp_path):
    """Verify that dict/list data is stored as JSON and accurately parsed."""
    mgr = ArtifactManager(base_dir=tmp_path / "artifacts")
    payload = {"attack": "FGSM", "epsilon": 0.03, "success": True}

    rel_path = mgr.save_artifact(
        data=payload,
        scan_id="SCAN-002",
        category="xai",
        filename_prefix="xai_metadata",
    )
    assert rel_path is not None
    assert rel_path.endswith(".json")

    loaded = mgr.load_artifact(rel_path)
    assert loaded == payload


def test_artifact_manager_text_and_bytes(tmp_path):
    """Verify text and binary artifact persistence."""
    mgr = ArtifactManager(base_dir=tmp_path / "artifacts")

    # String / Text
    txt_path = mgr.save_artifact(
        data="# Security Report\nAll checks passed.",
        scan_id="SCAN-003",
        category="reports",
        filename_prefix="summary",
    )
    assert txt_path.endswith(".txt")
    loaded_txt = mgr.load_artifact(txt_path)
    assert "All checks passed." in loaded_txt

    # Bytes
    bin_data = b"\x00\x01\x02\x03\x04\xff"
    bin_path = mgr.save_artifact(
        data=bin_data,
        scan_id="SCAN-003",
        category="attacks",
        filename_prefix="raw_data",
    )
    assert bin_path.endswith(".bin")
    loaded_bin = mgr.load_artifact(bin_path)
    assert loaded_bin == bin_data


def test_artifact_manager_null_and_missing(tmp_path):
    """Verify safe handling of null data and non-existent paths."""
    mgr = ArtifactManager(base_dir=tmp_path / "artifacts")
    assert mgr.save_artifact(None, "SCAN-001", "attacks", "none") is None
    assert mgr.load_artifact(None) is None
    assert mgr.load_artifact("non_existent/path.npz") is None


# ==============================================================================
# StorageRepository Tests
# ==============================================================================

def test_create_and_get_assessment(temp_storage_env):
    """Verify assessment record creation with linked model and dataset records."""
    repo, _, _ = temp_storage_env

    model_info = {
        "model_id": "resnet50-cifar10",
        "model_name": "ResNet-50",
        "architecture": "ResNet",
        "framework": "pytorch",
        "modality": "image",
    }
    dataset_info = {
        "dataset_id": "cifar10",
        "dataset_name": "CIFAR-10",
        "modality": "image",
        "number_of_samples": 500,
    }

    scan = repo.create_assessment(
        scan_id="SCAN-TEST01",
        assessment_name="ResNet Robustness Run",
        model_info=model_info,
        dataset_info=dataset_info,
        configuration={"attacks": ["FGSM", "PGD"]},
        modality="image",
    )

    assert scan["scan_id"] == "SCAN-TEST01"
    assert scan["status"] == "running"
    assert scan["model_name"] == "ResNet-50"
    assert scan["dataset_name"] == "CIFAR-10"

    # Fetch from repository
    fetched = repo.get_assessment("SCAN-TEST01")
    assert fetched is not None
    assert fetched["assessment_name"] == "ResNet Robustness Run"
    assert fetched["configuration"] == {"attacks": ["FGSM", "PGD"]}


def test_update_assessment_status(temp_storage_env):
    """Verify status transition from running to completed with metrics."""
    repo, _, _ = temp_storage_env
    repo.create_assessment(scan_id="SCAN-STATUS")

    success = repo.update_assessment_status(
        scan_id="SCAN-STATUS",
        status="completed",
        overall_summary={"mean_asr": 42.5, "vulnerability_index": "High"},
    )
    assert success is True

    fetched = repo.get_assessment("SCAN-STATUS")
    assert fetched["status"] == "completed"
    assert fetched["completed_at"] is not None
    assert fetched["overall_summary"]["mean_asr"] == 42.5

    # Non-existent scan ID returns False
    assert repo.update_assessment_status("SCAN-NONEXISTENT", "completed") is False


def test_record_attack_execution_with_tensor_offload(temp_storage_env):
    """Verify recording an attack execution offloads adversarial samples to filesystem."""
    repo, _, artifacts_dir = temp_storage_env
    repo.create_assessment(scan_id="SCAN-ATK")

    fake_adv_tensor = np.zeros((10, 3, 32, 32), dtype=np.float32)

    atk_res = {
        "attack_name": "PGD",
        "attack_type": "white_box",
        "epsilon_used": 0.031,
        "clean_accuracy": 92.0,
        "adversarial_accuracy": 34.0,
        "accuracy_drop": 58.0,
        "asr_percentage": 63.0,
        "execution_time": 4.52,
        "total_samples": 10,
    }

    recorded = repo.record_attack_execution(
        scan_id="SCAN-ATK",
        attack_result=atk_res,
        attack_params={"steps": 10, "alpha": 0.007},
        adversarial_examples=fake_adv_tensor,
    )

    assert recorded["attack_name"] == "PGD"
    assert recorded["attack_success_rate"] == 63.0
    assert recorded["adversarial_samples_path"] is not None
    assert recorded["adversarial_samples_path"].endswith(".npz")

    # Verify that the artifact actually exists on disk
    full_path = artifacts_dir / recorded["adversarial_samples_path"]
    assert full_path.is_file()

    # Load via ArtifactManager
    loaded_samples = repo.artifacts.load_artifact(recorded["adversarial_samples_path"])
    assert loaded_samples.shape == (10, 3, 32, 32)


def test_record_defense_execution(temp_storage_env):
    """Verify recording defense execution and comparative metrics."""
    repo, _, _ = temp_storage_env
    repo.create_assessment(scan_id="SCAN-DEF")

    def_data = {
        "defense_name": "SpatialSmoothing",
        "defense_family": "preprocessing",
        "before_clean_accuracy": 90.0,
        "after_clean_accuracy": 88.5,
        "before_adv_accuracy": 30.0,
        "after_adv_accuracy": 72.0,
        "before_asr": 66.7,
        "after_asr": 20.0,
        "asr_reduction_percentage": 70.0,
        "accuracy_recovery": 42.0,
        "is_improved": True,
        "robustness_status": "Robustified",
    }

    recorded = repo.record_defense_execution(
        scan_id="SCAN-DEF",
        defense_result=def_data,
    )

    assert recorded["defense_name"] == "SpatialSmoothing"
    assert recorded["asr_reduction"] == 70.0
    assert recorded["is_improved"] is True

    # Retrieve full scan and verify defense appears in children
    scan = repo.get_assessment("SCAN-DEF")
    assert len(scan["defenses"]) == 1
    assert scan["defenses"][0]["defense_name"] == "SpatialSmoothing"


def test_record_xai_result(temp_storage_env):
    """Verify XAI explanation persistence and multi-dimensional attribution storage."""
    repo, _, _ = temp_storage_env
    repo.create_assessment(scan_id="SCAN-XAI")

    attribution_map = np.random.rand(32, 32)

    xai_data = {
        "technique": "gradcam",
        "modality": "image",
        "target_class": 3,
        "clean_prediction": 3,
        "adversarial_prediction": 8,
        "clean_confidence": 0.94,
        "adversarial_confidence": 0.81,
        "prediction_changed": True,
        "local_explanation": {"attribution_drift": 0.42},
        "feature_salience_distortion": 0.65,
        "message": "Salience shifted from object center to background high-frequency noise.",
    }

    rec = repo.record_xai_result(
        scan_id="SCAN-XAI",
        xai_result=xai_data,
        raw_attributions=attribution_map,
    )

    assert rec["explainer_technique"] == "gradcam"
    assert rec["attribution_drift"] == 0.42
    assert rec["attribution_artifact_path"] is not None

    # Load back the saved attribution map
    loaded_map = repo.artifacts.load_artifact(rec["attribution_artifact_path"])
    np.testing.assert_allclose(loaded_map, attribution_map)


def test_record_vulnerabilities_and_recommendations(temp_storage_env):
    """Verify diagnostic findings and hardening recommendations persistence."""
    repo, _, _ = temp_storage_env
    repo.create_assessment(scan_id="SCAN-VULN")

    # Findings
    vulns = {
        "vulnerabilities": [
            {
                "name": "High Frequency Boundary Sensitivity",
                "severity": "Critical",
                "score": 88.5,
                "description": "Model easily misled by sub-pixel perturbations.",
            },
            {
                "name": "Confidence Overcalibration",
                "severity": "Medium",
                "score": 45.0,
                "description": "Erroneous predictions exhibit confidence > 90%.",
            },
        ]
    }
    rec_findings = repo.record_vulnerabilities(scan_id="SCAN-VULN", vuln_result=vulns)
    assert len(rec_findings) == 2
    assert rec_findings[0]["severity"] == "Critical"

    # Recommendations
    recs = [
        {
            "technique": "TotalVariationMinimization",
            "defense_family": "preprocessing",
            "confidence": 0.92,
            "rationale": "Suppresses high frequency noise before model ingestion.",
        }
    ]
    rec_recs = repo.record_recommendations(scan_id="SCAN-VULN", recommendations=recs)
    assert len(rec_recs) == 1
    assert rec_recs[0]["recommended_technique"] == "TotalVariationMinimization"

    # Verify on parent scan
    scan = repo.get_assessment("SCAN-VULN")
    assert len(scan["vulnerabilities"]) == 2
    assert len(scan["recommendations"]) == 1


def test_record_report(temp_storage_env):
    """Verify generated report record linkage."""
    repo, _, _ = temp_storage_env
    repo.create_assessment(scan_id="SCAN-REP")

    rep = repo.record_report(
        scan_id="SCAN-REP",
        report_path="/storage_data/artifacts/reports/SCAN-REP/report.pdf",
        report_name="AdverScan_Executive_Report.pdf",
        report_format="pdf",
        summary_metrics={"vulnerability_index": 72.0},
    )
    assert rep["report_name"] == "AdverScan_Executive_Report.pdf"
    assert rep["report_format"] == "pdf"


def test_list_assessments_filtering(temp_storage_env):
    """Verify querying historical assessments with filters and pagination."""
    repo, _, _ = temp_storage_env

    # Seed 3 distinct assessments
    repo.create_assessment(
        scan_id="SCAN-F1",
        assessment_name="Image Test",
        model_info={"model_name": "ResNet18", "domain": "image"},
        modality="image",
    )
    repo.update_assessment_status("SCAN-F1", "completed")

    repo.create_assessment(
        scan_id="SCAN-F2",
        assessment_name="Text Test",
        model_info={"model_name": "BERT-Base", "domain": "text"},
        modality="text",
    )
    repo.update_assessment_status("SCAN-F2", "completed")

    repo.create_assessment(
        scan_id="SCAN-F3",
        assessment_name="TS Running",
        model_info={"model_name": "PatchTSMixer", "domain": "time-series"},
        modality="time_series",
    )

    # Filter by modality
    img_scans = repo.list_assessments(modality="image")
    assert len(img_scans) == 1
    assert img_scans[0]["scan_id"] == "SCAN-F1"

    # Filter by status
    completed_scans = repo.list_assessments(status="completed")
    assert len(completed_scans) == 2

    # Filter by model name pattern
    bert_scans = repo.list_assessments(model_name="BERT")
    assert len(bert_scans) == 1
    assert bert_scans[0]["scan_id"] == "SCAN-F2"

    # Pagination
    paged = repo.list_assessments(limit=2, offset=0)
    assert len(paged) == 2


def test_delete_assessment_cascading(temp_storage_env):
    """Verify deleting an assessment removes all child attack and defense records."""
    repo, _, _ = temp_storage_env
    repo.create_assessment(scan_id="SCAN-DEL")
    repo.record_attack_execution("SCAN-DEL", {"attack_name": "FGSM", "clean_accuracy": 90.0})
    repo.record_defense_execution("SCAN-DEL", {"defense_name": "GaussianSmoothing"})

    # Confirm present
    assert repo.get_assessment("SCAN-DEL") is not None

    # Delete
    deleted = repo.delete_assessment("SCAN-DEL")
    assert deleted is True

    # Confirm gone
    assert repo.get_assessment("SCAN-DEL") is None

    # Deleting non-existent scan returns False
    assert repo.delete_assessment("SCAN-DEL") is False


def test_get_storage_stats(temp_storage_env):
    """Verify diagnostic stats reporting."""
    repo, _, _ = temp_storage_env
    repo.create_assessment(scan_id="SCAN-STAT1")
    repo.update_assessment_status("SCAN-STAT1", "completed")
    repo.create_assessment(scan_id="SCAN-STAT2")
    repo.update_assessment_status("SCAN-STAT2", "failed")

    stats = repo.get_storage_stats()
    assert stats["database_exists"] is True
    assert stats["total_assessments"] == 2
    assert stats["completed_assessments"] == 1
    assert stats["failed_assessments"] == 1
    assert stats["wal_mode_enabled"] is True

