"""
Tests for AdverScan Frontend Assessment History & Comparison Services.

Covers:
- assessment_history_service: listing, filtering, detail retrieval, deletion, database diagnostics
- assessment_comparison_service: delta calculation, model comparison, attack matching,
  defense comparison, verdict scoring, and modality mismatch handling
- session_state: stage transitions for history and comparison
"""

import pytest

from frontend.services.assessment_history_service import (
    delete_scan,
    get_database_status,
    get_scan_detail,
    get_scan_history,
    get_scans_for_selection,
    get_storage_repository,
)
from frontend.services.assessment_comparison_service import compare_assessments
from frontend.utils.session_state import (
    get_workflow_stage,
    init_session_state,
    reset_workflow_state,
    set_workflow_stage,
    get_active_scan_id,
    set_active_scan_id,
)


@pytest.fixture(autouse=True)
def clean_storage_and_state(tmp_path, monkeypatch):
    """Ensure each test runs with an isolated database and reset session state."""
    db_file = tmp_path / "test_frontend_scans.db"
    artifacts_dir = tmp_path / "artifacts"

    from app.storage.storage import StorageRepository
    test_repo = StorageRepository(db_path=str(db_file), artifact_dir=artifacts_dir)

    # Monkeypatch the singleton in assessment_history_service
    monkeypatch.setattr("frontend.services.assessment_history_service._REPO_INSTANCE", test_repo)

    init_session_state()
    reset_workflow_state()
    yield test_repo


def test_session_state_history_and_comparison_stages():
    """Verify that 'history' and 'comparison' are recognized as valid workflow stages."""
    set_workflow_stage("history")
    assert get_workflow_stage() == "history"

    set_workflow_stage("comparison")
    assert get_workflow_stage() == "comparison"

    set_active_scan_id("SCAN-TEST1234")
    assert get_active_scan_id() == "SCAN-TEST1234"

    reset_workflow_state()
    assert get_workflow_stage() == "dashboard"
    assert get_active_scan_id() is None


def test_assessment_history_service_crud_and_filters(clean_storage_and_state):
    """Verify listing scans, applying filters, fetching details, and deletion."""
    repo = clean_storage_and_state

    # Seed two assessments
    repo.create_assessment(
        scan_id="SCAN-H1",
        assessment_name="ResNet Evaluation",
        model_info={"model_name": "ResNet-50", "domain": "image"},
        modality="image",
    )
    repo.update_assessment_status("SCAN-H1", "completed")

    repo.create_assessment(
        scan_id="SCAN-H2",
        assessment_name="BERT Evaluation",
        model_info={"model_name": "BERT-Base", "domain": "text"},
        modality="text",
    )
    repo.update_assessment_status("SCAN-H2", "failed", error_message="OOM")

    # List all
    all_scans = get_scan_history()
    assert len(all_scans) == 2

    # Filter by modality
    img_scans = get_scan_history(modality="image")
    assert len(img_scans) == 1
    assert img_scans[0]["scan_id"] == "SCAN-H1"

    # Filter by status
    failed_scans = get_scan_history(status="failed")
    assert len(failed_scans) == 1
    assert failed_scans[0]["scan_id"] == "SCAN-H2"

    # Selection format helper
    select_items = get_scans_for_selection()
    assert len(select_items) == 2
    assert any(s["scan_id"] == "SCAN-H1" for s in select_items)

    # Detail lookup
    detail = get_scan_detail("SCAN-H1")
    assert detail is not None
    assert detail["model_name"] == "ResNet-50"

    # Diagnostics
    stats = get_database_status()
    assert stats["total_assessments"] == 2
    assert stats["completed_assessments"] == 1
    assert stats["failed_assessments"] == 1

    # Deletion
    del_ok = delete_scan("SCAN-H2")
    assert del_ok is True
    assert get_scan_detail("SCAN-H2") is None
    assert len(get_scan_history()) == 1


def test_assessment_comparison_service_improved_verdict(clean_storage_and_state):
    """Verify that a comparison scan with lower ASR triggers 'Robustness Improved'."""
    repo = clean_storage_and_state

    # 1. Baseline Scan (High ASR)
    repo.create_assessment(
        scan_id="SCAN-BASE",
        assessment_name="Base Insecure Model",
        model_info={"model_name": "Model-A", "domain": "image"},
        modality="image",
    )
    repo.record_attack_execution(
        scan_id="SCAN-BASE",
        attack_result={
            "attack_name": "PGD",
            "clean_accuracy": 92.0,
            "adversarial_accuracy": 20.0,
            "asr_percentage": 78.0,
            "epsilon_used": 0.03,
        },
    )

    # 2. Defended Comparison Scan (Low ASR)
    repo.create_assessment(
        scan_id="SCAN-DEFENDED",
        assessment_name="Hardened Model",
        model_info={"model_name": "Model-A-Defended", "domain": "image"},
        modality="image",
        parent_scan_id="SCAN-BASE",
    )
    repo.record_attack_execution(
        scan_id="SCAN-DEFENDED",
        attack_result={
            "attack_name": "PGD",
            "clean_accuracy": 90.0,
            "adversarial_accuracy": 70.0,
            "asr_percentage": 22.0,
            "epsilon_used": 0.03,
        },
    )

    # Run comparison
    res = compare_assessments("SCAN-BASE", "SCAN-DEFENDED")
    assert res["success"] is True

    summary = res["summary"]
    assert summary["overall_verdict"] == "Robustness Improved"
    assert summary["verdict_badge"] == "success"
    # delta_asr should be 22.0 - 78.0 = -56.0%
    assert summary["avg_asr_delta"] == -56.0
    # robust accuracy delta: 70.0 - 20.0 = +50.0%
    assert summary["avg_adv_acc_delta"] == 50.0

    # Verify attack comparison breakdown
    atks = res["attack_comparisons"]
    assert len(atks) == 1
    assert atks[0]["attack_name"] == "PGD"
    assert atks[0]["eval_status"] == "improved"
    assert atks[0]["delta_asr"] == -56.0


def test_assessment_comparison_service_degraded_verdict(clean_storage_and_state):
    """Verify that a comparison scan with higher ASR triggers 'Robustness Degraded'."""
    repo = clean_storage_and_state

    repo.create_assessment(
        scan_id="SCAN-SAFE",
        model_info={"model_name": "Robust-V1", "domain": "tabular"},
        modality="tabular",
    )
    repo.record_attack_execution(
        scan_id="SCAN-SAFE",
        attack_result={"attack_name": "HopSkipJump", "clean_accuracy": 85.0, "adversarial_accuracy": 65.0, "asr_percentage": 23.5},
    )

    repo.create_assessment(
        scan_id="SCAN-WEAK",
        model_info={"model_name": "Weak-V2", "domain": "tabular"},
        modality="tabular",
    )
    repo.record_attack_execution(
        scan_id="SCAN-WEAK",
        attack_result={"attack_name": "HopSkipJump", "clean_accuracy": 85.0, "adversarial_accuracy": 15.0, "asr_percentage": 82.0},
    )

    res = compare_assessments("SCAN-SAFE", "SCAN-WEAK")
    assert res["success"] is True
    assert res["summary"]["overall_verdict"] == "Robustness Degraded"
    assert res["summary"]["verdict_badge"] == "danger"
    assert res["attack_comparisons"][0]["eval_status"] == "degraded"


def test_assessment_comparison_mismatched_modality_flag(clean_storage_and_state):
    """Verify modality mismatch detection."""
    repo = clean_storage_and_state

    repo.create_assessment(scan_id="SCAN-IMG", modality="image")
    repo.create_assessment(scan_id="SCAN-TXT", modality="text")

    res = compare_assessments("SCAN-IMG", "SCAN-TXT")
    assert res["success"] is True
    assert res["models"]["modality_match"] is False


def test_assessment_comparison_invalid_scan_ids():
    """Verify graceful handling of non-existent scan IDs."""
    res = compare_assessments("NON-EXISTENT-A", "NON-EXISTENT-B")
    assert res["success"] is False
    assert "not found" in res["error"]

