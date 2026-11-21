"""
Tests for AdverScan SQLAlchemy database models and relational integrity.
"""

from datetime import datetime
import pytest

from app.storage.db import get_db_session, init_db
from app.storage.models import (
    AssessmentRecord,
    AttackExecutionRecord,
    DatasetRecord,
    DefenseExecutionRecord,
    DefenseRecommendationRecord,
    ModelRecord,
    ReportRecord,
    VulnerabilityFindingRecord,
    XAIResultRecord,
)


@pytest.fixture
def test_db(tmp_path):
    """Initialize an isolated database for model tests."""
    db_file = tmp_path / "models_test.db"
    init_db(str(db_file))
    return str(db_file)


def test_model_and_dataset_relationships(test_db):
    """Verify ModelRecord, DatasetRecord, and their foreign key association with AssessmentRecord."""
    with get_db_session(test_db) as session:
        m = ModelRecord(
            model_id="vit-base-cifar10",
            model_name="ViT Base CIFAR-10",
            architecture="VisionTransformer",
            framework="pytorch",
            modality="image",
            total_parameters=86000000,
        )
        d = DatasetRecord(
            dataset_id="cifar10-test",
            dataset_name="CIFAR-10 Test Split",
            modality="image",
            number_of_samples=10000,
            number_of_classes=10,
        )
        session.add(m)
        session.add(d)
        session.flush()

        scan = AssessmentRecord(
            scan_id="SCAN-M1-TEST",
            assessment_name="Model & Dataset Link Test",
            model_id=m.model_id,
            dataset_id=d.dataset_id,
            status="completed",
        )
        session.add(scan)

    with get_db_session(test_db) as session:
        retrieved_scan = session.query(AssessmentRecord).filter_by(scan_id="SCAN-M1-TEST").first()
        assert retrieved_scan is not None
        assert retrieved_scan.model is not None
        assert retrieved_scan.model.model_name == "ViT Base CIFAR-10"
        assert retrieved_scan.dataset is not None
        assert retrieved_scan.dataset.dataset_name == "CIFAR-10 Test Split"
        assert len(retrieved_scan.model.assessments) == 1


def test_cascading_deletes(test_db):
    """Verify that deleting an assessment cleanly cascades to attacks, defenses, XAI, findings, and reports."""
    with get_db_session(test_db) as session:
        scan = AssessmentRecord(scan_id="SCAN-CASCADE-TEST", status="completed")
        session.add(scan)
        session.flush()

        atk = AttackExecutionRecord(
            attack_run_id="ATK-001",
            scan_id=scan.scan_id,
            attack_name="PGD",
            clean_accuracy=95.0,
            adversarial_accuracy=10.0,
            attack_success_rate=85.0,
        )
        dfn = DefenseExecutionRecord(
            defense_run_id="DEF-001",
            scan_id=scan.scan_id,
            defense_name="GaussianFilter",
            asr_before=85.0,
            asr_after=5.0,
            asr_reduction=80.0,
        )
        xai = XAIResultRecord(
            xai_id="XAI-001",
            scan_id=scan.scan_id,
            explainer_technique="saliency",
            target_class=1,
        )
        vuln = VulnerabilityFindingRecord(
            finding_id="VULN-001",
            scan_id=scan.scan_id,
            vulnerability_name="High Gradient Sensitivity",
            severity="High",
            vulnerability_score=85.0,
        )
        rec = DefenseRecommendationRecord(
            recommendation_id="REC-001",
            scan_id=scan.scan_id,
            recommended_technique="spatial_smoothing",
        )
        rep = ReportRecord(
            report_id="REP-001",
            scan_id=scan.scan_id,
            report_name="Security Report",
            report_path="/path/to/report.pdf",
            report_format="pdf",
        )
        session.add_all([atk, dfn, xai, vuln, rec, rep])

    # Verify all records exist
    with get_db_session(test_db) as session:
        assert session.query(AttackExecutionRecord).filter_by(attack_run_id="ATK-001").count() == 1
        assert session.query(DefenseExecutionRecord).filter_by(defense_run_id="DEF-001").count() == 1
        assert session.query(XAIResultRecord).filter_by(xai_id="XAI-001").count() == 1
        assert session.query(VulnerabilityFindingRecord).filter_by(finding_id="VULN-001").count() == 1
        assert session.query(DefenseRecommendationRecord).filter_by(recommendation_id="REC-001").count() == 1
        assert session.query(ReportRecord).filter_by(report_id="REP-001").count() == 1

        # Delete parent scan
        scan = session.query(AssessmentRecord).filter_by(scan_id="SCAN-CASCADE-TEST").first()
        session.delete(scan)

    # Verify cascading delete purged child records
    with get_db_session(test_db) as session:
        assert session.query(AttackExecutionRecord).filter_by(attack_run_id="ATK-001").count() == 0
        assert session.query(DefenseExecutionRecord).filter_by(defense_run_id="DEF-001").count() == 0
        assert session.query(XAIResultRecord).filter_by(xai_id="XAI-001").count() == 0
        assert session.query(VulnerabilityFindingRecord).filter_by(finding_id="VULN-001").count() == 0
        assert session.query(DefenseRecommendationRecord).filter_by(recommendation_id="REC-001").count() == 0
        assert session.query(ReportRecord).filter_by(report_id="REP-001").count() == 0


def test_model_to_dict_serialization(test_db):
    """Verify that all models implement to_dict() and serialize dates/JSON cleanly."""
    with get_db_session(test_db) as session:
        scan = AssessmentRecord(
            scan_id="SCAN-DICT-TEST",
            assessment_name="Dictionary Test",
            status="completed",
            configuration={"epsilon": 0.3, "batch_size": 32},
            overall_summary={"risk": "High"},
        )
        session.add(scan)
        session.flush()

        d = scan.to_dict()
        assert isinstance(d, dict)
        assert d["scan_id"] == "SCAN-DICT-TEST"
        assert d["configuration"]["epsilon"] == 0.3
        assert d["overall_summary"]["risk"] == "High"
        assert isinstance(d["created_at"], str)
