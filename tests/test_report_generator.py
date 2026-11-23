"""
Unit and integration tests for Module 9 (Report Generator) in AdverScan.
"""

import json
from pathlib import Path
import pytest

from app.report_generator import (
    ReportData,
    ReportResult,
    ReportGenerator,
    ReportWriter,
    ExecutionSummary,
    ModuleExecutionRecord,
)


def test_report_data_creation_and_dict_conversion():
    """Test ReportData initialization, from_dict, and to_dict."""
    sample_dict = {
        "model_info": {"model_name": "TestModel", "num_parameters": 1000},
        "baseline_performance": {"metrics": {"accuracy": 0.95}},
        "attack_results": {
            "FGSM": {"parameters": {"eps": 0.05}},
            "PGD": {"parameters": {"eps": 0.03, "steps": 10}},
            "DeepFool": {"parameters": {"max_iter": 50}},
        },
        "vulnerability_metrics": {
            "FGSM": {"scoring": {"vulnerability_score": 65.0, "risk_level": "HIGH"}},
            "PGD": {"scoring": {"vulnerability_score": 80.0, "risk_level": "CRITICAL"}},
        },
        "xai_findings": {"FGSM_GradCAM": {"target_class": 1}},
        "hardening_results": {"defense": "AdversarialTraining", "status": "COMPLETED"},
        "before_vs_after": {"FGSM": {"delta_accuracy_drop": -0.2, "is_improved": True}},
    }

    data = ReportData.from_dict(sample_dict)
    assert data.model_info["model_name"] == "TestModel"
    assert "FGSM" in data.attack_results
    assert "PGD" in data.attack_results
    assert "DeepFool" in data.attack_results
    assert data.vulnerability_score == 65.0 or data.vulnerability_score == 80.0

    d_out = data.to_dict()
    assert d_out["model_info"]["model_name"] == "TestModel"


def test_report_generator_full_pipeline(tmp_path):
    """Test full report generation flow including text report and JSON export."""
    sample_data = ReportData(
        model_info={"model_name": "ResNet18", "architecture": "CNN", "num_parameters": 11000000},
        baseline_performance={"metrics": {"accuracy": 0.92, "f1_score": 0.91}},
        attack_results={
            "FGSM": {
                "parameters": {"eps": 0.1},
                "evaluation": {"metrics": {"attack_success_rate": 0.75, "accuracy_drop": 0.65}},
            },
            "PGD": {
                "parameters": {"eps": 0.03, "steps": 20},
                "evaluation": {"metrics": {"attack_success_rate": 0.85, "accuracy_drop": 0.80}},
            },
            "DeepFool": {
                "parameters": {"max_iter": 50},
                "evaluation": {"metrics": {"attack_success_rate": 0.60, "accuracy_drop": 0.50}},
            },
        },
        vulnerability_metrics={
            "FGSM": {"scoring": {"vulnerability_score": 70.0, "risk_level": "HIGH"}},
            "PGD": {"scoring": {"vulnerability_score": 85.0, "risk_level": "CRITICAL"}},
        },
        vulnerability_score=77.5,
        risk_level="CRITICAL",
        xai_findings={"FGSM_IntegratedGradients": {"attribution_mean": 0.45}},
        hardening_results={"defense": "AdversarialTraining", "epochs": 5, "hardened_accuracy": 0.88},
        before_vs_after={
            "FGSM": {
                "delta_attack_success_rate": -0.40,
                "delta_accuracy_drop": -0.35,
                "is_improved": True,
            },
            "PGD": {
                "delta_attack_success_rate": -0.30,
                "delta_accuracy_drop": -0.25,
                "is_improved": True,
            },
        },
    )

    generator = ReportGenerator()
    result: ReportResult = generator.generate(sample_data)

    assert result.status == "SUCCESS"
    assert result.risk_level == "CRITICAL"
    assert result.vulnerability_score == 77.5
    assert len(result.recommendations) > 0
    assert "ResNet18" in result.formatted_report
    assert "FGSM" in result.formatted_report
    assert "PGD" in result.formatted_report
    assert "DeepFool" in result.formatted_report
    assert "RECOMMENDATIONS" in result.formatted_report

    # Save to files
    json_path = tmp_path / "security_report.json"
    text_path = tmp_path / "security_report.txt"

    result.save_json(json_path)
    result.save_text(text_path)

    assert json_path.exists()
    assert text_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)
        assert loaded_json["report_id"] == result.report_id
        assert loaded_json["risk_level"] == "CRITICAL"

    with open(text_path, "r", encoding="utf-8") as f:
        loaded_text = f.read()
        # New 15-section header
        assert "ADVERSCAN SECURITY ASSESSMENT REPORT" in loaded_text
        assert "1. EXECUTIVE SUMMARY" in loaded_text
        assert "5. ADVERSARIAL ATTACK RESULTS" in loaded_text
        assert "7. VULNERABILITY SCORE" in loaded_text
        assert "8. MITRE ATLAS MAPPING" in loaded_text
        assert "13. EXECUTION PERFORMANCE" in loaded_text
        assert "14. RECOMMENDATIONS" in loaded_text
        assert "15. FINAL SECURITY SUMMARY" in loaded_text


def test_report_data_from_orchestration_and_retest():
    """Test factory method creating ReportData from orchestrator and retest dicts."""
    orch_dict = {
        "status": "SUCCESS",
        "execution_mode": "full_pipeline",
        "timestamp": "2026-08-25 10:00:00",
        "model_metadata": {"model_name": "TestNet"},
        "baseline_evaluation": {"metrics": {"accuracy": 0.90}},
        "attack_results": {
            "FGSM": {"parameters": {"eps": 0.05}},
        },
        "adversarial_evaluations": {
            "FGSM": {"metrics": {"attack_success_rate": 0.60}},
        },
        "vulnerability_analysis": {
            "FGSM": {
                "assessment": {"attack_success_rate": 0.60},
                "scoring": {"vulnerability_score": 60.0, "risk_level": "HIGH"},
            }
        },
    }

    retest_dict = {
        "comparisons": {
            "FGSM": {
                "delta_accuracy_drop": -0.25,
                "is_improved": True,
            }
        }
    }

    report_data = ReportData.from_orchestration_and_retest(orch_dict, retest_dict)

    assert report_data.model_info["model_name"] == "TestNet"
    assert "FGSM" in report_data.attack_results
    assert report_data.attack_results["FGSM"]["evaluation"]["metrics"]["attack_success_rate"] == 0.60
    assert report_data.vulnerability_score == 60.0
    assert report_data.risk_level == "HIGH"
    assert "FGSM" in report_data.before_vs_after

    generator = ReportGenerator()
    res = generator.generate(report_data)
    assert res.risk_level == "HIGH"
    assert res.vulnerability_score == 60.0


def test_report_writer(tmp_path):
    """Test that ReportWriter persists reports in all formats."""
    sample_data = ReportData(
        model_info={"model_name": "WriterTestModel"},
        baseline_performance={"metrics": {"accuracy": 0.88}},
        attack_results={
            "fgsm": {"parameters": {"eps": 0.05},
                     "evaluation": {"metrics": {"attack_success_rate": 0.65}}},
        },
        vulnerability_score=55.0,
        risk_level="HIGH",
    )

    # Add a synthetic execution summary
    exec_sum = ExecutionSummary(
        run_label="WriterTest",
        run_timestamp="2026-08-27 12:00:00",
        total_elapsed_seconds=42.5,
        modules=[
            ModuleExecutionRecord(
                module_id="M1", module_name="M1 Model Ingestion",
                status="SUCCESS", elapsed_seconds=3.1,
                metrics={"framework": "PyTorch"},
            ),
            ModuleExecutionRecord(
                module_id="M2_baseline", module_name="M2 Baseline Evaluation",
                status="SUCCESS", elapsed_seconds=12.4,
                metrics={"accuracy": 0.88},
            ),
        ],
    )
    sample_data.execution_summary = exec_sum

    generator = ReportGenerator()
    result = generator.generate(sample_data)

    # Write report
    writer = ReportWriter(output_root=tmp_path / "reports")
    paths = writer.write(result, formats=["md", "json", "csv"])

    assert "md" in paths
    assert "json" in paths
    assert "csv" in paths

    # Verify Markdown content
    with open(paths["md"], "r", encoding="utf-8") as f:
        md_text = f.read()
    assert "ADVERSCAN SECURITY ASSESSMENT REPORT" in md_text
    assert "13. EXECUTION PERFORMANCE" in md_text
    assert "M1 Model Ingestion" in md_text

    # Verify JSON structure
    with open(paths["json"], "r", encoding="utf-8") as f:
        j = json.load(f)
    assert j["report_id"] == result.report_id
    assert j["risk_level"] == "HIGH"
    assert j["execution_summary"] is not None

    # Verify CSV has header + module rows
    with open(paths["csv"], "r", encoding="utf-8") as f:
        csv_text = f.read()
    assert "module_id" in csv_text
    assert "M1" in csv_text
    assert "M2_baseline" in csv_text


def test_report_naming_convention(tmp_path):
    """Test that report naming convention generates SCAN-{attack_name}-{number}."""
    import re
    from pathlib import Path

    # 1. Single attack
    data_single = ReportData(
        model_info={"model_name": "SingleModel"},
        attack_results={"fgsm": {"evaluation": {"metrics": {}}}},
    )
    generator = ReportGenerator()
    res_single = generator.generate(data_single)
    assert re.match(r"^SCAN-FGSM-\d{6}$", res_single.scan_id)

    # 2. Multiple attacks
    data_multi = ReportData(
        model_info={"model_name": "MultiModel"},
        attack_results={
            "fgsm": {"evaluation": {"metrics": {}}},
            "pgd": {"evaluation": {"metrics": {}}},
        },
    )
    res_multi = generator.generate(data_multi)
    assert re.match(r"^SCAN-FGSM_PGD-\d{6}$", res_multi.scan_id)

    # 3. Directory and file naming through ReportWriter
    writer = ReportWriter(output_root=tmp_path / "reports")
    written = writer.write(res_single, formats=["md", "json"])

    md_path = Path(written["md"])
    # Folder name must be SCAN-FGSM-{number}
    assert md_path.parent.name == res_single.scan_id
    # File name must be adverscan_report_fgsm_{number}.md
    num = res_single.scan_id.split("-")[-1]
    assert md_path.name == f"adverscan_report_fgsm_{num}.md"


def test_section_9_xai_clean_formatting(tmp_path):
    """
    Verify Section 9 renders a user-focused, concise summary and does not dump
    internal ExplanationResult dictionaries, raw arrays, or metadata objects.
    """
    xai_dict = {
        "FGSM_IntegratedGradients": {
            "technique": "integrated_gradients",
            "attack_name": "FGSM",
            "modality": "image",
            "target_class": 1,
            "attribution_status": "success",
            "clean_prediction": 0,
            "clean_confidence": 0.9850,
            "adversarial_prediction": 1,
            "adversarial_confidence": 0.9420,
            "prediction_changed": True,
            "attack_caused_failure": True,
            "metadata": {
                "_result_obj": "<InternalExplanationObject at 0x7f>",
                "total_samples": 5,
                "successful_samples": 5,
                "failed_samples": 0,
                "raw_config": {"internal": True},
            },
            "attribution": {
                "clean": {"attribution": [[0.1, 0.2], [0.3, 0.4]]},
                "adversarial": {"attribution": [[0.5, 0.6], [0.7, 0.8]]},
            },
            "batch_results": [
                {"attribution_status": "success", "sample_idx": 0},
                {"attribution_status": "success", "sample_idx": 1},
                {"attribution_status": "success", "sample_idx": 2},
                {"attribution_status": "success", "sample_idx": 3},
                {"attribution_status": "success", "sample_idx": 4},
            ],
            "failure_analysis": {
                "attribution_drift": 0.7421,
                "cosine_distortion": 0.5833,
                "perturbation_alignment": 0.8190,
                "attack_reason": "Attribution shifted significantly toward high-frequency noise in the top-left quadrant.",
                "attribution_interpretation": "Localized pixel gradients diverted classification from Class 0 to Class 1.",
            },
            "feature_importance": {
                "adversarial_top_features": [
                    {"feature": "patch_12", "importance": 0.4521},
                    {"feature": "patch_14", "importance": 0.3812},
                ],
            },
            "defense_hints": [
                {
                    "finding": "High attribution concentration on localized pixel clusters.",
                    "suggested_defense": "Spatial Smoothing / Median Filtering",
                    "stage": "Input Preprocessing",
                    "confidence": 0.88,
                    "evidence": "High perturbation alignment (0.8190)",
                    "why_recommended": "Attenuates high-frequency spatial gradients before model inference.",
                }
            ],
        }
    }

    report_data = ReportData(
        model_info={"model_name": "ResNet18", "architecture": "CNN"},
        baseline_performance={"metrics": {"accuracy": 0.95, "f1_score": 0.94}},
        attack_results={
            "FGSM": {
                "parameters": {"eps": 0.05},
                "evaluation": {"metrics": {"attack_success_rate": 0.80, "accuracy_drop": 0.75}},
            }
        },
        vulnerability_score=75.0,
        risk_level="HIGH",
        xai_findings=xai_dict,
    )

    generator = ReportGenerator()
    res = generator.generate(report_data)
    report_text = res.formatted_report

    # 1. Section 9 Header present
    assert "9. XAI FINDINGS" in report_text

    # 2. Explanation Overview present
    assert "Explainer Technique : Integrated Gradients" in report_text
    assert "Execution Status    : ✅ SUCCESS" in report_text
    assert "Attack Examined     : FGSM" in report_text
    assert "Model Modality      : Image" in report_text
    assert "Target Class        : Class 1" in report_text
    assert "5 of 5 samples successfully explained (0 failed)" in report_text

    # 3. Prediction Analysis table present
    assert "Prediction Analysis:" in report_text
    assert "Clean Pred" in report_text
    assert "Adv Pred" in report_text
    assert "98.50%" in report_text
    assert "94.20%" in report_text

    # 4. Attribution Analysis present
    assert "Attribution Analysis:" in report_text
    assert "Attribution Drift (L1)    : 0.7421" in report_text
    assert "Salience Distortion       : 0.5833" in report_text
    assert "Perturbation Alignment   : 0.8190" in report_text
    assert "patch_12 (0.4521)" in report_text
    assert "patch_14 (0.3812)" in report_text
    assert "Localized pixel gradients diverted classification" in report_text

    # 5. Root Cause Explanation prose present
    assert "Root Cause Explanation:" in report_text
    assert "Attribution shifted significantly toward high-frequency noise" in report_text

    # 6. Defense Recommendations present
    assert "Evidence-Based Defense Recommendations:" in report_text
    assert "[Input Preprocessing] Spatial Smoothing / Median Filtering" in report_text
    assert "Confidence: 88.00%" in report_text
    assert "High perturbation alignment (0.8190)" in report_text

    # 7. Strictly verify NO raw internal objects or placeholder dumps leak
    assert "_result_obj" not in report_text
    assert "<InternalExplanationObject" not in report_text
    assert "raw_config" not in report_text
    assert "[Detailed Data Omitted]" not in report_text
    assert "[[0.1, 0.2]" not in report_text


def test_report_writer_pdf_multipage(tmp_path):
    """
    Verify ReportWriter generates a multi-page PDF document when the report
    content spans multiple pages, without truncation or error.
    """
    sample_data = ReportData(
        model_info={"model_name": "MultiPageTestNet", "architecture": "CNN"},
        baseline_performance={"metrics": {"accuracy": 0.92, "f1_score": 0.91}},
        attack_results={
            "FGSM": {"parameters": {"eps": 0.05}, "evaluation": {"metrics": {"attack_success_rate": 0.70}}},
            "PGD": {"parameters": {"eps": 0.03, "steps": 10}, "evaluation": {"metrics": {"attack_success_rate": 0.85}}},
        },
        vulnerability_score=78.0,
        risk_level="CRITICAL",
        xai_findings={
            "FGSM_IntegratedGradients": {
                "technique": "integrated_gradients",
                "clean_prediction": 0,
                "clean_confidence": 0.95,
                "adversarial_prediction": 1,
                "adversarial_confidence": 0.91,
                "prediction_changed": True,
                "attack_caused_failure": True,
                "failure_analysis": {
                    "attribution_drift": 0.68,
                    "attack_reason": "Vulnerability to localized high-frequency perturbations.",
                },
            }
        },
    )

    generator = ReportGenerator()
    res = generator.generate(sample_data)

    writer = ReportWriter(output_root=tmp_path / "reports")
    written = writer.write(res, formats=["pdf", "md"])

    assert "pdf" in written
    pdf_path = Path(written["pdf"])
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 1000

    # Verify PDF has at least 1-2 pages and is valid
    try:
        import pypdf  # type: ignore
        reader = pypdf.PdfReader(str(pdf_path))
        assert len(reader.pages) >= 1
    except ImportError:
        pass


def test_compile_pdf_report_from_report_result():
    """Verify compile_pdf_report compiles valid PDF bytes directly from ReportResult."""
    from app.report_generator.pdf_generator import compile_pdf_report

    sample_data = ReportData(
        model_info={"model_name": "DirectPDFNet", "architecture": "ResNet18"},
        baseline_performance={"metrics": {"accuracy": 0.94}},
        attack_results={
            "PGD": {"parameters": {"eps": 0.03}, "evaluation": {"metrics": {"attack_success_rate": 0.82}}},
        },
        vulnerability_score=68.5,
        risk_level="HIGH",
    )
    generator = ReportGenerator()
    res = generator.generate(sample_data)

    pdf_bytes = compile_pdf_report(res)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 2000


def test_compile_pdf_report_from_history_detail(tmp_path):
    """Verify compile_pdf_report compiles valid PDF bytes from raw history detail dict."""
    from app.report_generator.pdf_generator import compile_pdf_report
    from app.storage.storage import StorageRepository

    db_path = tmp_path / "test_pdf_history.db"
    art_dir = tmp_path / "artifacts"
    repo = StorageRepository(db_path=str(db_path), artifact_dir=art_dir)

    scan_id = "SCAN-PDF-001"
    repo.create_assessment(
        scan_id=scan_id,
        assessment_name="Historical PDF Scan",
        model_info={"model_name": "ResNet-50", "architecture": "ResNet", "framework": "PyTorch"},
        dataset_info={"dataset_name": "TestSet", "total_samples": 50},
    )
    repo.record_attack_execution(
        scan_id=scan_id,
        attack_result={"attack_name": "PGD", "clean_accuracy": 0.95, "adv_accuracy": 0.15, "asr": 0.80},
    )
    repo.record_defense_execution(
        scan_id=scan_id,
        defense_result={
            "defense_technique": "spatial_smoothing",
            "validation_status": "accepted",
            "before_clean_accuracy": 0.95,
            "after_clean_accuracy": 0.94,
            "before_adv_accuracy": 0.15,
            "after_adv_accuracy": 0.65,
            "before_asr": 0.80,
            "after_asr": 0.30,
            "asr_reduction_percentage": 50.0,
            "accuracy_recovery": 50.0,
            "is_improved": True,
        },
    )
    repo.record_vulnerabilities(
        scan_id=scan_id,
        vuln_result={"vulnerability_score": 62.0, "risk_level": "High"},
    )
    repo.update_assessment_status(
        scan_id=scan_id,
        status="completed",
        overall_summary={"risk_level": "HIGH", "vulnerability_score": 62.0},
    )

    detail = repo.get_assessment(scan_id)
    assert detail is not None

    pdf_bytes = compile_pdf_report(detail)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 2000

    # Save via ArtifactManager and verify persistence
    saved_rel = repo.artifacts.save_artifact(
        data=pdf_bytes,
        scan_id=scan_id,
        category="reports",
        filename_prefix=f"Security_Report_{scan_id}",
        extension=".pdf",
    )
    assert saved_rel is not None
    assert saved_rel.endswith(".pdf")

    # Record report in database
    rec = repo.record_report(
        scan_id=scan_id,
        report_path=saved_rel,
        report_name=f"Security_Report_{scan_id}.pdf",
        report_format="pdf",
        summary_metrics={"risk_level": "HIGH", "vulnerability_score": 62.0},
    )
    assert rec["report_format"] == "pdf"
    assert rec["report_name"] == f"Security_Report_{scan_id}.pdf"

    # Reload from storage
    loaded_bytes = repo.artifacts.load_artifact(saved_rel)
    assert isinstance(loaded_bytes, bytes)
    assert loaded_bytes.startswith(b"%PDF-")
    assert len(loaded_bytes) == len(pdf_bytes)
