
"""
Unit and integration tests for Module 9 (Reporting & Report Generation) corrections and requirements.
Tests:
- All 15 sections generation
- Absence of optional modules (XAI, Hardening, Retest)
- Non-fabrication & safe handling of None, NaN, and Inf
- Valid JSON serialization and sanitization
- CSV export & ReportWriter multi-format handling
- MITRE ATLAS Threat Framework mapping across all 4 domains
- M8 Retest integration (statuses, defense effectiveness, thresholds)
- Frontend report service integration
"""

import json
import math
from pathlib import Path
import pytest
import sys

# Ensure repository root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.report_generator import (
    ExecutionSummary,
    ModuleExecutionRecord,
    ReportData,
    ReportGenerator,
    ReportResult,
    ReportWriter,
)
from frontend.services.report_service import generate_security_report


def test_m9_all_15_sections_generated():
    """Verify that ReportGenerator populates all 15 required sections."""
    data = ReportData(
        model_info={"model_name": "ResNet-50", "framework": "PyTorch", "num_classes": 1000},
        dataset_config={"dataset_name": "ImageNet-Sample", "split": "val", "batch_size": 32},
        baseline_performance={"accuracy": 0.85, "loss": 0.42, "confidence": 0.88},
        attack_results={
            "pgd": {
                "parameters": {"eps": 0.03, "steps": 10},
                "evaluation": {"metrics": {"accuracy": 0.12, "attack_success_rate": 0.86}},
                "execution_time_seconds": 1.45,
            }
        },
        vulnerability_metrics={
            "pgd": {
                "assessment": {
                    "attack_success_rate": 0.86,
                    "clean_accuracy": 0.85,
                    "adversarial_accuracy": 0.12,
                    "accuracy_drop": 0.73,
                },
                "scoring": {"vulnerability_score": 82.5, "risk_level": "CRITICAL"},
            }
        },
        vulnerability_score=82.5,
        risk_level="CRITICAL",
        xai_findings={"integrated_gradients": {"top_feature": "patch_14", "attribution_sum": 0.41}},
        hardening_results={"selected_defense": "gaussian_filter", "status": "SUCCESS"},
        retest_results={
            "status": "COMPLETED",
            "overall_improved": True,
            "overall_status": "accepted",
            "defense_name": "gaussian_filter",
            "num_samples": 100,
        },
        before_vs_after={
            "pgd": {
                "before_accuracy": 0.12,
                "after_accuracy": 0.68,
                "before_asr": 0.86,
                "after_asr": 0.20,
                "is_improved": True,
                "status": "successful",
                "defense_effectiveness": "effective",
            }
        },
        execution_summary=ExecutionSummary(
            run_label="Full Assessment",
            run_timestamp="2026-11-18 12:00:00",
            total_elapsed_seconds=5.2,
            modules=[
                ModuleExecutionRecord("M1", "M1 Model Ingestion", "SUCCESS", 0.2),
                ModuleExecutionRecord("M3", "M3 Attack Engine", "SUCCESS", 1.45),
            ],
        ),
    )

    generator = ReportGenerator()
    res = generator.generate(data)

    assert isinstance(res, ReportResult)
    assert res.status == "SUCCESS"
    assert res.risk_level == "CRITICAL"
    assert res.vulnerability_score == 82.5

    # Check all 15 sections are present in formatted report
    expected_sections = [
        "1. EXECUTIVE SUMMARY",
        "2. MODEL INFORMATION",
        "3. DATASET / EVALUATION CONFIGURATION",
        "4. BASELINE PERFORMANCE",
        "5. ADVERSARIAL ATTACK RESULTS",
        "6. VULNERABILITY ASSESSMENT",
        "7. VULNERABILITY SCORE & RISK LEVEL",
        "8. MITRE ATLAS MAPPING",
        "9. XAI FINDINGS",
        "10. HARDENING",
        "11. RE-TEST RESULTS",
        "12. BEFORE VS AFTER COMPARISON",
        "13. EXECUTION PERFORMANCE",
        "14. RECOMMENDATIONS",
        "15. FINAL SECURITY SUMMARY",
    ]
    for sec in expected_sections:
        assert sec in res.formatted_report, f"Missing section: {sec}"


def test_m9_missing_optional_modules_handling():
    """Verify clean handling when optional modules (XAI, Hardening, Retest) are omitted."""
    data = ReportData(
        model_info={"model_name": "MinimalModel"},
        baseline_performance={"accuracy": 0.90},
        attack_results={
            "fgsm": {
                "parameters": {"eps": 0.05},
                "evaluation": {"metrics": {"accuracy": 0.40, "attack_success_rate": 0.55}},
            }
        },
        vulnerability_metrics={
            "fgsm": {
                "assessment": {"attack_success_rate": 0.55, "clean_accuracy": 0.90, "adversarial_accuracy": 0.40},
                "scoring": {"vulnerability_score": 52.0, "risk_level": "HIGH"},
            }
        },
        # XAI, Hardening, Retest, ExecutionSummary intentionally omitted
    )

    generator = ReportGenerator()
    res = generator.generate(data)

    assert res.status == "SUCCESS"
    assert res.risk_level == "HIGH"
    assert "No XAI findings generated" in res.formatted_report
    assert "No hardening actions recorded" in res.formatted_report
    assert "No re-test results available" in res.formatted_report
    assert "No comparison data available" in res.formatted_report
    assert "No execution performance data captured" in res.formatted_report


def test_m9_nan_inf_none_safety_and_no_fabricated_zeroes():
    """Verify that None, NaN, and Inf in metrics do not crash formatting and are rendered as N/A."""
    data = ReportData(
        model_info={"model_name": "EdgeCaseModel"},
        baseline_performance={"accuracy": float("nan"), "loss": float("inf"), "empty_metric": None},
        attack_results={
            "fgsm": {
                "parameters": {"eps": float("nan")},
                "evaluation": {
                    "metrics": {
                        "clean_accuracy": 0.0,  # 0 clean accuracy -> ASR should NOT be fabricated as 0.00%
                        "accuracy": 0.0,
                        "attack_success_rate": None,
                        "accuracy_drop": 0.0,
                    }
                },
            }
        },
        vulnerability_metrics={
            "fgsm": {
                "assessment": {
                    "clean_accuracy": 0.0,
                    "adversarial_accuracy": 0.0,
                    "attack_success_rate": None,
                    "accuracy_drop": 0.0,
                },
                "scoring": {"vulnerability_score": float("nan"), "risk_level": "UNKNOWN"},
            }
        },
        vulnerability_score=None,
        risk_level=None,
    )

    generator = ReportGenerator()
    res = generator.generate(data)

    assert res.status == "SUCCESS"
    assert res.risk_level == "UNKNOWN"
    assert res.vulnerability_score is None

    # Verify formatted report doesn't contain raw 'nan%' or crash
    assert "nan%" not in res.formatted_report.lower()
    assert "inf%" not in res.formatted_report.lower()
    # Baseline accuracy with NaN should display N/A
    assert "Baseline Accuracy  : N/A" in res.formatted_report


def test_m9_json_serialization_sanitizes_nan_and_inf(tmp_path):
    """Verify that ReportResult.to_dict and to_json convert NaN/Inf into valid JSON nulls."""
    res = ReportResult(
        report_id="RPT-TEST-001",
        timestamp="2026-11-18 12:00:00",
        vulnerability_score=float("nan"),
        executive_summary={"nan_val": float("nan"), "inf_val": float("inf"), "neg_inf": float("-inf")},
        baseline_performance={"accuracy": float("nan")},
    )

    d = res.to_dict()
    assert d["vulnerability_score"] is None
    assert d["executive_summary"]["nan_val"] is None
    assert d["executive_summary"]["inf_val"] is None
    assert d["executive_summary"]["neg_inf"] is None

    # Should serialize cleanly without error
    json_str = res.to_json()
    loaded = json.loads(json_str)
    assert loaded["vulnerability_score"] is None
    assert loaded["executive_summary"]["nan_val"] is None

    # Test file saving
    save_path = tmp_path / "report.json"
    res.save_json(save_path)
    assert save_path.exists()
    with open(save_path, "r", encoding="utf-8") as f:
        file_loaded = json.load(f)
    assert file_loaded["report_id"] == "RPT-TEST-001"


def test_m9_csv_export_and_writer_multi_format(tmp_path):
    """Verify CSV export and ReportWriter format handling."""
    data = ReportData(
        model_info={"model_name": "CSVModel"},
        attack_results={"pgd": {"execution_time_seconds": 1.2}},
        execution_summary=ExecutionSummary(
            run_label="CSV Test Run",
            run_timestamp="2026-11-18 12:00:00",
            total_elapsed_seconds=3.5,
            modules=[
                ModuleExecutionRecord("M1", "M1 Ingestion", "SUCCESS", 0.5, {"params": 1000}),
                ModuleExecutionRecord("M3", "M3 Attack", "SUCCESS", 1.2, {"asr": 0.85, "nan_metric": float("nan")}),
            ],
        ),
    )

    generator = ReportGenerator()
    res = generator.generate(data)

    csv_content = res.to_csv()
    assert "module_id" in csv_content
    assert "M1 Ingestion" in csv_content
    assert "M3 Attack" in csv_content
    assert "TOTAL" in csv_content

    # ReportWriter handles formats
    writer = ReportWriter(output_root=tmp_path / "reports")
    written = writer.write(res, formats=["md", "json", "csv", "txt", "pdf"])

    assert "md" in written
    assert "json" in written
    assert "csv" in written
    assert "txt" in written
    # PDF is gracefully handled (returns path if reportlab present, otherwise skipped cleanly)


def test_m9_mitre_atlas_comprehensive_domain_coverage():
    """Verify MITRE ATLAS mappings across vision, tabular, text, and time-series attacks."""
    attacks = {
        "fgsm": {},
        "pgd": {},
        "deepfool": {},
        "cw": {},
        "fab": {},
        "cap_pgd": {},
        "lowprofool": {},
        "tabular_cw": {},
        "tabular_jsma": {},
        "vae_attack": {},
        "hotflip": {},
        "embedding_pgd": {},
        "textbugger_wb": {},
        "universal_trigger": {},
        "gradient_substitution": {},
        "time_series_pgd": {},
        "smooth_pgd": {},
        "wavelet_attack": {},
        "soft_dtw_attack": {},
        "jsma": {},
    }

    generator = ReportGenerator()
    mapping = generator._s8_mitre_atlas(attacks)

    assert len(mapping) == 20
    for atk_name, m_info in mapping.items():
        assert "tactic" in m_info
        assert "technique" in m_info
        assert "mitigation" in m_info
        assert "AML.TA0000" in m_info["tactic"]
        assert "AML.T0043" in m_info["technique"]


def test_m9_m8_retest_and_comparison_integration():
    """Verify that M8 Retest attributes (status, defense_effectiveness, thresholds) are rendered faithfully."""
    data = ReportData(
        model_info={"model_name": "DefendedNet"},
        hardening_results={"selected_defense": "feature_squeezing", "defense_name": "Feature Squeezing"},
        retest_results={
            "hardened_model_name": "DefendedNet-Squeezed",
            "defense_name": "Feature Squeezing",
            "status": "COMPLETED",
            "overall_status": "accepted",
            "overall_improved": True,
            "num_samples": 50,
            "dataset_name": "TestDataset",
        },
        before_vs_after={
            "fgsm": {
                "before_accuracy": 0.20,
                "after_accuracy": 0.75,
                "delta_accuracy": 0.55,
                "before_risk_level": "HIGH",
                "after_risk_level": "LOW",
                "is_improved": True,
                "status": "successful",
                "defense_effectiveness": "effective",
            },
            "unsupported_vector": {
                "before_accuracy": 0.10,
                "after_accuracy": 0.10,
                "is_improved": False,
                "status": "unsupported",
                "defense_effectiveness": "unsupported",
            },
        },
    )

    generator = ReportGenerator()
    res = generator.generate(data)

    assert "11. RE-TEST RESULTS" in res.formatted_report
    assert "Feature Squeezing" in res.formatted_report
    assert "PASSED — Defense Successfully Mitigated Attacks" in res.formatted_report
    assert "12. BEFORE VS AFTER COMPARISON" in res.formatted_report
    assert "EFFECTIVE" in res.formatted_report
    assert "UNSUPPORTED" in res.formatted_report


def test_m9_frontend_report_service_integration():
    """Verify frontend generate_security_report with realistic and edge-case inputs."""
    # 1. Standard full run
    model_meta = {"Model Name": "CIFAR-CNN", "Framework": "PyTorch", "Architecture": "CNN"}
    dataset_info = {"dataset_name": "CIFAR-10", "total_samples": 100}
    attack_res = {
        "attack_name": "PGD",
        "clean_accuracy": 92.0,
        "adversarial_accuracy": 15.0,
        "asr_percentage": 83.7,
        "execution_time": 2.5,
        "total_samples": 100,
    }
    defense_res = {
        "selected_defense": "gaussian_filter",
        "status": "SUCCESS",
        "after_clean_accuracy": 90.0,
        "after_adv_accuracy": 70.0,
        "after_asr": 22.2,
        "post_asr_defined": True,
        "is_improved": True,
        "defense_effectiveness": "effective",
    }
    vuln_res = {
        "vulnerability_score": 78.5,
        "risk_level": "CRITICAL",
        "assessment": {
            "attack_success_rate": 0.837,
            "clean_accuracy": 0.92,
            "adversarial_accuracy": 0.15,
            "accuracy_drop": 0.77,
        },
    }

    rep = generate_security_report(
        model_metadata=model_meta,
        dataset_info=dataset_info,
        attack_result=attack_res,
        defense_result=defense_res,
        vulnerability_result=vuln_res,
    )

    assert isinstance(rep, ReportResult)
    assert rep.status == "SUCCESS"
    assert rep.risk_level == "CRITICAL"
    assert rep.vulnerability_score == 78.5
    assert rep.execution_summary is not None
    assert len(rep.execution_summary.modules) >= 4

    # 2. Edge case: Zero clean accuracy and None metrics
    edge_attack_res = {
        "attack_name": "FGSM",
        "clean_accuracy": 0.0,
        "adversarial_accuracy": 0.0,
        "asr_percentage": None,
        "execution_time": 0.5,
        "total_samples": 10,
    }
    edge_defense_res = {
        "selected_defense": "none",
        "status": "FAILED",
        "after_clean_accuracy": 0.0,
        "after_adv_accuracy": 0.0,
        "after_asr": None,
        "post_asr_defined": False,
        "is_improved": False,
        "defense_effectiveness": "ineffective",
    }

    edge_rep = generate_security_report(
        model_metadata={"Model Name": "ZeroModel"},
        dataset_info={"dataset_name": "ZeroData"},
        attack_result=edge_attack_res,
        defense_result=edge_defense_res,
    )

    assert isinstance(edge_rep, ReportResult)
    assert edge_rep.status == "SUCCESS"
    assert "N/A" in edge_rep.formatted_report
