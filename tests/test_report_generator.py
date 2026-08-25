"""
Unit and integration tests for Module 9 (Report Generator) in AdverScan.
"""

import json
import pytest

from app.report_generator import ReportData, ReportResult, ReportGenerator


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
        assert "ADVERSCAN SECURITY REPORT" in loaded_text


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
