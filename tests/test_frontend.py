"""
Tests for AdverScan Frontend Services:
- Defense Service (Module 7 & Module 8 integration)
- Report Service (Module 9 integration)
- Session State Workflow Management
"""

import pytest
import torch
import torch.nn as nn

from frontend.services.defense_service import (
    DOMAIN_DEFENSE_CATALOG,
    execute_configured_defense,
    get_available_defenses_for_domain,
)
from frontend.services.report_service import generate_security_report
from frontend.utils.session_state import (
    get_workflow_stage,
    init_session_state,
    reset_workflow_state,
    set_defense_result,
    get_defense_result,
    set_report_result,
    get_report_result,
    set_workflow_stage,
)


class SimpleVisionModel(nn.Module):
    """Simple 2D CNN model for testing image defenses."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        h = torch.relu(self.conv(x))
        h = self.pool(h).flatten(1)
        return self.fc(h)


def test_get_available_defenses_for_domain():
    """Verify domain-specific defense catalogs."""
    img_defenses = get_available_defenses_for_domain("image")
    assert len(img_defenses) >= 4
    def_ids = [d["id"] for d in img_defenses]
    assert "auto" in def_ids
    assert "spatial_smoothing" in def_ids
    assert "randomized_smoothing" in def_ids

    tab_defenses = get_available_defenses_for_domain("tabular")
    assert any(d["id"] == "auto" for d in tab_defenses)
    assert any(d["id"] == "feature_squeezing" for d in tab_defenses)

    txt_defenses = get_available_defenses_for_domain("nlp")
    assert any(d["id"] == "confidence_rejection" for d in txt_defenses)


def test_execute_configured_defense_specific():
    """Verify specific defense execution with real tensors."""
    model = SimpleVisionModel()
    inputs = torch.rand(4, 3, 16, 16)
    labels = torch.tensor([0, 1, 0, 1])
    adv_inputs = inputs + 0.05 * torch.randn_like(inputs)

    attack_res = {
        "attack_name": "fgsm",
        "clean_accuracy": 100.0,
        "adversarial_accuracy": 25.0,
        "asr_percentage": 75.0,
        "clean_confidence": 95.0,
        "adversarial_confidence": 60.0,
        "confidence_drop": 35.0,
        "total_samples": 4,
        "original_predictions": [0, 1, 0, 1],
        "adversarial_predictions": [1, 0, 1, 1],
    }

    result = execute_configured_defense(
        model=model,
        domain="image",
        defense_choice="spatial_smoothing",
        test_inputs=inputs,
        test_labels=labels,
        adversarial_examples=adv_inputs,
        attack_result=attack_res,
        fallback_to_best=True,
        custom_params={"kernel_size": 3},
    )

    assert result is not None
    assert result["selected_defense"] == "spatial_smoothing"
    assert "after_clean_accuracy" in result
    assert "after_adv_accuracy" in result
    assert "asr_reduction_percentage" in result
    assert len(result["sample_restorations"]) == 4
    assert result["_hardened_model"] is not None


def test_execute_configured_defense_auto_and_fallback():
    """Verify auto iterative defense execution with fallback."""
    model = SimpleVisionModel()
    inputs = torch.rand(4, 3, 16, 16)
    labels = torch.tensor([0, 1, 0, 1])
    adv_inputs = inputs + 0.08 * torch.randn_like(inputs)

    attack_res = {
        "attack_name": "pgd",
        "clean_accuracy": 100.0,
        "adversarial_accuracy": 0.0,
        "asr_percentage": 100.0,
        "total_samples": 4,
        "original_predictions": [0, 1, 0, 1],
        "adversarial_predictions": [1, 0, 1, 0],
    }

    result = execute_configured_defense(
        model=model,
        domain="image",
        defense_choice="auto",
        test_inputs=inputs,
        test_labels=labels,
        adversarial_examples=adv_inputs,
        attack_result=attack_res,
        fallback_to_best=True,
    )

    assert result is not None
    assert result["selected_defense"] is not None
    assert len(result["candidate_attempts"]) > 0


def test_generate_security_report_end_to_end():
    """Verify 15-section report generation, markdown formatting, and JSON serialization."""
    model_meta = {
        "Model Name": "TestResNet",
        "Architecture": "ResNet-18",
        "Framework": "PyTorch",
        "Total Parameters": 11200000,
        "Trainable Parameters": 11200000,
        "Input Shape": "[3, 32, 32]",
        "Classes": 10,
        "device": "cpu",
    }
    dataset_info = {
        "dataset_name": "CIFAR-10 Test",
        "total_samples": 10,
    }
    attack_result = {
        "attack_name": "FGSM",
        "clean_accuracy": 100.0,
        "adversarial_accuracy": 20.0,
        "asr_percentage": 80.0,
        "clean_confidence": 92.5,
        "confidence_drop": 30.0,
        "execution_time": 0.45,
        "total_samples": 10,
        "epsilon_used": 0.05,
    }
    defense_result = {
        "selected_defense": "spatial_smoothing",
        "status": "ACCEPTED",
        "improvement_sufficient": True,
        "fallback_used": False,
        "before_asr": 80.0,
        "after_asr": 30.0,
        "before_clean_accuracy": 100.0,
        "after_clean_accuracy": 98.0,
        "before_adv_accuracy": 20.0,
        "after_adv_accuracy": 70.0,
        "before_vulnerability_score": 80.0,
        "after_vulnerability_score": 30.0,
        "vulnerability_improvement": 50.0,
        "candidate_attempts": [
            {
                "Defense": "Spatial Smoothing",
                "Status": "ACCEPTED",
                "ASR Reduction": "50.0%",
                "Vuln Improvement": "+50.0 pts",
                "Clean Acc Drop": "2.0%",
                "Details": "Met all criteria",
            }
        ],
        "recommendations": ["Deploy spatial smoothing filter in production."],
    }

    report = generate_security_report(
        model_metadata=model_meta,
        dataset_info=dataset_info,
        attack_result=attack_result,
        defense_result=defense_result,
        domain="image",
    )

    assert report is not None
    assert report.report_id is not None
    assert report.risk_level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert report.vulnerability_score is not None

    # Verify Markdown export
    md_text = report.formatted_report
    assert len(md_text) > 100
    assert "TestResNet" in md_text

    # Verify JSON export
    json_str = report.to_json()
    assert "report_id" in json_str
    assert "model_info" in json_str


def test_session_state_workflow_transitions():
    """Verify session state stages including defense and report."""
    init_session_state()

    set_workflow_stage("defense_config")
    assert get_workflow_stage() == "defense_config"

    set_workflow_stage("defense_results")
    assert get_workflow_stage() == "defense_results"

    set_workflow_stage("report")
    assert get_workflow_stage() == "report"

    # Test defense result caching
    mock_def = {"selected_defense": "spatial_smoothing", "status": "ACCEPTED"}
    set_defense_result(mock_def)
    assert get_defense_result() == mock_def

    # Test report caching
    mock_rep = {"report_id": "REP-12345", "status": "SUCCESS"}
    set_report_result(mock_rep)
    assert get_report_result() == mock_rep

    # Reset
    reset_workflow_state()
    assert get_workflow_stage() == "dashboard"
    assert get_defense_result() is None
    assert get_report_result() is None


def test_theme_manager_initialization(monkeypatch):
    """Verify theme manager synchronization with native theme context."""
    from frontend.utils.theme_manager import initialize_theme, load_theme
    import streamlit as st

    # 1. Test default dark theme
    if "theme" in st.session_state:
        del st.session_state["theme"]
    t = initialize_theme()
    assert t in ("Dark", "Light")
    assert st.session_state["theme"] == t

    # 2. Test explicit Light theme loading
    st.session_state["theme"] = "Light"
    t_light = initialize_theme()
    assert t_light == "Light"

    # 3. Test load_theme execution
    load_theme("Dark")
    load_theme("Light")


def test_targeted_attack_execution_and_propagation():
    """Verify targeted attack configuration, parameter propagation, and outcome classification (A01, A02, A03, A04)."""
    from frontend.services.attack_service import execute_configured_attack

    model = SimpleVisionModel()
    inputs = torch.rand(4, 3, 16, 16)
    labels = torch.tensor([0, 0, 0, 0])

    params = {
        "targeted": True,
        "target_label": 1,
    }

    result = execute_configured_attack(
        model=model,
        attack_name="fgsm",
        inputs=inputs,
        labels=labels,
        epsilon=0.1,
        clip_min=0.0,
        clip_max=1.0,
        params=params,
    )

    assert result is not None
    assert result["is_targeted"] is True
    assert result["target_label"] == 1
    assert "parameters_used" in result
    assert result["parameters_used"]["targeted"] is True
    assert result["parameters_used"]["target_label"] == 1
    assert "sample_comparisons" in result
    assert len(result["sample_comparisons"]) == 4


def test_asr_denominator_and_clean_misclassification_handling():
    """Verify ASR excludes cleanly misclassified samples from denominator (A06, A07)."""
    from frontend.services.attack_service import execute_configured_attack

    model = SimpleVisionModel()
    inputs = torch.rand(4, 3, 16, 16)
    
    # Get model's clean predictions and invert them so all samples are cleanly misclassified
    with torch.no_grad():
        preds = model(inputs).argmax(dim=-1)
    labels = 1 - preds  # labels are valid (0 or 1) but opposite to clean predictions

    result = execute_configured_attack(
        model=model,
        attack_name="fgsm",
        inputs=inputs,
        labels=labels,
        epsilon=0.05,
    )

    assert result["clean_accuracy"] == 0.0
    assert result["clean_correct_count"] == 0
    # ASR is undefined and 0.0 when 0 clean samples are correct
    assert result["asr_undefined"] is True
    assert result["asr_percentage"] == 0.0


def test_defense_service_rejects_missing_adversarial_tensor():
    """Verify defense execution never substitutes clean inputs for missing adversarial examples (A08)."""
    model = SimpleVisionModel()
    inputs = torch.rand(4, 3, 16, 16)
    labels = torch.tensor([0, 1, 0, 1])

    attack_res = {
        "attack_name": "fgsm",
        "clean_accuracy": 100.0,
        "adversarial_accuracy": 0.0,
        "asr_percentage": 100.0,
        "total_samples": 4,
    }

    with pytest.raises(ValueError, match="Adversarial examples tensor is missing"):
        execute_configured_defense(
            model=model,
            domain="image",
            defense_choice="spatial_smoothing",
            test_inputs=inputs,
            test_labels=labels,
            adversarial_examples=None,
            attack_result=attack_res,
        )


def test_vulnerability_service_authentic_scoring_and_explanations():
    """Verify authentic M5 VulnerabilityEngine assessment and scoring (A11, A12, A13, A14)."""
    from frontend.services.vulnerability_service import assess_and_score_vulnerability

    model = SimpleVisionModel()
    inputs = torch.rand(4, 3, 16, 16)
    labels = torch.tensor([0, 1, 0, 1])
    adv_inputs = inputs + 0.08 * torch.randn_like(inputs)

    attack_result = {
        "attack_name": "FGSM",
        "attack_class": "FGSM",
        "execution_time": 0.12,
        "total_samples": 4,
        "clean_accuracy": 100.0,
        "adversarial_accuracy": 25.0,
        "asr_percentage": 75.0,
        "clean_confidence": 92.0,
        "adversarial_confidence": 55.0,
        "confidence_drop": 37.0,
        "original_predictions": [0, 1, 0, 1],
        "adversarial_predictions": [1, 1, 1, 1],
        "perturbation_metrics": {"linf": 0.08, "l2": 0.25, "l0": 100},
        "_adversarial_examples": adv_inputs,
        "epsilon_used": 0.08,
    }

    vuln_res = assess_and_score_vulnerability(
        model=model,
        attack_result=attack_result,
        inputs=inputs,
        labels=labels,
        domain="image",
    )

    assert vuln_res is not None
    assert "vulnerability_score" in vuln_res
    assert vuln_res["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert "sub_scores" in vuln_res
    assert "asr_score" in vuln_res["sub_scores"]
    assert "degradation_score" in vuln_res["sub_scores"]
    assert "stealth_score" in vuln_res["sub_scores"]
    assert "metric_explanations" in vuln_res
    assert "summary" in vuln_res["metric_explanations"]
    assert "asr_component" in vuln_res["metric_explanations"]
    assert "degradation_component" in vuln_res["metric_explanations"]
    assert "stealth_component" in vuln_res["metric_explanations"]
    assert "risk_rationale" in vuln_res["metric_explanations"]


def test_synthetic_dataset_fallback_flagging():
    """Verify synthetic fallback clearly marks dataset as synthetic (A05)."""
    from frontend.services.ingestion_service import load_model_and_dataset

    model = SimpleVisionModel()
    # Attempting to load non-existent dataset will trigger synthetic fallback
    res = load_model_and_dataset(
        model_identifier=model,
        dataset_name="non_existent_dataset_abc_123",
        data_domain="image",
        sample_size=4,
    )

    assert res["is_synthetic_dataset"] is True
    assert "Synthetic Fallback" in res["dataset_name"]
    assert res["telemetry"]["is_synthetic_dataset"] is True
    assert res["telemetry"]["dataset_source"] == "synthetic_fallback"


def test_report_service_no_fabricated_values():
    """Verify reports use measured values and do not fabricate loss or stealth metrics (A15)."""
    model_meta = {"Model Name": "M", "Architecture": "A", "Framework": "PyTorch"}
    dataset_info = {"dataset_name": "D", "total_samples": 4}
    attack_result = {
        "attack_name": "FGSM",
        "clean_accuracy": 100.0,
        "adversarial_accuracy": 0.0,
        "asr_percentage": 100.0,
        "perturbation_metrics": {"linf": 0.05},
        "total_samples": 4,
    }

    report = generate_security_report(
        model_metadata=model_meta,
        dataset_info=dataset_info,
        attack_result=attack_result,
        domain="image",
    )

    # Verify baseline performance loss is "Not Measured" when not provided, not 0.25
    assert report.baseline_performance.get("loss") == "Not Measured"
    # Verify report contains valid vulnerability score
    assert report.vulnerability_score is not None


def test_session_state_vulnerability_result():
    """Verify session state holds and retrieves vulnerability results (A13)."""
    from frontend.utils.session_state import (
        get_vulnerability_result,
        set_vulnerability_result,
    )

    mock_vuln = {"vulnerability_score": 72.5, "risk_level": "HIGH"}
    set_vulnerability_result(mock_vuln)
    assert get_vulnerability_result() == mock_vuln


def test_vulnerability_recommendations_generation():
    """Verify vulnerability recommendations are generated with proper metadata."""
    from frontend.services.vulnerability_service import (
        generate_vulnerability_recommendations,
    )

    # Test untargeted image domain
    recs_img = generate_vulnerability_recommendations(
        domain="image",
        attack_name="FGSM",
        vuln_score=85.0,
        risk_level="CRITICAL",
        asr=90.0,
        is_targeted=False,
    )
    assert len(recs_img) >= 3
    for r in recs_img:
        assert "category" in r
        assert "icon" in r
        assert "title" in r
        assert "priority" in r
        assert "description" in r
        assert len(r["description"]) > 10

    # Test targeted tabular domain
    recs_tab = generate_vulnerability_recommendations(
        domain="tabular",
        attack_name="HopSkipJump",
        vuln_score=65.0,
        risk_level="HIGH",
        asr=75.0,
        is_targeted=True,
    )
    assert len(recs_tab) >= 4
    # Targeted recommendation should be first
    assert recs_tab[0]["category"] == "Targeted Threat Defense"
    assert recs_tab[0]["priority"] == "Urgent"


def test_no_bar_charts_in_stage_4_attack_results():
    """Verify that unnecessary bar charts have been removed from attack results stage."""
    from pathlib import Path

    app_code = Path("frontend/streamlit_app.py").read_text(encoding="utf-8")
    # Find the attack_results stage block
    stage_start = app_code.find('stage == "attack_results":')
    stage_end = app_code.find('stage == "defense_config":')
    assert stage_start != -1 and stage_end != -1

    stage_code = app_code[stage_start:stage_end]
    # Verify no st.bar_chart calls are present in stage 4
    assert "st.bar_chart" not in stage_code
    # Verify proper tables are used instead
    assert "st.table" in stage_code
    assert "Perturbation Norms & Geometric Bounds" in stage_code
    assert "Actionable Security & Hardening Recommendations" in stage_code


def test_dataset_sample_population_and_selection_options():
    """Verify dataset total samples resolution and UI selection options (half, full, custom)."""
    from frontend.services.ingestion_service import get_dataset_total_samples
    from pathlib import Path

    # Test preset & disk lookups
    assert get_dataset_total_samples("adult_census", "tabular", "tabular-mlp-adult-census") == 250
    assert get_dataset_total_samples("credit_fraud", "tabular", "tabular-mlp-credit-fraud") == 250
    assert get_dataset_total_samples("bazyl/GTSRB", "image", "bazyl/gtsrb-model") == 12630
    assert get_dataset_total_samples("time_series_ecg", "time-series", "time-series-ecg-classifier") == 200

    # Verify frontend/streamlit_app.py contains half/full/custom selectors
    app_code = Path("frontend/streamlit_app.py").read_text(encoding="utf-8")
    assert "Half Dataset" in app_code
    assert "Full Dataset" in app_code
    assert "Custom Number of Samples" in app_code
    assert "Dataset Population in Model/Benchmark" in app_code


def test_perturbation_metrics_backend_format_and_outcome_counts():
    """Verify backend format support for perturbation norms and sample outcome counts."""
    from app.attack_engine.utils.perturbation import compute_perturbation_metrics
    import torch

    x1 = torch.zeros(2, 3, 4, 4)
    x2 = x1 + 0.05
    metrics = compute_perturbation_metrics(x1, x2)

    # Verify both standard and extended backend formats are present
    assert "linf" in metrics and "linf_mean" in metrics and "linf_max" in metrics
    assert "l2" in metrics and "l2_mean" in metrics and "l2_max" in metrics
    assert "l0" in metrics and "l0_mean_count" in metrics and "l0_mean_pct" in metrics
    assert "mean_abs_error" in metrics and "mae_mean" in metrics
    assert metrics["l0_mean_pct"] == 100.0
    assert abs(metrics["linf_mean"] - 0.05) < 1e-4

    # Verify sample outcome count logic in streamlit_app.py
    from pathlib import Path
    app_code = Path("frontend/streamlit_app.py").read_text(encoding="utf-8")
    assert 'succ_o = int(result.get("successful_count"' in app_code
    assert 'clean_correct = int(result.get("clean_correct_count"' in app_code





