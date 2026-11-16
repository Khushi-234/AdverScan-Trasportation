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


def test_report_generation_with_vulnerability_and_defense_results():
    """Verify generate_security_report runs without UnboundLocalError when vulnerability_result and defense_result are provided."""
    model_meta = {
        "model_name": "TestVisionModel",
        "domain": "image",
        "framework": "pytorch",
        "parameters": 1000,
        "input_shape": [3, 32, 32],
    }
    attack_res = {
        "attack_name": "FGSM",
        "clean_accuracy": 100.0,
        "adversarial_accuracy": 20.0,
        "accuracy_drop": 80.0,
        "asr_percentage": 80.0,
        "perturbation_metrics": {
            "linf": 0.05,
            "linf_mean": 0.05,
            "linf_max": 0.05,
            "l2": 1.25,
            "l2_mean": 1.25,
            "l0": 3000,
            "l0_mean_count": 3000,
            "l0_mean_pct": 97.6,
            "mean_abs_error": 0.04,
            "mae_mean": 0.04,
        },
    }
    vuln_res = {
        "vulnerability_score": 75.5,
        "risk_level": "HIGH",
        "assessment": {
            "attack_success_rate": 0.80,
            "clean_accuracy": 1.0,
            "adversarial_accuracy": 0.20,
            "accuracy_drop": 0.80,
            "perturbation": {
                "linf_mean": 0.05,
                "l2_mean": 1.25,
                "l0_mean_count": 3000,
                "mae_mean": 0.04,
            },
        },
        "sub_scores": {
            "attack_success_rate_score": 80.0,
            "accuracy_drop_score": 80.0,
            "stealth_score": 70.0,
        },
    }
    defense_res = {
        "selected_defense": "spatial_smoothing",
        "status": "sufficient",
        "is_improved": True,
        "improvement_sufficient": True,
        "clean_accuracy_before": 100.0,
        "clean_accuracy_after": 98.0,
        "clean_accuracy_drop": 2.0,
        "asr_before": 80.0,
        "asr_after": 25.0,
        "asr_reduction": 55.0,
        "vulnerability_score_before": 75.5,
        "vulnerability_score_after": 32.0,
        "vulnerability_score_improvement": 43.5,
        "thresholds": {
            "max_clean_accuracy_drop": 0.02,
            "min_asr_reduction": 0.10,
            "min_vuln_score_improvement": 5.0,
        },
    }

    report = generate_security_report(
        model_metadata=model_meta,
        dataset_info={"dataset_name": "cifar10", "num_samples": 5},
        attack_result=attack_res,
        defense_result=defense_res,
        vulnerability_result=vuln_res,
        domain="image",
    )
    assert report is not None
    assert report.report_id is not None
    assert report.vulnerability_score == 75.5
    assert report.risk_level == "HIGH"
    assert "spatial_smoothing" in report.formatted_report
    assert "RPT-" in report.report_id


def test_defense_execution_batch_size_alignment():
    """Verify execute_configured_defense safely aligns batch sizes when clean inputs/labels have 10 samples and adversarial has 5."""
    model = SimpleVisionModel()
    inputs_10 = torch.randn(10, 3, 32, 32)
    labels_10 = torch.randint(0, 2, (10,))
    adv_5 = inputs_10[:5] + 0.05
    atk_res = {
        "attack_name": "fgsm",
        "clean_accuracy": 100.0,
        "adversarial_accuracy": 0.0,
        "asr_percentage": 100.0,
        "original_predictions": labels_10[:5],
        "adversarial_predictions": 1 - labels_10[:5],
    }

    res = execute_configured_defense(
        model=model,
        domain="image",
        defense_choice="spatial_smoothing",
        test_inputs=inputs_10,
        test_labels=labels_10,
        adversarial_examples=adv_5,
        attack_result=atk_res,
    )
    assert res is not None
    assert res["total_samples"] == 5
    assert res["selected_defense"] == "spatial_smoothing"
    assert "threshold_compliance" in res


def test_defense_results_no_unnecessary_charts_and_threshold_compliance():
    """Verify that stage 6 (defense_results) has no st.bar_chart calls and includes threshold compliance."""
    from pathlib import Path

    app_code = Path("frontend/streamlit_app.py").read_text(encoding="utf-8")
    stage_start = app_code.find('stage == "defense_results":')
    stage_end = app_code.find('stage == "report":')
    assert stage_start != -1 and stage_end != -1

    stage_code = app_code[stage_start:stage_end]
    # Verify no st.bar_chart calls remain in defense_results
    assert "st.bar_chart" not in stage_code
    # Verify modern table and threshold compliance are present
    assert "Acceptance Thresholds & Robustness Compliance" in stage_code
    assert "Compliance Verdict" in stage_code or "threshold_compliance" in stage_code


def test_class_labels_mapping_and_targeted_attack_telemetry():
    """Verify class semantic mappings and targeted attack metadata."""
    from frontend.services.attack_service import (
        get_class_labels_mapping,
        execute_configured_attack,
    )

    # 1. Test class label mappings for known benchmarks
    census_map = get_class_labels_mapping(dataset_name="adult_census")
    assert 0 in census_map and 1 in census_map
    assert "<=50K" in census_map[0]
    assert ">50K" in census_map[1]

    cifar_map = get_class_labels_mapping(dataset_name="uoft-cs/cifar10")
    assert len(cifar_map) == 10
    assert cifar_map[0] == "Airplane"
    assert cifar_map[3] == "Cat"

    # 2. Test targeted attack telemetry execution
    model = SimpleVisionModel()
    inputs = torch.randn(4, 3, 32, 32)
    labels = torch.zeros(4, dtype=torch.long)  # All ground truth = 0

    res = execute_configured_attack(
        model=model,
        attack_name="fgsm",
        inputs=inputs,
        labels=labels,
        epsilon=0.05,
        params={"targeted": True, "target_label": 1},
        dataset_name="adult_census",
    )
    assert res is not None
    assert res["is_targeted"] is True
    assert res["target_label"] == 1
    assert ">50K" in res["target_label_name"]
    assert "targeted_hit_count" in res
    assert "collateral_flip_count" in res
    assert len(res["sample_comparisons"]) == 4


def test_sample_table_pagination_and_filter_controls():
    """Verify pagination and outcome filter controls exist in both attack_results and defense_results stages."""
    from pathlib import Path

    app_code = Path("frontend/streamlit_app.py").read_text(encoding="utf-8")

    # In attack_results stage
    assert "atk_sample_filter" in app_code
    assert "atk_page_size" in app_code
    assert "atk_prev_page" in app_code
    assert "atk_next_page" in app_code

    # In defense_results stage
    assert "def_sample_filter" in app_code
    assert "def_page_size" in app_code
    assert "def_prev_page" in app_code
    assert "def_next_page" in app_code


def test_stage3_sample_selection_defaults_to_all():
    """Verify Stage 3 evaluation sample selection defaults to all ingested samples and supports half/custom."""
    from pathlib import Path

    app_code = Path("frontend/streamlit_app.py").read_text(encoding="utf-8")

    # Verify sample population selection structure
    assert "Evaluation Sample Population (Batch Slicing)" in app_code
    assert 'f"All Ingested ({available_samples:,} samples • 100%)"' in app_code
    assert 'f"Half Ingested ({half_avail:,} samples • 50%)"' in app_code
    assert '"Custom Number of Samples"' in app_code
    # Must default to index=0 (All Ingested)
    assert "index=0" in app_code


def test_cw_dynamic_attack_parameters_has_no_alpha():
    """Verify Carlini & Wagner (CW) configuration never uses an alpha step size and exposes correct L2 parameters."""
    from pathlib import Path
    from frontend.services.attack_service import execute_configured_attack

    app_code = Path("frontend/streamlit_app.py").read_text(encoding="utf-8")

    # Verify CW dynamic block does not assign alpha to params_dict
    cw_block_start = app_code.find("if is_cw:")
    cw_block_end = app_code.find("elif is_deepfool:", cw_block_start)
    assert cw_block_start != -1 and cw_block_end != -1
    cw_code = app_code[cw_block_start:cw_block_end]

    assert 'params_dict["alpha"]' not in cw_code
    assert "Step Size (α)" not in cw_code
    assert "initial_const" in cw_code
    assert "confidence" in cw_code
    assert "learning_rate" in cw_code
    assert "max_iterations" in cw_code

    # Test backend execution with CW parameters
    model = SimpleVisionModel()
    inputs = torch.rand(2, 3, 16, 16)
    labels = torch.tensor([0, 1])
    res = execute_configured_attack(
        model=model,
        attack_name="cw",
        inputs=inputs,
        labels=labels,
        epsilon=0.0,
        clip_min=0.0,
        clip_max=1.0,
        params={
            "max_iterations": 5,
            "learning_rate": 0.01,
            "initial_const": 1.0,
            "confidence": 0.0,
            "binary_search_steps": 1,
        },
    )
    assert res is not None
    assert "alpha" not in res["parameters_used"]
    assert res["parameters_used"]["max_iterations"] == 5


def test_deepfool_and_jsma_dynamic_parameters():
    """Verify DeepFool and JSMA configure algorithm-accurate parameters without alpha."""
    from pathlib import Path
    from frontend.services.attack_service import execute_configured_attack

    app_code = Path("frontend/streamlit_app.py").read_text(encoding="utf-8")

    # DeepFool
    df_block_start = app_code.find("elif is_deepfool:")
    df_block_end = app_code.find("elif is_jsma:", df_block_start)
    assert df_block_start != -1 and df_block_end != -1
    df_code = app_code[df_block_start:df_block_end]
    assert 'params_dict["alpha"]' not in df_code
    assert "overshoot" in df_code
    assert "top_k" in df_code
    assert "max_iter" in df_code

    # JSMA
    jsma_block_start = app_code.find("elif is_jsma:")
    jsma_block_end = app_code.find("elif is_soft_dtw:", jsma_block_start)
    assert jsma_block_start != -1 and jsma_block_end != -1
    jsma_code = app_code[jsma_block_start:jsma_block_end]
    assert 'params_dict["alpha"]' not in jsma_code
    assert "max_features" in jsma_code
    assert "step_size" in jsma_code

    # FGSM
    fgsm_block_start = app_code.find("elif is_fgsm:")
    fgsm_block_end = app_code.find("elif is_fab:", fgsm_block_start)
    assert fgsm_block_start != -1 and fgsm_block_end != -1
    fgsm_code = app_code[fgsm_block_start:fgsm_block_end]
    assert 'params_dict["alpha"]' not in fgsm_code
    assert 'params_dict["num_steps"]' not in fgsm_code


def test_batched_attack_execution_prevents_oom():
    """Verify that execute_configured_attack processes batches exceeding single-chunk thresholds without failure."""
    from frontend.services.attack_service import execute_configured_attack

    model = SimpleVisionModel()
    # 70 vision samples exceeds vision chunk size (32) to test multi-chunk execution & concatenation
    inputs = torch.rand(70, 3, 16, 16)
    labels = torch.randint(0, 2, (70,))

    res = execute_configured_attack(
        model=model,
        attack_name="fgsm",
        inputs=inputs,
        labels=labels,
        epsilon=0.05,
    )

    assert res is not None
    assert len(res["sample_comparisons"]) == 70
    assert res["_adversarial_examples"].size(0) == 70
    assert "clean_accuracy" in res
    assert "adversarial_accuracy" in res
    assert "asr_percentage" in res










