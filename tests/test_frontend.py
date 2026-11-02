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
