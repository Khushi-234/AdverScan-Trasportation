"""
Comprehensive regression tests for M6 Explainability implementation and corrections.

Covers:
1. Global and local explanations.
2. Feature attribution correctness and non-fabrication.
3. Multiclass behavior and target class alignment.
4. Multidomain compatibility:
   - Vision (4D tensors, spatial heatmaps)
   - Tabular (2D feature vectors, column attributions)
   - Text (discrete tokens, token alignment)
   - Time Series (3D sequential data, temporal and sensor attributions)
5. Unsupported model handling and routing validation.
6. Unavailable dependencies handling (graceful 'unavailable' diagnostic).
7. Explanation failure resilience (does not crash vulnerability assessment).
8. M5 integration (preserves AssessmentResult and ScoringResult metadata).
9. Frontend service and report integration.
"""

from typing import Any, Dict
import numpy as np
import pytest
import torch
import torch.nn as nn

from app.attack_engine.models import AttackMetadata, AttackResult
from app.evaluation.results import EvaluationResult
from app.explainability.aggregate import aggregate_explanations
from app.explainability.config import XAIConfig
from app.explainability.explainer import XAIExplainer
from app.explainability.explanation_result import ExplanationResult
from app.explainability.modality import Modality
from app.explainability.techniques.gradient_explainer import GradientExplainer
from app.explainability.techniques.lime_explainer import LIMEExplainer
from app.explainability.techniques.registry import (
    TechniqueRegistry,
    check_override_compatibility,
    technique_registry,
)
from app.explainability.techniques.shap_explainer import SHAPExplainer
from app.explainability.techniques.tree_shap_explainer import TreeSHAPExplainer
from app.vulnerability_analysis.assessment_result import AssessmentResult
from app.vulnerability_analysis.results import PerturbationResult
from app.vulnerability_analysis.scorer import ScoringResult
from frontend.services.report_service import generate_security_report
from frontend.services.xai_service import explain_assessment, get_available_xai_techniques


# ---------------------------------------------------------------------------
# Test Fixtures & Dummy Architectures
# ---------------------------------------------------------------------------
class DummyTabularNet(nn.Module):
    def __init__(self, in_features: int = 4, num_classes: int = 3):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DummyVisionCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(x))
        h = self.pool(h).view(x.size(0), -1)
        return self.fc(h)


class DummyTextModel(nn.Module):
    def __init__(self, vocab_size: int = 100, embed_dim: int = 16, num_classes: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: Any) -> torch.Tensor:
        if isinstance(x, (list, tuple)):
            if x and isinstance(x[0], str):
                # Map token strings or words to integer indices
                words = " ".join(x).split() if len(x) == 1 else x
                indices = [abs(hash(w)) % 100 for w in words] or [0]
                x = torch.tensor([indices], dtype=torch.long)
            else:
                x = torch.tensor(x, dtype=torch.long)
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.long)
        elif x.dtype != torch.long:
            x = x.long()

        if x.ndim == 1:
            x = x.unsqueeze(0)
        emb = self.embedding(x).mean(dim=1)
        return self.fc(emb)


class DummyTimeSeriesLSTM(nn.Module):
    def __init__(self, num_sensors: int = 3, hidden_dim: int = 8, num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(num_sensors, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ---------------------------------------------------------------------------
# 1. Global and Local Explanations
# ---------------------------------------------------------------------------
def test_local_and_global_explanations():
    """Verify local single-sample attribution and global dataset-level aggregation."""
    model = DummyTabularNet(in_features=4, num_classes=2)
    clean = torch.tensor([[1.0, 2.0, 0.5, 0.1]])
    adv = torch.tensor([[1.0, 0.2, 2.5, 0.1]])

    explainer = XAIExplainer()
    local_res = explainer.explain(
        model=model,
        clean_input=clean,
        adversarial_input=adv,
        technique="saliency",
        modality="tabular",
    )

    assert isinstance(local_res, ExplanationResult)
    assert local_res.attribution_status in ("success", "valid")
    assert local_res.clean_attribution is not None

    # Run on multiple samples to test global aggregation
    batch_clean = torch.tensor([[1.0, 2.0, 0.5, 0.1], [0.5, 1.0, 3.0, 0.2]])
    batch_adv = torch.tensor([[1.0, 0.2, 2.5, 0.1], [0.5, 0.1, 0.2, 0.2]])
    batch_res = explainer.explain_batch(
        model=model,
        clean_inputs=batch_clean,
        adversarial_inputs=batch_adv,
        technique="saliency",
        modality="tabular",
    )

    assert isinstance(batch_res, list)
    assert len(batch_res) == 2

    global_summary = aggregate_explanations(batch_res)
    assert "global_top_features" in global_summary
    assert "prediction_transitions" in global_summary
    assert global_summary["total_samples_evaluated"] == 2


# ---------------------------------------------------------------------------
# 2. Feature Attribution Correctness & Directionality
# ---------------------------------------------------------------------------
def test_feature_attribution_correctness():
    """A linear model where output = 10 * x[0] - 5 * x[1] must assign highest positive attribution to feature 0."""
    weight = torch.tensor([[10.0, -5.0, 0.0, 0.0]])
    model = nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(weight)

    x = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    grad_exp = GradientExplainer(method="saliency")
    res = grad_exp.explain(model, x, target_class=1)

    assert res["status"] == "success"
    attr = np.squeeze(np.array(res["attribution"]))
    assert attr[0] > attr[1]  # feature 0 has stronger positive weight than feature 1
    assert np.all(np.isfinite(attr))


# ---------------------------------------------------------------------------
# 3. Multiclass Behavior and Target Class Alignment
# ---------------------------------------------------------------------------
def test_multiclass_class_specific_explanations():
    """Different target classes must generate distinct class-aligned attributions."""
    # 3-class linear model
    weights = torch.tensor([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
    ])
    model = nn.Linear(3, 3, bias=False)
    with torch.no_grad():
        model.weight.copy_(weights)

    x = torch.tensor([[1.0, 1.0, 1.0]])
    grad_exp = GradientExplainer(method="saliency")

    res_c0 = grad_exp.explain(model, x, target_class=0)
    res_c1 = grad_exp.explain(model, x, target_class=1)
    res_c2 = grad_exp.explain(model, x, target_class=2)

    attr_0 = np.squeeze(np.array(res_c0["attribution"]))
    attr_1 = np.squeeze(np.array(res_c1["attribution"]))
    attr_2 = np.squeeze(np.array(res_c2["attribution"]))

    assert attr_0[0] > attr_0[1] and attr_0[0] > attr_0[2]
    assert attr_1[1] > attr_1[0] and attr_1[1] > attr_1[2]
    assert attr_2[2] > attr_2[0] and attr_2[2] > attr_2[1]


# ---------------------------------------------------------------------------
# 4. Multidomain Compatibility
# ---------------------------------------------------------------------------
def test_vision_modality_spatial_attribution():
    """Vision 4D inputs produce spatial attribution matching input image dimensions."""
    model = DummyVisionCNN()
    img = torch.rand(1, 3, 16, 16)
    adv_img = img + 0.05 * torch.randn_like(img)

    explainer = XAIExplainer()
    res = explainer.explain(
        model=model,
        clean_input=img,
        adversarial_input=adv_img,
        technique="saliency",
        modality="image",
    )

    assert res.attribution_status in ("success", "valid")
    attr = np.array(res.clean_attribution)
    # Raw attribution matches (B, C, H, W)
    assert attr.shape == (1, 3, 16, 16)
    # Postprocessed 2D heatmap has spatial dimensions (16, 16)
    heatmap = np.array(res.attribution["clean"]["postprocessed"]["heatmap"])
    assert heatmap.shape == (16, 16)
    assert np.all(np.isfinite(heatmap))


def test_tabular_modality_column_attribution():
    """Tabular 2D inputs produce 1D feature attribution of length matching feature dimension."""
    model = DummyTabularNet(in_features=5, num_classes=2)
    clean = torch.randn(1, 5)
    adv = clean + 0.1

    explainer = XAIExplainer()
    res = explainer.explain(
        model=model,
        clean_input=clean,
        adversarial_input=adv,
        technique="integrated_gradients",
        modality="tabular",
        feature_names=["f0", "f1", "f2", "f3", "f4"],
    )

    assert res.attribution_status in ("success", "valid")
    attr = np.squeeze(np.array(res.clean_attribution))
    assert attr.shape == (5,)
    assert len(res.feature_importance.get("clean_top_features", [])) > 0


def test_text_modality_discrete_token_ids():
    """Text inputs with integer token IDs are explained via Layer IG without treating IDs as continuous floats."""
    model = DummyTextModel(vocab_size=100, embed_dim=16, num_classes=2)
    # Integer token IDs
    clean_ids = torch.tensor([[12, 45, 67, 89]], dtype=torch.long)
    adv_ids = torch.tensor([[12, 45, 99, 89]], dtype=torch.long)

    explainer = XAIExplainer()
    res = explainer.explain(
        model=model,
        clean_input=clean_ids,
        adversarial_input=adv_ids,
        technique="layer_integrated_gradients",
        modality="text",
    )

    assert res.attribution_status in ("success", "valid")
    attr = np.array(res.clean_attribution)
    assert len(attr) > 0


def test_timeseries_modality_sensor_attribution():
    """Time-series 3D inputs produce (T, F) temporal and sensor attributions."""
    model = DummyTimeSeriesLSTM(num_sensors=3, hidden_dim=8, num_classes=2)
    clean_ts = torch.randn(1, 6, 3)
    adv_ts = clean_ts + 0.05

    explainer = XAIExplainer()
    res = explainer.explain(
        model=model,
        clean_input=clean_ts,
        adversarial_input=adv_ts,
        technique="integrated_gradients",
        modality="timeseries",
    )

    assert res.attribution_status in ("success", "valid")
    attr = np.squeeze(np.array(res.clean_attribution))
    assert attr.shape == (6, 3)


# ---------------------------------------------------------------------------
# 5. Unsupported Model and Incompatible Technique Handling
# ---------------------------------------------------------------------------
def test_unsupported_model_raises_compatibility_error():
    """Requesting TreeSHAP on a PyTorch neural network raises a clear descriptive ValueError."""
    model = DummyTabularNet(in_features=4, num_classes=2)

    with pytest.raises(ValueError, match="requires tree-based models"):
        technique_registry.select_by_mode(
            mode="standard",
            model=model,
            modality="tabular",
            override="tree_shap",
        )


def test_gradcam_on_non_conv_model_rejected():
    """Requesting Grad-CAM on a non-convolutional tabular model raises a clear compatibility error."""
    model = DummyTabularNet(in_features=4, num_classes=2)

    with pytest.raises(ValueError, match="requires convolutional layers"):
        technique_registry.select_by_mode(
            mode="standard",
            model=model,
            modality="tabular",
            override="gradcam",
        )


# ---------------------------------------------------------------------------
# 6. Unavailable Dependency Handling (Graceful Degradation)
# ---------------------------------------------------------------------------
def test_unavailable_dependency_returns_diagnostic():
    """When a technique library is marked unavailable, explainer returns 'unavailable' status with diagnostic."""
    explainer = SHAPExplainer()
    # Simulate environment where library is not installed
    explainer.is_available = False

    res = explainer.explain(model=DummyTabularNet(), inputs=np.zeros((1, 4)))
    assert res["status"] == "unavailable"
    assert res["executed"] is False
    assert res["attribution"] is None
    assert "not installed" in res["message"].lower()


# ---------------------------------------------------------------------------
# 7. Explanation Failure Resilience (No Crash on M5/Vulnerability Assessment)
# ---------------------------------------------------------------------------
def test_explanation_failure_does_not_crash_pipeline():
    """When attack result is missing inputs, explain_attack_result with raise_on_error=False returns error DTO."""
    explainer = XAIExplainer()
    # Missing original_inputs
    broken_attack_res = {
        "attack_name": "broken_test",
        "original_inputs": None,
        "adversarial_examples": None,
    }

    res = explainer.explain_attack_result(
        model=DummyTabularNet(),
        attack_result=broken_attack_res,
        raise_on_error=False,
    )

    assert isinstance(res, ExplanationResult)
    assert res.attribution_status == "error"
    assert "missing" in res.metadata.get("error", "").lower()


# ---------------------------------------------------------------------------
# 8. M5 Integration: Context Preservation and Immutability
# ---------------------------------------------------------------------------
def test_m5_integration_preserves_scoring_and_assessment():
    """M5 AssessmentResult and ScoringResult metadata are preserved intact without modifying assessment scores."""
    model = DummyTabularNet(in_features=4, num_classes=2)
    clean = torch.tensor([[1.0, 0.5, 0.2, 0.8]])
    adv = torch.tensor([[0.2, 0.5, 0.2, 0.8]])

    assessment_res = AssessmentResult(
        attack_name="pgd",
        dataset_name="synthetic",
        num_samples=1,
        attack_success_rate=1.0,
        perturbation=PerturbationResult(linf_mean=0.08, is_estimated=False),
        accuracy_drop=1.0,
        f1_drop=0.8,
        confidence_drop=0.6,
        model_degradation=0.7,
        clean_accuracy=1.0,
        adversarial_accuracy=0.0,
        clean_f1=1.0,
        adversarial_f1=0.0,
        clean_confidence=0.95,
        adversarial_confidence=0.35,
    )

    scoring_res = ScoringResult(
        attack_name="pgd",
        vulnerability_score=78.5,
        risk_level="CRITICAL",
        sub_scores={"asr_score": 100.0, "degradation_score": 70.0, "stealth_score": 50.0},
        metadata={"domain": "tabular"},
    )

    attack_result_dict = {
        "attack_name": "pgd",
        "original_inputs": clean,
        "adversarial_examples": adv,
        "labels": torch.tensor([0]),
    }

    explainer = XAIExplainer()
    xai_res = explainer.explain_attack_result(
        model=model,
        attack_result=attack_result_dict,
        assessment_result=assessment_res,
        scoring_result=scoring_res,
        technique="saliency",
        modality="tabular",
    )

    assert "assessment_result" in xai_res.metadata
    assert xai_res.metadata["assessment_result"]["attack_success_rate"] == 1.0
    assert "scoring_result" in xai_res.metadata
    assert xai_res.metadata["scoring_result"]["vulnerability_score"] == 78.5
    assert xai_res.metadata["scoring_result"]["risk_level"] == "CRITICAL"


# ---------------------------------------------------------------------------
# 9. Frontend XAI Service Integration
# ---------------------------------------------------------------------------
def test_frontend_xai_service_explain_assessment():
    """Frontend xai_service coordinates authentic XAI without crashing and populates report findings."""
    model = DummyTabularNet(in_features=4, num_classes=2)
    inputs = torch.randn(2, 4)
    adv_inputs = inputs + 0.1

    attack_res = {
        "attack_name": "FGSM",
        "_clean_inputs": inputs,
        "_adversarial_examples": adv_inputs,
        "labels": torch.tensor([0, 1]),
    }

    vuln_res = {
        "vulnerability_score": 65.0,
        "risk_level": "HIGH",
    }

    # 1. Run frontend XAI service
    xai_res = explain_assessment(
        model=model,
        attack_result=attack_res,
        vulnerability_result=vuln_res,
        domain="tabular",
        technique="saliency",
    )

    assert xai_res["status"] == "success"
    assert xai_res["executed"] is True
    assert "local_explanation" in xai_res
    assert "global_explanation" in xai_res

    # 2. Verify available techniques query
    avail_techs = get_available_xai_techniques(domain="tabular", model=model)
    assert len(avail_techs) > 0
    assert "saliency" in avail_techs

    # 3. Verify report integration
    m_info = {"model_name": "TestNet", "domain": "tabular"}
    d_info = {"dataset_name": "Synthetic"}
    report = generate_security_report(
        model_metadata=m_info,
        dataset_info=d_info,
        attack_result=attack_res,
        domain="tabular",
        vulnerability_result=vuln_res,
        xai_result=xai_res,
    )

    assert report is not None
    assert report.xai_findings == xai_res
