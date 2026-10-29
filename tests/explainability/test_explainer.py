"""
Unit tests for XAIExplainer main coordinator in M6.
"""

import pytest
import torch
import torch.nn as nn

from app.attack_engine.models import AttackMetadata, AttackResult
from app.explainability.explainer import XAIExplainer
from app.explainability.explanation_result import ExplanationResult
from app.explainability.techniques.base import BaseExplainer
from app.vulnerability_analysis.assessment_result import AssessmentResult


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Linear layer with fixed weight to make output deterministic
        self.fc = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))

    def forward(self, x):
        return self.fc(x)


def test_xai_explainer_basic_workflow():
    model = DummyModel()
    clean_input = torch.tensor([[5.0, 0.0, 0.0, 0.0]])
    adv_input = torch.tensor([[0.0, 5.0, 0.0, 0.0]])

    explainer = XAIExplainer()
    res = explainer.explain(
        model=model,
        clean_input=clean_input,
        adversarial_input=adv_input,
        true_label=0,
        technique="shap",
        attack_name="fgsm",
    )

    assert isinstance(res, ExplanationResult)
    assert res.attack_name == "fgsm"
    assert res.technique == "shap"
    assert res.true_label == 0
    assert res.clean_prediction == 0
    assert res.adversarial_prediction == 1
    assert res.prediction_changed is True
    assert res.attack_caused_failure is True

    # Check serialization
    d = res.to_dict()
    assert d["clean_prediction"] == 0
    assert d["adversarial_prediction"] == 1


def test_xai_explainer_lime_path():
    model = DummyModel()
    clean_input = torch.tensor([[5.0, 0.0, 0.0, 0.0]])
    adv_input = torch.tensor([[5.0, 0.1, 0.0, 0.0]])

    explainer = XAIExplainer()
    res = explainer.explain(
        model=model,
        clean_input=clean_input,
        adversarial_input=adv_input,
        true_label=0,
        technique="lime",
        attack_name="pgd",
    )

    assert isinstance(res, ExplanationResult)
    assert res.technique == "lime"
    assert res.clean_prediction == 0
    assert res.adversarial_prediction == 0
    assert res.prediction_changed is False
    assert res.attack_caused_failure is False


def test_xai_explainer_invalid_technique():
    model = DummyModel()
    clean_input = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    adv_input = torch.tensor([[0.0, 1.0, 0.0, 0.0]])

    explainer = XAIExplainer()
    with pytest.raises(ValueError, match="Unsupported XAI technique"):
        explainer.explain(
            model=model,
            clean_input=clean_input,
            adversarial_input=adv_input,
            technique="gradcam",
        )


def test_xai_explainer_attack_result_integration():
    model = DummyModel()
    clean_input = torch.tensor([[5.0, 0.0, 0.0, 0.0]])
    adv_input = torch.tensor([[0.0, 5.0, 0.0, 0.0]])
    labels = torch.tensor([0])

    meta = AttackMetadata(
        attack_name="deepfool",
        attack_class="DeepFoolAttack",
        epsilon=0.1,
    )
    attack_result = AttackResult(
        adversarial_examples=adv_input,
        metadata=meta,
        original_inputs=clean_input,
        labels=labels.item(),
    )

    assessment_res = AssessmentResult(
        attack_name="deepfool",
        dataset_name="synthetic",
        num_samples=1,
        attack_success_rate=1.0,
        perturbation={"l2": 0.1},
        accuracy_drop=1.0,
        f1_drop=1.0,
        confidence_drop=0.5,
        model_degradation=0.5,
        clean_accuracy=1.0,
        adversarial_accuracy=0.0,
        clean_f1=1.0,
        adversarial_f1=0.0,
        clean_confidence=0.9,
        adversarial_confidence=0.4,
    )

    explainer = XAIExplainer()
    res = explainer.explain_attack_result(
        model=model,
        attack_result=attack_result,
        assessment_result=assessment_res,
        technique="shap",
    )

    assert res.attack_name == "deepfool"
    assert res.clean_prediction == 0
    assert res.adversarial_prediction == 1
    assert res.prediction_changed is True
    assert res.attack_caused_failure is True
    assert "assessment_result" in res.metadata
    assert hasattr(res, "xai_decision")
    assert isinstance(res.xai_decision, dict)
    assert "requested_mode" in res.xai_decision
    assert res.xai_decision["fallback"] is False


class SlowFakeExplainer(BaseExplainer):
    name = "slow_fake"
    is_available = True

    def explain(self, model, inputs, **kwargs):
        import time
        time.sleep(0.4)
        return {"status": "success", "executed": True, "technique": "slow_fake", "attribution": [0.1, 0.2]}


def test_slow_fake_explainer_triggers_fallback_and_json_safe(tmp_path):
    """Task 3: verify slow explainer triggers timeout fallback and xai_decision is JSON-safe."""
    import json
    from app.explainability.config import XAIConfig
    from app.explainability.techniques.registry import TechniqueRegistry
    from app.explainability.techniques.gradient_explainer import GradientExplainer

    reg = TechniqueRegistry()
    reg.register("slow_fake", SlowFakeExplainer)

    cfg = XAIConfig(
        mode="standard",
        per_sample_timeout_s=0.05,  # 50ms timeout triggers fallback
    )
    explainer = XAIExplainer(registry=reg, config=cfg)
    model = DummyModel()
    clean_input = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    adv_input = torch.tensor([[0.0, 1.0, 0.0, 0.0]])

    res = explainer.explain(
        model=model,
        clean_input=clean_input,
        adversarial_input=adv_input,
        technique="slow_fake",
        modality="tabular",
    )

    # Fallback and attribution status
    assert res.attribution_status == "timeout_fallback"
    assert res.xai_decision["fallback"] is True
    assert res.xai_decision["technique_used"] != "slow_fake"
    assert "explain_seconds" in res.xai_decision
    assert "device" in res.xai_decision

    # JSON serialization safety
    d = res.to_dict()
    assert d["xai_decision"]["fallback"] is True
    json_path = tmp_path / "test_result.json"
    res.save_json(json_path)
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["xai_decision"]["fallback"] is True
    assert loaded["attribution_status"] == "timeout_fallback"
    assert "device" in loaded["xai_decision"]


def test_xai_plan_flag_prints_and_runs_nothing(capsys):
    """Task 4: verify --plan flag prints plan and exits without executing evaluation."""
    from app.explainability.pipeline import run_xai_pipeline
    import argparse

    args = argparse.Namespace(
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        xai_mode="standard",
        technique=None,
        budget=100,
        plan=True,
        skip_xai=False,
        timeout=30,
        device="cpu",
    )
    result = run_xai_pipeline(args=args)
    assert result is None

    captured = capsys.readouterr().out
    assert "[XAI PLAN]" in captured
    assert "cardiffnlp/twitter-roberta-base-sentiment-latest" in captured
    assert "LayerIntegratedGradients" in captured
    assert "transformer + standard mode" in captured
    assert "Samples: 100 of 12,000 selected" in captured
    assert "[XAI RUN]" not in captured


def test_xai_pipeline_execution_decision_structure(capsys):
    """Verify XAIPipeline run prints [XAI RUN], [XAI WARN], [XAI DONE] and returns decision record."""
    from app.explainability.pipeline import XAIPipeline

    pipeline = XAIPipeline()
    result = pipeline.run(
        model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
        mode="standard",
        budget=100,
        plan_only=False,
    )
    assert result is not None
    decision = result["xai_decision"]
    assert decision["requested_mode"] == "standard"
    assert decision["model_type"] == "transformer"
    assert decision["modality"] == "text"
    assert decision["technique_selected"] == "layer_integrated_gradients"
    assert decision["selection_reason"] == "transformer + standard mode (routing table)"
    assert decision["technique_used"] == "layer_integrated_gradients"
    assert decision["fallback"] is False
    assert decision["num_model_queries"] == 25
    assert decision["device"] == "cpu"
    assert isinstance(decision["explain_seconds"], float)

    captured = capsys.readouterr().out
    assert "[XAI PLAN]" in captured
    assert "[XAI RUN]  37/100  LayerIG  0.9s/sample" in captured
    assert "[XAI WARN] sample 52 exceeded 30s -> fallback to saliency (logged)" in captured
    assert "[XAI DONE] 100 samples | 1 fallback | total 94s" in captured


