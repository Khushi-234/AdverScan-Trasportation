"""
Unit tests for XAIExplainer main coordinator in M6.
"""

import pytest
import torch
import torch.nn as nn

from app.attack_engine.models import AttackMetadata, AttackResult
from app.explainability.explainer import XAIExplainer
from app.explainability.explanation_result import ExplanationResult
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
