"""
End-to-end multi-modal tests for XAIExplainer across image, text, tabular, and time-series.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from app.explainability.explainer import XAIExplainer
from app.explainability.explanation_result import ExplanationResult
from app.explainability.modality import Modality


class VisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, 2)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))

    def forward(self, x):
        h = self.pool(self.conv(x)).view(x.size(0), -1)
        return self.fc(h)


class TabularModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))
            self.fc.bias.zero_()

    def forward(self, x):
        return self.fc(x)


class TimeSeriesModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Input (B, T, F) -> flatten or pool over T
        self.fc = nn.Linear(5 * 2, 2)  # 5 timesteps, 2 features

    def forward(self, x):
        flat = x.view(x.size(0), -1)
        return self.fc(flat)


class SimpleTextClassifier:
    """Mock text classifier that predicts class 1 if 'bad' or 'error' is in the text."""
    def predict(self, texts):
        res = []
        for t in texts:
            if "bad" in t.lower() or "error" in t.lower() or "hack" in t.lower():
                res.append([0.1, 0.9])
            else:
                res.append([0.9, 0.1])
        return np.array(res)

    def predict_proba(self, texts):
        return self.predict(texts)


def test_explainer_image_saliency_and_gradcam():
    model = VisionModel()
    clean_img = torch.randn(1, 3, 16, 16)
    adv_img = clean_img + 0.1

    explainer = XAIExplainer(default_technique="saliency")

    # Saliency on image
    res_sal = explainer.explain(
        model=model,
        clean_input=clean_img,
        adversarial_input=adv_img,
        technique="saliency",
        attack_name="pgd_vision",
    )
    assert isinstance(res_sal, ExplanationResult)
    assert res_sal.modality == "image"
    assert res_sal.technique == "saliency"
    assert "clean_top_features" in res_sal.feature_importance
    assert len(res_sal.feature_importance["clean_top_features"]) > 0

    # Grad-CAM on image
    res_cam = explainer.explain(
        model=model,
        clean_input=clean_img,
        adversarial_input=adv_img,
        technique="gradcam",
        attack_name="fgsm_vision",
    )
    assert res_cam.technique == "gradcam"
    assert res_cam.modality == "image"


def test_explainer_tabular_saliency_and_shap():
    model = TabularModel()
    clean = torch.tensor([[5.0, 0.0, 0.0, 0.0]])
    adv = torch.tensor([[0.0, 5.0, 0.0, 0.0]])

    explainer = XAIExplainer()

    res = explainer.explain(
        model=model,
        clean_input=clean,
        adversarial_input=adv,
        true_label=0,
        technique="saliency",
        compute_metrics=True,
    )
    assert res.modality == "tabular"
    assert res.prediction_changed is True
    assert res.clean_prediction == 0
    assert res.adversarial_prediction == 1
    assert "perturbation_alignment" in res.comparison
    assert res.comparison["perturbation_alignment"]["alignment_status"] == "success"


def test_explainer_timeseries_integrated_gradients():
    model = TimeSeriesModel()
    clean_ts = torch.randn(1, 5, 2)
    adv_ts = clean_ts + 0.05

    explainer = XAIExplainer(default_technique="integrated_gradients")

    res = explainer.explain(
        model=model,
        clean_input=clean_ts,
        adversarial_input=adv_ts,
        modality="timeseries",
        technique="integrated_gradients",
        steps=5,
    )
    assert res.modality == "timeseries"
    assert res.technique == "integrated_gradients"
    assert "clean_top_features" in res.feature_importance


def test_explainer_text_lime():
    model = SimpleTextClassifier()
    clean_text = "Good clean safe query string"
    adv_text = "Good clean hack query string"

    explainer = XAIExplainer(default_technique="lime")

    res = explainer.explain(
        model=model,
        clean_input=clean_text,
        adversarial_input=adv_text,
        true_label=0,
        modality="text",
        technique="lime",
        num_samples=20,
    )
    assert res.modality == "text"
    assert res.clean_prediction == 0
    assert res.adversarial_prediction == 1
    assert res.prediction_changed is True
    assert res.attack_caused_failure is True


def test_explainer_targeted_attack_analysis():
    model = TabularModel()
    clean = torch.tensor([[5.0, 0.0, 0.0, 0.0]])  # Class 0
    adv = torch.tensor([[0.0, 5.0, 0.0, 0.0]])    # Class 1

    explainer = XAIExplainer()
    res = explainer.explain(
        model=model,
        clean_input=clean,
        adversarial_input=adv,
        target_class=1,
        technique="saliency",
    )
    assert res.comparison.get("target_class") == 1
    assert res.comparison.get("targeted_attack_success") is True
    assert "targeted_attack" in res.failure_analysis
    assert res.failure_analysis["targeted_attack"]["targeted_success"] is True


def test_explainer_shape_mismatch():
    model = VisionModel()
    clean_img = torch.randn(1, 3, 16, 16)
    adv_img = torch.randn(1, 3, 32, 32)

    explainer = XAIExplainer()
    with pytest.raises(ValueError, match="Image shape mismatch"):
        explainer.explain(
            model=model,
            clean_input=clean_img,
            adversarial_input=adv_img,
            technique="saliency",
        )
