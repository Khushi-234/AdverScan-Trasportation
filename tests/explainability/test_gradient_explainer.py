"""
Unit tests for GradientExplainer (Saliency, Integrated Gradients, Grad-CAM).
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from app.explainability.techniques.gradient_explainer import GradientExplainer


class VisionConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        h = self.pool(self.conv1(x)).view(x.size(0), -1)
        return self.fc(h)


class TabularMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_saliency_tabular():
    model = TabularMLP()
    inputs = torch.randn(1, 4)

    explainer = GradientExplainer(method="saliency")
    res = explainer.explain(model, inputs)

    assert res["status"] == "success"
    assert res["executed"] is True
    assert res["technique"] == "saliency"
    assert res["attribution"] is not None
    assert np.asarray(res["attribution"]).shape == (1, 4)


def test_integrated_gradients_tabular():
    model = TabularMLP()
    inputs = torch.randn(1, 4)
    baseline = torch.zeros(1, 4)

    explainer = GradientExplainer(method="integrated_gradients")
    res = explainer.explain(model, inputs, baseline=baseline, steps=10)

    assert res["status"] == "success"
    assert res["executed"] is True
    assert res["technique"] == "integrated_gradients"
    assert res["attribution"] is not None
    assert np.asarray(res["attribution"]).shape == (1, 4)


def test_gradcam_vision():
    model = VisionConvModel()
    inputs = torch.randn(1, 3, 16, 16)

    explainer = GradientExplainer(method="gradcam")
    res = explainer.explain(model, inputs, target_class=0)

    assert res["status"] == "success"
    assert res["executed"] is True
    assert res["technique"] == "gradcam"
    assert res["attribution"] is not None
    cam_arr = np.asarray(res["attribution"])
    assert cam_arr.shape == (1, 1, 16, 16)


def test_gradient_explainer_unsupported_model():
    # Non-torch callable or object
    class NonTorchModel:
        def predict(self, x):
            return np.array([0, 1])

    explainer = GradientExplainer(method="saliency")
    res = explainer.explain(NonTorchModel(), torch.randn(1, 4))

    assert res["status"] == "unsupported"
    assert res["executed"] is False
    assert res["attribution"] is None


def test_gradcam_unsupported_model_no_conv():
    # Tabular model has no Conv2d layers
    model = TabularMLP()
    explainer = GradientExplainer(method="gradcam")
    res = explainer.explain(model, torch.randn(1, 4))

    assert res["status"] == "unsupported"
    assert res["executed"] is False
    assert "no suitable Conv2d layers" in res["message"]
