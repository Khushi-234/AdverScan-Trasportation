"""
Unit tests for SHAPExplainer technique in M6.
"""

from dataclasses import dataclass
import numpy as np
import pytest
import torch
import torch.nn as nn

torch.set_num_threads(2)

from app.explainability.techniques.shap_explainer import SHAPExplainer
from app.ingestion.adapters.pytorch_adapter import PyTorchAdapter


class SimpleVectorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


@dataclass
class ImageClassifierOutput:
    logits: torch.Tensor


class HFImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        h = self.conv(x).mean(dim=[2, 3])
        logits = self.fc(h)
        return ImageClassifierOutput(logits=logits)


class NonCallablePredictModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        # Non-callable property named 'predict'
        self.predict = np.array([0.5, 0.5])

    def forward(self, x):
        return self.fc(x)


def test_1_tiny_synthetic_vector_model():
    model = SimpleVectorModel()
    inputs = torch.randn(1, 4)

    explainer = SHAPExplainer()
    res = explainer.explain(model, inputs, target_class=0, nsamples=8)

    assert isinstance(res, dict)
    assert res["status"] == "success"
    assert res["executed"] is True
    assert res["attribution"] is not None
    attr = np.array(res["attribution"])
    assert attr.shape == (1, 4)
    assert not np.isnan(attr).any()
    assert np.isfinite(attr).all()


def test_2_tiny_synthetic_image_model():
    model = HFImageModel()
    inputs = torch.randn(1, 3, 32, 32)

    explainer = SHAPExplainer()
    # grid_size=4 -> 16 features max
    res = explainer.explain(model, inputs, target_class=0, nsamples=8, grid_size=4)

    assert res["status"] == "success"
    assert res["executed"] is True
    assert res["attribution"] is not None
    attr = np.array(res["attribution"])
    assert attr.shape == (1, 3, 32, 32)
    assert not np.isnan(attr).any()
    assert np.isfinite(attr).all()


def test_3_huggingface_style_output():
    hf_model = HFImageModel()
    inputs = torch.randn(1, 3, 32, 32)

    explainer = SHAPExplainer()
    res = explainer.explain(hf_model, inputs, target_class=1, nsamples=8, grid_size=4)

    assert res["status"] == "success"
    assert res["executed"] is True
    assert res["attribution"] is not None


def test_4_non_callable_predict_attribute():
    model = NonCallablePredictModel()
    inputs = torch.randn(1, 4)

    explainer = SHAPExplainer()
    res = explainer.explain(model, inputs, target_class=0, nsamples=8)

    assert res["status"] == "success"
    assert res["executed"] is True
    assert res["attribution"] is not None


def test_5_multi_sample_vector_feature_mean_background():
    model = SimpleVectorModel()
    # Multi-sample vector batch (3, 4)
    inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0], [5.0, 6.0, 7.0, 8.0]])

    explainer = SHAPExplainer()
    res = explainer.explain(model, inputs, target_class=0, nsamples=8)

    assert res["status"] == "success"
    assert res["executed"] is True
    attr = np.array(res["attribution"])
    assert attr.shape == (3, 4)
    assert np.isfinite(attr).all()


def test_6_explicit_background_inputs_respected():
    model = SimpleVectorModel()
    inputs = torch.randn(1, 4)
    explicit_bg = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])

    explainer = SHAPExplainer()
    res = explainer.explain(model, inputs, target_class=0, background_inputs=explicit_bg, nsamples=8)

    assert res["status"] == "success"
    assert res["executed"] is True
    attr = np.array(res["attribution"])
    assert attr.shape == (1, 4)
    assert np.isfinite(attr).all()


def test_7_image_background_finite_and_shaped():
    model = HFImageModel()
    inputs = torch.rand(2, 3, 16, 16)  # 2 sample batch

    explainer = SHAPExplainer()
    res = explainer.explain(model, inputs, target_class=0, nsamples=8, grid_size=4)

    assert res["status"] == "success"
    assert res["executed"] is True
    assert res["technique"] == "shap"
    attr = np.array(res["attribution"])
    assert attr.shape == (2, 3, 16, 16)
    assert np.isfinite(attr).all()

