"""
Tests for Stage 2 fixes: Technique-level bugs (E2.1 - E2.7).
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

from app.explainability.modality import Modality
from app.explainability.techniques.gradient_explainer import GradientExplainer
from app.explainability.techniques.lime_explainer import LIMEExplainer
from app.explainability.techniques.shap_explainer import SHAPExplainer, TreeSHAPExplainer


def test_e2_1_shap_slice_shapes_lstm_and_tree():
    """
    Test E2.1:
    - SHAP slice shapes for (1, T, F) single-output LSTM are preserved and feature axis is NOT sliced.
    - TreeSHAP correctly slices (N, F, C) by class index or handles list of arrays.
    """
    # 1. Single-output LSTM: input shape (1, 10, 4)
    class DummyLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=4, hidden_size=8, batch_first=True)
            self.fc = nn.Linear(8, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])  # (N, 1)

    lstm_model = DummyLSTM().eval()
    x_ts = np.random.randn(1, 10, 4).astype(np.float32)

    explainer = SHAPExplainer()
    res = explainer.explain(
        model=lstm_model,
        inputs=x_ts,
        target_class=1,
        output_mode="regression",
        modality=Modality.TIMESERIES,
    )
    assert res["status"] == "success"
    attr = np.array(res["attribution"])
    # Crucial assertion: the shape MUST be (1, 10, 4) — feature axis 4 must NOT be sliced!
    assert attr.shape == (1, 10, 4)
    assert res.get("unsliced_shape") in ((1, 10, 4), (1, 10, 4, 1))

    # 2. Tree model with multi-class output (N, F, C)
    X = np.random.randn(40, 5)
    y = np.array([0, 1, 2] * 13 + [0])
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)

    tree_expl = TreeSHAPExplainer()
    sample = X[:1]
    res_tree_c0 = tree_expl.explain(rf, sample, target_class=0)
    res_tree_c1 = tree_expl.explain(rf, sample, target_class=1)

    assert res_tree_c0["status"] == "success"
    assert res_tree_c1["status"] == "success"
    attr_c0 = np.array(res_tree_c0["attribution"])
    attr_c1 = np.array(res_tree_c1["attribution"])
    assert attr_c0.shape == (1, 5)
    assert attr_c1.shape == (1, 5)
    # The two class attributions should differ for a 3-class model
    assert not np.allclose(attr_c0, attr_c1)


def test_e2_2_shap_background_strategy_and_3d():
    """
    Test E2.2:
    - SHAP honors baseline_strategy='mean'.
    - SHAP handles 3D time-series data without failing or timing out.
    """
    def dummy_pred(x):
        # x is (N, 5, 2)
        return np.mean(x, axis=(1, 2), keepdims=True)

    explainer = SHAPExplainer()
    x_ts = np.ones((1, 5, 2), dtype=np.float32)
    bg = np.zeros((10, 5, 2), dtype=np.float32)

    res = explainer.explain(
        model=dummy_pred,
        inputs=x_ts,
        background_inputs=bg,
        baseline_strategy="mean",
        output_mode="regression",
    )
    assert res["status"] == "success"
    attr = np.array(res["attribution"])
    assert attr.shape == (1, 5, 2)


def test_e2_3_and_e2_5_lime_target_class_and_tabular_kwargs():
    """
    Test E2.3 and E2.5:
    - LIME with target != top label returns nonzero weights for target class.
    - All features (num_features=F) are explained.
    - Non-LIME kwargs (like top_k, feature_names) do not cause TypeError.
    """
    # A model where class 0 is top label, but we ask for class 1
    def predict_fn(x):
        # class 0 has 0.8 prob, class 1 has 0.2 prob
        n = len(x)
        p0 = 0.8 * np.ones((n, 1))
        # add slight dependency on x[:, 1]
        p1 = 0.2 * np.ones((n, 1)) + 0.01 * x[:, 1:2]
        total = p0 + p1
        return np.hstack([p0 / total, p1 / total])

    explainer = LIMEExplainer()
    x_sample = np.array([[1.0, 2.0, 3.0, 4.0]])

    res = explainer.explain(
        model=predict_fn,
        inputs=x_sample,
        target_class=1,  # Non-top class!
        top_k=3,         # Extra kwarg that previously caused TypeError in explain_instance
        feature_names=["f0", "f1", "f2", "f3"],
        num_samples=50,
        modality="tabular",
    )

    assert res["status"] == "success"
    assert res["target_class"] == 1
    attr = res["attribution"]
    assert len(attr) == 4
    # Weights for target class should not be all zeros
    assert np.any(np.abs(attr) > 0.0)


def test_e2_4_lime_image_continuous_weights():
    """
    Test E2.4:
    - LIME image attribution is a continuous weight map, NOT a ternary mask {-1, 0, 1}.
    """
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
            self.fc = nn.Linear(4 * 16 * 16, 2)

        def forward(self, x):
            h = torch.relu(self.conv(x))
            return self.fc(h.view(x.size(0), -1))

    model = SimpleCNN().eval()
    img = np.random.rand(1, 3, 16, 16).astype(np.float32)

    explainer = LIMEExplainer()
    res = explainer.explain(
        model=model,
        inputs=img,
        target_class=0,
        num_samples=25,
        modality=Modality.IMAGE,
    )

    assert res["status"] == "success"
    attr = np.array(res["attribution"])
    # Attribution must have spatial dimensions (16, 16)
    assert attr.shape == (16, 16)
    unique_vals = np.unique(attr)
    # If it was a ternary mask, unique values would only be a subset of {-1, 0, 1}
    # Continuous map will have continuous float weights
    assert not set(unique_vals).issubset({-1.0, 0.0, 1.0})


def test_e2_6_lime_timeseries_sensor_attribution():
    """
    Test E2.6:
    - Time-series LIME unrolls (T, F), generates attributions, and returns (T, F) map.
    """
    def ts_predict_fn(x):
        # x shape (N, 6, 3)
        mean_v = np.mean(x, axis=(1, 2))
        p1 = 1.0 / (1.0 + np.exp(-mean_v))
        return np.stack([1.0 - p1, p1], axis=-1)

    explainer = LIMEExplainer()
    ts_sample = np.random.randn(1, 6, 3).astype(np.float32)

    res = explainer.explain(
        model=ts_predict_fn,
        inputs=ts_sample,
        target_class=1,
        num_samples=30,
        sensor_names=["temp", "vibration", "pressure"],
        modality=Modality.TIMESERIES,
    )

    assert res["status"] == "success"
    attr = np.array(res["attribution"])
    assert attr.shape == (6, 3)
    assert np.any(np.abs(attr) > 0.0)


def test_e2_7_gradient_saliency_n1_model_and_gradcam_inplace_relu():
    """
    Test E2.7:
    - Saliency on (N, 1) model handles class 0 and class 1 correctly with flipped signs.
    - Grad-CAM functions properly with inplace ReLU and Conv2d / Conv1d.
    - Integrated Gradients runs with background mean baseline.
    """
    # 1. Single-output (N, 1) MLP
    class SingleOutputMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 1, bias=False)
            with torch.no_grad():
                self.fc1.weight.copy_(torch.tensor([[1.0, 2.0, -1.0, 0.5]]))

        def forward(self, x):
            return self.fc1(x)

    mlp = SingleOutputMLP().eval()
    x = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

    saliency_expl = GradientExplainer(method="saliency")
    res_c1 = saliency_expl.explain(mlp, x, target_class=1)
    res_c0 = saliency_expl.explain(mlp, x, target_class=0)

    assert res_c1["status"] == "success"
    assert res_c0["status"] == "success"
    grad_c1 = np.array(res_c1["attribution"])
    grad_c0 = np.array(res_c0["attribution"])

    # Gradients for class 0 must have flipped sign of class 1
    assert np.allclose(grad_c1, -grad_c0)
    assert np.allclose(grad_c1[0], [1.0, 2.0, -1.0, 0.5])

    # 2. Grad-CAM on model with inplace ReLU
    class InplaceReLUConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU(inplace=True)  # inplace ReLU!
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(4, 2)

        def forward(self, inp):
            h = self.conv1(inp)
            h = self.relu1(h)
            h = self.pool(h)
            return self.fc(h.view(inp.size(0), -1))

    conv_model = InplaceReLUConvNet().eval()
    img_1ch = torch.randn(1, 1, 8, 8)

    gradcam_expl = GradientExplainer(method="gradcam")
    res_cam = gradcam_expl.explain(conv_model, img_1ch, target_class=1)
    assert res_cam["status"] == "success"
    cam_arr = np.array(res_cam["attribution"])
    assert cam_arr.shape == (1, 1, 8, 8)

    # 3. Grad-CAM on Conv1d model
    class Conv1DNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(2, 4, kernel_size=3, padding=1)
            self.fc = nn.Linear(4 * 10, 2)

        def forward(self, inp):
            h = torch.relu(self.conv(inp))
            return self.fc(h.view(inp.size(0), -1))

    c1d_model = Conv1DNet().eval()
    seq_inp = torch.randn(1, 2, 10)
    res_cam_1d = gradcam_expl.explain(c1d_model, seq_inp, target_class=0)
    assert res_cam_1d["status"] == "success"
    cam_1d_arr = np.array(res_cam_1d["attribution"])
    assert cam_1d_arr.shape == (1, 1, 10)

    # 4. Integrated Gradients with background mean
    ig_expl = GradientExplainer(method="integrated_gradients")
    bg = torch.zeros(10, 4)
    res_ig = ig_expl.explain(mlp, x, target_class=1, steps=20, background_inputs=bg)
    assert res_ig["status"] == "success"
    ig_arr = np.array(res_ig["attribution"])
    assert ig_arr.shape == (1, 4)
