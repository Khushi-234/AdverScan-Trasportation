"""
Unit tests for STAGE 1: Prediction and confidence correctness across models and modalities.
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn

from app.attack_engine.models import AttackMetadata, AttackResult
from app.explainability.explainer import XAIExplainer
from app.explainability.explanation_result import ExplanationResult
from app.explainability.failure_analysis import is_prediction_correct
from app.explainability.input_utils import (
    extract_prediction_and_confidence,
    predict_proba_np,
    unwrap_model,
)
from app.explainability.modality import Modality, detect_modality


class DummyAdapter:
    """Mock model adapter mimicking M1 adapters."""
    def __init__(self, inner):
        self._model = inner

    def get_model(self):
        return self._model


class SigmoidBinaryMLP(nn.Module):
    """Binary classifier with single logit output."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[2.0, -1.0, 0.0, 0.0]]))
            self.fc.bias.copy_(torch.tensor([0.5]))

    def forward(self, x):
        return self.fc(x)


class SoftmaxMultiClassNet(nn.Module):
    """Network that already returns normalized softmax probabilities."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        logits = self.fc(x)
        return torch.softmax(logits, dim=-1)


class SingleOutputLSTM(nn.Module):
    """LSTM with single continuous output for sequential data."""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=4, batch_first=True)
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class MockAutoencoder(nn.Module):
    """Autoencoder that outputs reconstructed inputs."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 2)
        self.decoder = nn.Linear(2, 4)

    def forward(self, x):
        return self.decoder(torch.relu(self.encoder(x)))


def test_e1_1_and_e1_6_tree_model_predict_proba_and_shap():
    """Verify RandomForest uses predict_proba (not hard labels), confidence < 1.0, and TreeSHAP works."""
    X_train = np.random.randn(50, 4)
    y_train = np.random.randint(0, 2, size=50)
    rf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_train, y_train)
    adapted_rf = DummyAdapter(rf)

    clean_x = np.array([[2.0, 2.0, 2.0, 2.0]])
    adv_x = np.array([[-2.0, -2.0, -2.0, -2.0]])

    # 1. predict_proba_np unwraps adapter and returns calibrated probabilities
    probs = predict_proba_np(adapted_rf, clean_x)
    assert probs.shape == (1, 2)
    assert np.isclose(np.sum(probs), 1.0)
    assert 0.0 < probs[0, 0] < 1.0  # Not hard labels 0.0 or 1.0

    # 2. XAIExplainer routes to TreeSHAP and produces varying confidences
    explainer = XAIExplainer()
    res = explainer.explain(
        model=adapted_rf,
        clean_input=clean_x,
        adversarial_input=adv_x,
        technique="shap",
    )

    assert isinstance(res, ExplanationResult)
    assert res.technique in ("treeshap", "shap")
    assert res.clean_confidence < 1.0
    assert res.adversarial_confidence < 1.0
    assert res.clean_confidence != res.adversarial_confidence
    assert res.attribution["clean"]["attribution"] is not None


def test_e1_2_sigmoid_binary_mlp():
    """Verify single-output (N,1) models are mapped to [1-p, p], not constant 0 / 1.0."""
    model = SigmoidBinaryMLP()
    clean_x = torch.tensor([[2.0, 0.0, 0.0, 0.0]])  # logit = 2*2 + 0.5 = 4.5 -> p1 ~ 0.989
    adv_x = torch.tensor([[-2.0, 0.0, 0.0, 0.0]])  # logit = -2*2 + 0.5 = -3.5 -> p1 ~ 0.029

    probs_clean = predict_proba_np(model, clean_x)
    assert probs_clean.shape == (1, 2)
    assert probs_clean[0, 1] > 0.95
    assert probs_clean[0, 0] < 0.05

    probs_adv = predict_proba_np(model, adv_x)
    assert probs_adv[0, 1] < 0.05
    assert probs_adv[0, 0] > 0.95

    explainer = XAIExplainer(default_technique="saliency")
    res = explainer.explain(
        model=model,
        clean_input=clean_x,
        adversarial_input=adv_x,
        technique="saliency",
    )

    assert res.clean_prediction == 1
    assert res.adversarial_prediction == 0
    assert res.prediction_changed is True
    assert 0.95 <= res.clean_confidence <= 1.0
    assert 0.95 <= res.adversarial_confidence <= 1.0


def test_e1_3_softmax_not_applied_twice():
    """Verify already normalized softmax outputs do not undergo double softmax."""
    model = SoftmaxMultiClassNet()
    x = torch.randn(1, 4)

    raw_probs = model(x).detach().cpu().numpy()
    calibrated_probs = predict_proba_np(model, x)

    # If softmax were applied twice, max prob would decrease (e.g. 0.9 -> 0.6)
    assert np.allclose(raw_probs, calibrated_probs, atol=1e-5)


def test_e1_4_single_output_lstm_and_autoencoder():
    """Verify single-output LSTM (regression) and Autoencoder (anomaly_score) run without crashing."""
    # 1. Regression mode with LSTM
    lstm_model = SingleOutputLSTM()
    clean_ts = torch.randn(1, 10, 3)
    adv_ts = clean_ts + 0.2

    explainer = XAIExplainer(default_technique="integrated_gradients")
    res_reg = explainer.explain(
        model=lstm_model,
        clean_input=clean_ts,
        adversarial_input=adv_ts,
        modality="timeseries",
        output_mode="regression",
        steps=5,
    )
    assert isinstance(res_reg, ExplanationResult)
    assert "relative_change" in res_reg.comparison

    # 2. Anomaly score mode with Autoencoder
    ae_model = MockAutoencoder()
    clean_tab = torch.randn(1, 4)
    adv_tab = clean_tab + 1.5

    res_ae = explainer.explain(
        model=ae_model,
        clean_input=clean_tab,
        adversarial_input=adv_tab,
        modality="tabular",
        output_mode="anomaly_score",
        technique="shap",
    )
    assert isinstance(res_ae, ExplanationResult)
    assert "relative_change" in res_ae.comparison


def test_e1_5_comparison_target_apples_to_apples():
    """Verify comparison_target='clean_pred' evaluates both attributions w.r.t the clean class."""
    model = SigmoidBinaryMLP()
    clean_x = torch.tensor([[3.0, 0.0, 0.0, 0.0]])  # Class 1
    adv_x = torch.tensor([[-3.0, 0.0, 0.0, 0.0]])   # Class 0

    explainer = XAIExplainer(default_technique="saliency")

    # Clean pred target
    res_clean_target = explainer.explain(
        model=model,
        clean_input=clean_x,
        adversarial_input=adv_x,
        comparison_target="clean_pred",
        technique="saliency",
    )
    assert res_clean_target.attribution["target_class_explained"] == res_clean_target.clean_prediction

    # 'both' target computes additional attribution for adv_pred
    res_both = explainer.explain(
        model=model,
        clean_input=clean_x,
        adversarial_input=adv_x,
        comparison_target="both",
        technique="saliency",
    )
    assert "adversarial_adv_class" in res_both.attribution


def test_e1_7_batch_handling_and_one_hot():
    """Verify explain() rejects N > 1, explain_batch() succeeds, and one-hot labels are parsed."""
    model = SigmoidBinaryMLP()
    clean_batch = torch.randn(4, 4)
    adv_batch = clean_batch + 0.1

    explainer = XAIExplainer(default_technique="saliency")

    # explain() rejects batch N=4
    with pytest.raises(ValueError, match="accepts exactly one sample"):
        explainer.explain(model=model, clean_input=clean_batch, adversarial_input=adv_batch)

    # explain_batch() processes N=4
    results = explainer.explain_batch(
        model=model,
        clean_inputs=clean_batch,
        adversarial_inputs=adv_batch,
        true_labels=[1, 0, 1, 0],
    )
    assert len(results) == 4
    for r in results:
        assert isinstance(r, ExplanationResult)

    # One-hot label handling in is_prediction_correct
    one_hot_label = np.array([0.0, 1.0])
    assert is_prediction_correct(prediction=1, true_label=one_hot_label) is True
    assert is_prediction_correct(prediction=0, true_label=one_hot_label) is False


def test_e1_8_timeseries_modality_routing():
    """Verify (32, 30, 3) with explicit modality='timeseries' routes to timeseries without image confusion."""
    t_seq = torch.randn(1, 30, 3)
    mod = detect_modality(t_seq, hint="timeseries")
    assert mod == Modality.TIMESERIES

    # Test that (32, 30, 3) with explicit hint is timeseries
    mod_batch = detect_modality(torch.randn(32, 30, 3), hint="timeseries")
    assert mod_batch == Modality.TIMESERIES
