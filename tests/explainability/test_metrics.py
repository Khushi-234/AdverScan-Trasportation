"""
Unit tests for XAI evaluation metrics (perturbation alignment, faithfulness, stability).
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from app.explainability.metrics import (
    compute_faithfulness,
    compute_perturbation_alignment,
    compute_stability,
)
from app.explainability.techniques.gradient_explainer import GradientExplainer


class SimpleClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]]))
            self.fc.bias.zero_()

    def forward(self, x):
        return self.fc(x)


def test_perturbation_alignment_metric():
    clean = np.array([1.0, 2.0, 3.0, 4.0])
    adv = np.array([1.0, 5.0, 3.0, 4.0])  # Perturbation on feature index 1
    attr = np.array([0.0, 1.0, 0.0, 0.0])  # Attribution strongly on feature index 1

    res = compute_perturbation_alignment(clean, adv, attr, top_k=1)
    assert res["alignment_status"] == "success"
    assert res["cosine_alignment"] > 0.9
    assert res["top_k_overlap"] == 1.0


def test_perturbation_alignment_unavailable():
    res = compute_perturbation_alignment(None, None, None)
    assert res["alignment_status"] == "unavailable"
    assert res["cosine_alignment"] is None


def test_faithfulness_metric():
    model = SimpleClassificationModel()
    inputs = torch.tensor([[5.0, 0.0, 0.0, 0.0]])  # High score for class 0
    # True attribution points to feature 0
    attribution = np.array([[10.0, 0.0, 0.0, 0.0]])

    res = compute_faithfulness(model, inputs, attribution, top_k=1, baseline_value=0.0)
    assert res["faithfulness_status"] == "success"
    assert res["faithfulness_score"] is not None
    assert res["faithfulness_score"] > 0.0  # Ablating feature 0 drops class 0 confidence
    assert res["confidence_drop"] > 0.0


def test_stability_metric():
    model = SimpleClassificationModel()
    inputs = torch.tensor([[1.0, 2.0, 0.5, 0.1]])
    explainer = GradientExplainer(method="saliency")

    res = compute_stability(explainer, model, inputs, noise_std=0.01, num_trials=2)
    assert res["stability_status"] == "success"
    assert res["mean_cosine_stability"] is not None
    assert res["mean_cosine_stability"] > 0.8
