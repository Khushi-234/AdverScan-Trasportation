"""
Unit tests for SHAPExplainer technique in M6.
"""

import numpy as np
import torch
import torch.nn as nn
from app.explainability.techniques.shap_explainer import SHAPExplainer


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


def test_shap_explainer_initialization():
    explainer = SHAPExplainer()
    assert hasattr(explainer, "is_available")
    assert hasattr(explainer, "explain")


def test_shap_explainer_explain_returns_structured_dict():
    model = SimpleModel()
    inputs = torch.randn(1, 4)

    explainer = SHAPExplainer()
    res = explainer.explain(model, inputs)

    assert isinstance(res, dict)
    assert "status" in res
    assert "executed" in res
    assert "technique" in res
    assert res["technique"] == "shap"

    if not explainer.is_available:
        assert res["status"] == "unavailable"
        assert res["executed"] is False
        assert res["attribution"] is None
        assert "not installed" in res["message"].lower()
    else:
        assert res["status"] in ("success", "error")
