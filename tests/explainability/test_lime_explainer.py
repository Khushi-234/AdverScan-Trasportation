"""
Unit tests for LIMEExplainer technique in M6.
"""

import numpy as np
import torch
import torch.nn as nn
from app.explainability.techniques.lime_explainer import LIMEExplainer


class SimpleVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        h = self.pool(self.conv(x)).view(x.size(0), -1)
        return self.fc(h)


def test_lime_explainer_initialization():
    explainer = LIMEExplainer()
    assert hasattr(explainer, "is_available")
    assert hasattr(explainer, "explain")


def test_lime_explainer_explain_returns_structured_dict():
    model = SimpleVisionModel()
    inputs = torch.randn(1, 3, 16, 16)

    explainer = LIMEExplainer()
    res = explainer.explain(model, inputs)

    assert isinstance(res, dict)
    assert "status" in res
    assert "executed" in res
    assert "technique" in res
    assert res["technique"] == "lime"

    if not explainer.is_available:
        assert res["status"] == "unavailable"
        assert res["executed"] is False
        assert res["attribution"] is None
        assert "not installed" in res["message"].lower()
    else:
        assert res["status"] in ("success", "error")
