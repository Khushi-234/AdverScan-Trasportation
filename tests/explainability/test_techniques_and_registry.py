"""
Unit tests for TechniqueRegistry and technique compatibility checks.
"""

import pytest
import torch
import torch.nn as nn

from app.explainability.modality import Modality
from app.explainability.techniques.base import BaseExplainer
from app.explainability.techniques.gradient_explainer import GradientExplainer
from app.explainability.techniques.lime_explainer import LIMEExplainer
from app.explainability.techniques.registry import TechniqueRegistry, technique_registry
from app.explainability.techniques.shap_explainer import SHAPExplainer


class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc(self.pool(self.conv(x)).view(x.size(0), -1))


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


def test_registry_listing_and_defaults():
    registry = TechniqueRegistry()
    techniques = registry.list_techniques()

    assert "shap" in techniques
    assert "lime" in techniques
    assert "saliency" in techniques
    assert "integrated_gradients" in techniques
    assert "gradcam" in techniques


def test_registry_get_explainer():
    registry = TechniqueRegistry()
    assert isinstance(registry.get("shap"), SHAPExplainer)
    assert isinstance(registry.get("lime"), LIMEExplainer)
    assert isinstance(registry.get("saliency"), GradientExplainer)
    assert isinstance(registry.get("integrated_gradients"), GradientExplainer)
    assert isinstance(registry.get("gradcam"), GradientExplainer)

    with pytest.raises(ValueError, match="Unsupported XAI technique"):
        registry.get("nonexistent_technique")


def test_registry_compatibility_checking():
    registry = TechniqueRegistry()
    conv_model = ConvModel()
    linear_model = LinearModel()

    # Image domain
    image_techs = registry.get_compatible_techniques("image", conv_model)
    assert "gradcam" in image_techs
    assert "saliency" in image_techs

    # Linear model without conv layers cannot use Grad-CAM
    linear_image_techs = registry.get_compatible_techniques("image", linear_model)
    assert "gradcam" not in linear_image_techs

    # Tabular domain
    tabular_techs = registry.get_compatible_techniques("tabular", linear_model)
    assert "shap" in tabular_techs
    assert "gradcam" not in tabular_techs


def test_registry_select_technique():
    registry = TechniqueRegistry()
    conv_model = ConvModel()
    linear_model = LinearModel()

    # Default selection for tabular
    explainer = registry.select_technique(Modality.TABULAR, model=linear_model)
    assert isinstance(explainer, BaseExplainer)

    # Valid preferred selection
    expl = registry.select_technique(Modality.IMAGE, model=conv_model, preferred="gradcam")
    assert isinstance(expl, GradientExplainer)
    assert expl.method == "gradcam"

    # Incompatible selection: gradcam on tabular data
    with pytest.raises(ValueError, match="Unsupported XAI technique 'gradcam'"):
        registry.select_technique(Modality.TABULAR, model=linear_model, preferred="gradcam")

    # Incompatible selection: gradcam on non-conv model
    with pytest.raises(ValueError, match="Unsupported XAI technique 'gradcam'"):
        registry.select_technique(Modality.IMAGE, model=linear_model, preferred="gradcam")
