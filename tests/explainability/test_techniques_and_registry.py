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


class DummyTreeModel:
    """Mock tree model with estimators_ attribute like RandomForest or XGBoost."""
    def __init__(self):
        self.estimators_ = [object()]


def test_select_by_mode_tree_model_selects_tree_shap():
    """Verify tree model automatically selects tree_shap across modes."""
    from app.explainability.techniques.registry import select_by_mode
    from app.explainability.techniques.tree_shap_explainer import TreeSHAPExplainer

    tree_model = DummyTreeModel()
    explainer, decision = select_by_mode("standard", tree_model, "tabular")

    assert isinstance(explainer, TreeSHAPExplainer)
    assert decision["technique_selected"] == "tree_shap"
    assert decision["requested_mode"] == "standard"
    assert decision["model_type"] == "tree (XGB/RF)"
    assert decision["modality"] == "tabular"
    assert "tree_shap" in decision["selection_reason"]


def test_select_by_mode_incompatible_override_raises_error():
    """Verify incompatible override raises a clear ValueError without switching silently."""
    from app.explainability.techniques.registry import select_by_mode

    tree_model = DummyTreeModel()
    linear_model = LinearModel()

    # Saliency on tree model -> error
    with pytest.raises(ValueError, match="incompatible with tree"):
        select_by_mode("standard", tree_model, "tabular", override="saliency")

    # GradCAM on tree model -> error
    with pytest.raises(ValueError, match="incompatible with tree"):
        select_by_mode("standard", tree_model, "tabular", override="gradcam")

    # GradCAM on tabular MLP -> error
    with pytest.raises(ValueError, match="requires convolutional layers"):
        select_by_mode("standard", linear_model, "tabular", override="gradcam")

    # TreeSHAP on non-tree model -> error
    with pytest.raises(ValueError, match="requires tree-based models"):
        select_by_mode("standard", linear_model, "tabular", override="tree_shap")


def test_select_by_mode_rules_table_and_fallback():
    """Verify rules table mapping across architectures and fallback for non-existent techniques."""
    from app.explainability.techniques.registry import select_by_mode

    conv_model = ConvModel()
    linear_model = LinearModel()

    # CNN fast -> gradcam
    _, cnn_fast_decision = select_by_mode("fast", conv_model, "image")
    assert cnn_fast_decision["technique_selected"] == "gradcam"
    assert cnn_fast_decision["model_type"] == "CNN"

    # CNN standard -> integrated_gradients
    _, cnn_std_decision = select_by_mode("standard", conv_model, "image")
    assert cnn_std_decision["technique_selected"] == "integrated_gradients"

    # Torch MLP fast -> saliency
    _, mlp_fast_decision = select_by_mode("fast", linear_model, "tabular")
    assert mlp_fast_decision["technique_selected"] == "saliency"

    # Torch MLP thorough -> gradient_shap does not exist -> fall back to integrated_gradients
    _, mlp_thorough_decision = select_by_mode("thorough", linear_model, "tabular")
    assert mlp_thorough_decision["technique_selected"] == "integrated_gradients"
    assert "does not exist yet" in mlp_thorough_decision["selection_reason"]

