"""
Unit tests for ExplanationCache and cache_key in AdverScan XAI module.
Verifies deterministic cache key generation, in-memory caching, and that
clean-input explanations are executed exactly once across multiple attacks on the same sample.
"""

from unittest.mock import patch
import numpy as np
import pytest
import torch
import torch.nn as nn

from app.explainability.cache import ExplanationCache, cache_key
from app.explainability.explainer import XAIExplainer
import app.explainability.explainer as explainer_mod


class SimpleModel(nn.Module):
    """Simple 2-class linear model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        with torch.no_grad():
            self.fc.weight.fill_(0.5)
            self.fc.bias.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_cache_key_deterministic_and_unique():
    """Verify cache_key produces deterministic strings and respects differing params."""
    x = torch.randn(1, 4)
    k1 = cache_key(x, "saliency", 0, "model_a")
    k2 = cache_key(x, "saliency", 0, "model_a")
    assert k1 == k2
    assert isinstance(k1, str)
    assert len(k1) > 0

    # Differing technique
    k_diff_tech = cache_key(x, "integrated_gradients", 0, "model_a")
    assert k1 != k_diff_tech

    # Differing target class
    k_diff_class = cache_key(x, "saliency", 1, "model_a")
    assert k1 != k_diff_class

    # Differing model_id
    k_diff_model = cache_key(x, "saliency", 0, "model_b")
    assert k1 != k_diff_model

    # Differing input
    x_other = x + 1.0
    k_diff_x = cache_key(x_other, "saliency", 0, "model_a")
    assert k1 != k_diff_x


def test_cache_key_modalities():
    """Verify cache_key works for numpy arrays, strings, and lists."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    k_np = cache_key(arr, "shap", 0, "m")
    assert isinstance(k_np, str)

    k_str = cache_key("sample text", "layer_ig", 0, "m")
    assert isinstance(k_str, str)

    k_list = cache_key(["token1", "token2"], "saliency", 0, "m")
    assert isinstance(k_list, str)


def test_explanation_cache_basic_ops():
    """Verify get, put, clear, len, and contains for ExplanationCache."""
    cache = ExplanationCache()
    assert len(cache) == 0
    assert cache.get("missing_key") is None
    assert "missing_key" not in cache

    cache.put("k1", {"result": 42})
    assert len(cache) == 1
    assert "k1" in cache
    assert cache.get("k1") == {"result": 42}

    cache.clear()
    assert len(cache) == 0
    assert cache.get("k1") is None


def test_clean_input_explained_once_across_three_attacks():
    """
    Verify that 3 attacks on the same sample explain the clean input only once
    by counting underlying profiling/explainer calls.
    """
    model = SimpleModel()
    model.eval()

    x_clean = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    x_adv1 = torch.tensor([[1.1, 2.1, 3.1, 4.1]])
    x_adv2 = torch.tensor([[1.2, 2.2, 3.2, 4.2]])
    x_adv3 = torch.tensor([[1.3, 2.3, 3.3, 4.3]])

    explainer = XAIExplainer(default_technique="saliency")
    clean_call_count = 0
    adv_call_count = 0

    original_profile_call = explainer_mod.profile_explainer_call

    def counting_profile_call(active_expl, m, inputs, *args, **kwargs):
        nonlocal clean_call_count, adv_call_count
        # Check if the input tensor matches clean input
        if torch.is_tensor(inputs) and torch.allclose(inputs, x_clean):
            clean_call_count += 1
        else:
            adv_call_count += 1
        return original_profile_call(active_expl, m, inputs, *args, **kwargs)

    with patch.object(explainer_mod, "profile_explainer_call", side_effect=counting_profile_call):
        res1 = explainer.explain(
            model=model,
            clean_input=x_clean,
            adversarial_input=x_adv1,
            attack_name="attack_fgsm",
            target_class=0,
            modality="tabular",
        )
        res2 = explainer.explain(
            model=model,
            clean_input=x_clean,
            adversarial_input=x_adv2,
            attack_name="attack_pgd",
            target_class=0,
            modality="tabular",
        )
        res3 = explainer.explain(
            model=model,
            clean_input=x_clean,
            adversarial_input=x_adv3,
            attack_name="attack_deepfool",
            target_class=0,
            modality="tabular",
        )

    # Clean input must be explained exactly ONCE
    assert clean_call_count == 1, f"Expected clean input to be explained once, but was called {clean_call_count} times"
    # Adversarial inputs must be explained for each of the 3 attacks
    assert adv_call_count == 3, f"Expected 3 adversarial calls, but was called {adv_call_count} times"

    # All 3 results must have valid attributions
    assert res1.attribution_status in ("success", "ok")
    assert res2.attribution_status in ("success", "ok")
    assert res3.attribution_status in ("success", "ok")
    assert len(explainer.cache) == 1
