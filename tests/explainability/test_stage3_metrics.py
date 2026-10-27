"""
Tests for Stage 3 fixes: Metrics and comparison (E3.1 - E3.4).
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from app.explainability.comparison import compare_attributions, compare_explanations
from app.explainability.explainer import XAIExplainer
from app.explainability.metrics.faithfulness import compute_faithfulness
from app.explainability.metrics.perturbation_alignment import compute_perturbation_alignment
from app.explainability.metrics.stability import compute_stability
from app.explainability.techniques.gradient_explainer import GradientExplainer


def test_e3_1_perturbation_alignment_spatial_and_metrics():
    """
    Test E3.1:
    - Alignment aligns spatial dimensions between Grad-CAM (1, 1, H, W) and image (1, 3, H, W).
    - Computes Spearman correlation and budget_fraction.
    - Rejects incompatible shapes with shape_mismatch status instead of silent min_len truncation.
    """
    # 1. Image and Grad-CAM spatial alignment
    clean = np.zeros((1, 3, 10, 10), dtype=np.float32)
    adv = np.zeros((1, 3, 10, 10), dtype=np.float32)
    # Attack perturbed the center 2x2 pixels
    adv[0, :, 4:6, 4:6] = 2.0
    gradcam_attr = np.zeros((1, 1, 10, 10), dtype=np.float32)
    gradcam_attr[0, 0, 4:6, 4:6] = 1.0  # clean model also focused on center

    res = compute_perturbation_alignment(clean, adv, gradcam_attr, top_k=4)
    assert res["alignment_status"] == "success"
    assert res["cosine_alignment"] > 0.9
    assert res["top_k_overlap"] > 0.9
    assert res["budget_fraction"] > 0.9  # Perturbation budget fell almost entirely on top-4 features
    assert "spearman_correlation" in res

    # 2. Percentage top_k for large feature spaces
    res_pct = compute_perturbation_alignment(clean, adv, gradcam_attr, top_k=0.05)
    assert res_pct["alignment_status"] == "success"
    assert res_pct["k_used"] == 5  # 5% of 100 pixels

    # 3. Shape mismatch returns shape_mismatch, NOT silent truncation
    bad_attr = np.ones((5, 5))
    res_bad = compute_perturbation_alignment(clean, adv, bad_attr)
    assert res_bad["alignment_status"] == "shape_mismatch"


def test_e3_2_faithfulness_curves_auc_and_negative_drop():
    """
    Test E3.2:
    - compute_faithfulness works on non-contiguous tensors.
    - Computes deletion curve, insertion curve, and random ablation AUC.
    - Does NOT clip negative drops.
    """
    # Model that positively weights feature 0, but negatively weights feature 1
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2, bias=False)
            with torch.no_grad():
                self.fc.weight.copy_(torch.tensor([[2.0, -2.0, 0.0, 0.0], [-2.0, 2.0, 0.0, 0.0]]))

        def forward(self, x):
            return self.fc(x)

    model = DummyModel().eval()
    inp = torch.tensor([[1.0, 1.0, 0.5, 0.5]])  # non-contiguous slice or tensor
    inp_noncontig = inp.t().t()

    # Attribution identifying feature 0 as most important
    attr = np.array([1.0, 0.1, 0.0, 0.0])
    bg = np.zeros((10, 4), dtype=np.float32)

    res = compute_faithfulness(model, inp_noncontig, attr, top_k=1, background_inputs=bg)
    assert res["faithfulness_status"] == "success"
    assert res["deletion_auc"] is not None
    assert res["insertion_auc"] is not None
    assert res["random_deletion_auc"] is not None
    assert res["confidence_drop"] > 0.0

    # Test an unfaithful attribution that increases target confidence when ablated
    # Ablating feature 1 (negative weight) will INCREASE class 0 confidence -> negative drop
    unfaithful_attr = np.array([0.0, 1.0, 0.0, 0.0])
    res_unf = compute_faithfulness(model, inp, unfaithful_attr, top_k=1, background_inputs=bg)
    assert res_unf["faithfulness_status"] == "success"
    # Negative score must be preserved to reveal unfaithful explanation!
    assert res_unf["confidence_drop"] < 0.0
    assert res_unf["faithfulness_score"] < 0.0


def test_e3_3_stability_fixed_target_and_sampling_variance():
    """
    Test E3.3:
    - Fixes target class across base and noisy trials.
    - Reports sampling variance and input sensitivity separately.
    """
    class LinearModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)
            with torch.no_grad():
                self.fc.weight.copy_(torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]))
                self.fc.bias.zero_()

        def forward(self, x):
            return self.fc(x)

    model = LinearModel().eval()
    explainer = GradientExplainer(method="saliency")
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    res = compute_stability(explainer, model, x, target_class=0, noise_std=0.01, num_trials=3)
    assert res["stability_status"] == "success"
    assert res["target_class"] == 0
    assert res["mean_cosine_stability"] > 0.99
    # Saliency is deterministic, so sampling variance must be ~ 0.0
    assert res["sampling_variance"] < 1e-6


def test_e3_4_comparison_scale_invariance_and_clean_drop():
    """
    Test E3.4:
    - compare_explanations reports clean_class_confidence_drop and logit_margin.
    - compare_attributions computes normalized_attribution_l1 and spearman correlation.
    """
    # 1. compare_explanations
    clean_probs = np.array([0.9, 0.1])
    adv_probs = np.array([0.05, 0.95])  # Attack succeeded confidently

    res_comp = compare_explanations(
        clean_prediction=0,
        adversarial_prediction=1,
        clean_confidence=0.9,
        adversarial_confidence=0.95,  # Adv confidence is HIGH for the adversarial class!
        clean_probs=clean_probs,
        adv_probs=adv_probs,
    )

    # confidence_difference was 0.9 - 0.95 = -0.05 (misleading negative value)
    assert res_comp["confidence_difference"] < 0.0
    # BUT clean_class_confidence_drop correctly measures drop on clean class (0.9 - 0.05 = 0.85)!
    assert np.isclose(res_comp["clean_class_confidence_drop"], 0.85)
    assert np.isclose(res_comp["logit_margin"], 0.8)

    # 2. compare_attributions scale-invariant L1
    attr_a = np.array([10.0, 20.0, 30.0])
    attr_b = np.array([100.0, 200.0, 300.0])  # Identical relative attribution, just scaled 10x
    res_attr = compare_attributions(attr_a, attr_b)

    assert res_attr["attribution_comparison_status"] == "success"
    # Normalized L1 should be 0 because relative distributions are identical
    assert np.isclose(res_attr["normalized_attribution_l1"], 0.0)
    assert np.isclose(res_attr["attribution_cosine_similarity"], 1.0)
    assert np.isclose(res_attr["attribution_spearman_correlation"], 1.0)


def test_e3_explainer_compute_metrics_wiring():
    """
    Test that explain(..., compute_metrics=True) generates and wires all metrics into comparison.
    """
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(x)

    model = SimpleMLP().eval()
    clean_x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    adv_x = np.array([[1.1, 1.9, 3.1, 3.9]], dtype=np.float32)

    explainer = XAIExplainer()
    res = explainer.explain(
        model=model,
        clean_input=clean_x,
        adversarial_input=adv_x,
        technique="saliency",
        compute_metrics=True,
    )

    assert res.comparison is not None
    assert "perturbation_alignment" in res.comparison
    assert "faithfulness" in res.comparison
    assert "stability" in res.comparison
    assert res.comparison["perturbation_alignment"]["alignment_status"] == "success"
    assert res.comparison["faithfulness"]["faithfulness_status"] == "success"
    assert res.comparison["stability"]["stability_status"] == "success"
