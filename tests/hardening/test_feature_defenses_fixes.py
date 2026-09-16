"""
Unit tests for bug fixes and edge-case behaviors in Feature Denoising and Feature Alignment defenses.
"""

import pytest
import torch
import torch.nn as nn
from app.hardening.defenses.feature import (
    FeatureDenoisingDefense,
    FeatureAlignmentDefense,
    denoise_features,
)
from app.hardening.defenses.feature.feature_denoising import FeatureDenoisedModelWrapper
from app.hardening.defenses.feature.feature_alignment import FeatureAlignmentModelWrapper
from app.hardening.exceptions import HardeningConfigurationError


class MultiLayerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class LinearOnlyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 15)
        self.fc2 = nn.Linear(15, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ============================================================================
# FEATURE DENOISING DEFENSE BUG FIX TESTS
# ============================================================================

def test_feature_denoising_target_layer_not_found_raises():
    """Verify that a typo in target_layer raises HardeningConfigurationError, preventing silent no-op."""
    model = MultiLayerCNN()
    defense = FeatureDenoisingDefense(target_layer="non_existent_layer_name")
    with pytest.raises(HardeningConfigurationError, match="Target layer 'non_existent_layer_name' not found"):
        defense.apply(model=model)


def test_feature_denoising_auto_detect_picks_mid_to_late_layer():
    """Verify auto-detect picks mid-to-late layer (e.g. conv2 or penultimate linear), not the very first layer."""
    model = MultiLayerCNN()
    defense = FeatureDenoisingDefense()
    res = defense.apply(model=model)
    assert res.success is True
    # Should pick conv2 (the last Conv2d layer before pool/classifier)
    assert res.metadata.parameters["target_layer"] == "conv2"
    res.hardened_model.remove_hook()

    # Linear-only model
    lin_model = LinearOnlyNet()
    lin_defense = FeatureDenoisingDefense()
    lin_res = lin_defense.apply(model=lin_model)
    assert lin_res.success is True
    # Should pick fc2 (penultimate Linear layer, representation before final logits)
    assert lin_res.metadata.parameters["target_layer"] == "fc2"
    lin_res.hardened_model.remove_hook()


def test_feature_denoising_small_feature_maps_not_skipped():
    """Verify small 4D feature maps (2x2 and 1x1) are smoothed rather than silently skipped."""
    # 2x2 feature map
    features_2x2 = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]])  # (1, 1, 2, 2) mean is 4.0
    denoised_2x2 = denoise_features(features_2x2, method="mean", strength=0.5)
    # Expected: (1 - 0.5) * feat + 0.5 * 4.0 -> not identical to raw
    assert not torch.allclose(features_2x2, denoised_2x2)
    assert denoised_2x2[0, 0, 0, 0] == pytest.approx(0.5 * 1.0 + 0.5 * 4.0)

    # 1x1 feature map with multiple channels
    features_1x1 = torch.tensor([[[[2.0]], [[6.0]]]])  # (1, 2, 1, 1) mean across channels is 4.0
    denoised_1x1 = denoise_features(features_1x1, method="mean", strength=0.5)
    assert not torch.allclose(features_1x1, denoised_1x1)
    assert denoised_1x1[0, 0, 0, 0] == pytest.approx(0.5 * 2.0 + 0.5 * 4.0)


def test_feature_denoising_per_sample_soft_threshold():
    """Verify soft_threshold computes thresholds per sample without cross-sample contamination."""
    # Sample 0 has small activations (mean 1.0); Sample 1 has huge activations (mean 100.0)
    sample_small = torch.ones(1, 4, 4, 4) * 1.0
    sample_large = torch.ones(1, 4, 4, 4) * 100.0
    batch = torch.cat([sample_small, sample_large], dim=0)  # (2, 4, 4, 4)

    # Denoise both individually and together
    out_small_alone = denoise_features(sample_small, method="soft_threshold", strength=0.2)
    out_batch = denoise_features(batch, method="soft_threshold", strength=0.2)

    # Sample 0's result when batched must match sample 0's result when isolated
    assert torch.allclose(out_batch[0:1], out_small_alone)


def test_feature_denoising_soft_threshold_gradient_flow():
    """Verify soft_threshold does NOT suffer from zero-gradient masking (passes non-zero gradients)."""
    features = torch.tensor([[-2.0, -0.5, 0.5, 2.0]], requires_grad=True)
    # strength 0.2, mean abs is (2 + 0.5 + 0.5 + 2)/4 = 1.25, threshold = 0.25
    denoised = denoise_features(features, method="soft_threshold", strength=0.2)
    loss = denoised.sum()
    loss.backward()

    # Gradients for entries with |x| > threshold must be non-zero (specifically ~1.0)
    assert features.grad is not None
    assert features.grad[0, 0].item() != 0.0  # x = -2.0
    assert features.grad[0, 3].item() != 0.0  # x = 2.0


# ============================================================================
# FEATURE ALIGNMENT DEFENSE BUG FIX TESTS
# ============================================================================

def test_feature_alignment_preserves_training_state():
    """Verify evaluate_consistency preserves and restores the base model's training flag."""
    model = MultiLayerCNN()
    model.train()
    assert model.training is True

    defense = FeatureAlignmentDefense()
    res = defense.apply(model=model)
    inputs = torch.rand(2, 3, 16, 16)

    # Calling evaluate_consistency must not leave model in eval mode
    res.hardened_model.evaluate_consistency(inputs)
    assert model.training is True


def test_feature_alignment_probe_clamping():
    """Verify probe values stay clamped within [clip_min, clip_max]."""
    model = MultiLayerCNN()
    inputs = torch.zeros(4, 3, 16, 16)  # at lower bound 0.0
    wrapper = FeatureAlignmentModelWrapper(
        base_model=model,
        noise_std=0.5,  # large noise
        clip_min=0.0,
        clip_max=1.0,
    )
    probe = wrapper._generate_probe(inputs)
    assert probe.min().item() >= 0.0
    assert probe.max().item() <= 1.0


def test_feature_alignment_alpha_weight_at_extremes():
    """Verify alpha = alignment_weight + (1 - alignment_weight) * sim honors extreme values."""
    model = MultiLayerCNN()
    inputs = torch.rand(2, 3, 16, 16)

    # If alignment_weight = 1.0, predictions must equal base model prediction (100% original trust)
    wrapper_trust_orig = FeatureAlignmentModelWrapper(
        base_model=model,
        alignment_weight=1.0,
    )
    wrapper_trust_orig.eval()
    model.eval()

    with torch.no_grad():
        out_defended = wrapper_trust_orig(inputs)
        out_base = torch.log_softmax(model(inputs), dim=-1)
        assert torch.allclose(out_defended, out_base, atol=1e-5)


def test_feature_alignment_configurable_threshold_and_multi_probe():
    """Verify configurable consistency_threshold and multi-probe variance reduction."""
    model = MultiLayerCNN()
    inputs = torch.rand(2, 3, 16, 16)

    # Configurable threshold
    defense = FeatureAlignmentDefense(consistency_threshold=0.9, num_probes=3)
    res = defense.apply(model=model, inputs=inputs)
    assert res.metadata.parameters["consistency_threshold"] == 0.9
    assert res.metadata.parameters["num_probes"] == 3

    cons = res.hardened_model.evaluate_consistency(inputs)
    # Check boolean tensor thresholding
    expected_is_consistent = cons["consistency_scores"] >= 0.9
    assert torch.equal(cons["is_consistent"], expected_is_consistent)
