"""
Comprehensive unit tests for all 6 hardening defense families in AdverScan:
- Preprocessing (Normalization, Gaussian, Median, Denoising, Squeezing)
- Smoothing (Randomized Smoothing)
- Detection (Confidence Rejection, Adversarial Detection)
- Transformation (Random Resize, Random Crop, Padding, Rotation, Translation)
- Feature-Level (Feature Denoising, Feature Alignment)
- Ensemble (Model Ensemble, Prediction Ensemble)
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.hardening.defenses import (
    BaseDefense,
    NormalizationDefense,
    GaussianFilterDefense,
    MedianFilterDefense,
    ImageDenoisingDefense,
    FeatureSqueezingDefense,
    RandomizedSmoothingDefense,
    ConfidenceRejectionDefense,
    AdversarialDetectionDefense,
    RandomResizeDefense,
    RandomCropDefense,
    PaddingDefense,
    RotationDefense,
    TranslationDefense,
    FeatureDenoisingDefense,
    FeatureAlignmentDefense,
    ModelEnsembleDefense,
    PredictionEnsembleDefense,
    PredictionEnsembleModelWrapper,
    get_defense_class,
    DEFENSE_REGISTRY,
)
from app.hardening.hardening_engine import HardeningEngine
from app.hardening.defense_selector import DefenseSelector
from app.hardening.exceptions import HardeningConfigurationError


class DummyConvNet(nn.Module):
    """Simple 2D CNN model for testing image defenses."""

    def __init__(self, in_channels: int = 3, num_classes: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(8 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.conv1(x))
        h = self.pool(h)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class DummyLinearNet(nn.Module):
    """Simple linear model for tabular/vector defenses."""

    def __init__(self, in_features: int = 16, num_classes: int = 2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


# ============================================================================
# 1. PREPROCESSING FAMILY TESTS
# ============================================================================

def test_normalization_defense():
    model = DummyConvNet()
    inputs = torch.rand(4, 3, 16, 16)
    orig_clone = inputs.clone()

    defense = NormalizationDefense(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], norm_type="mean_std")
    assert defense.requires_training is False
    assert "image" in defense.supported_domains

    result = defense.apply(model=model, inputs=inputs)
    assert result.success is True
    assert result.hardened_inputs is not None
    assert result.hardened_inputs.shape == inputs.shape
    # Ensure original tensor wasn't mutated
    assert torch.equal(inputs, orig_clone)

    # Test forward pass through wrapped model
    logits = result.hardened_model(inputs)
    assert logits.shape == (4, 3)

    # Test min-max norm
    defense_minmax = NormalizationDefense(norm_type="min_max")
    res_minmax = defense_minmax.apply(model=model, inputs=inputs)
    assert res_minmax.hardened_inputs.min() >= -1e-5
    assert res_minmax.hardened_inputs.max() <= 1.0 + 1e-5


def test_gaussian_filter_defense():
    model = DummyConvNet()
    inputs = torch.rand(2, 3, 16, 16)
    defense = GaussianFilterDefense(kernel_size=3, sigma=1.2)
    assert defense.requires_training is False

    result = defense.apply(model=model, inputs=inputs)
    assert result.success is True
    assert result.hardened_inputs.shape == inputs.shape
    logits = result.hardened_model(inputs)
    assert logits.shape == (2, 3)

    # Invalid kernel size validation
    with pytest.raises(HardeningConfigurationError):
        GaussianFilterDefense(kernel_size=4)


def test_median_filter_defense():
    model = DummyConvNet()
    # Add salt-and-pepper spikes
    inputs = torch.zeros(2, 3, 16, 16)
    inputs[:, :, 4, 4] = 10.0  # isolated spike

    defense = MedianFilterDefense(kernel_size=3)
    assert defense.requires_training is False
    result = defense.apply(model=model, inputs=inputs)
    assert result.success is True
    # Isolated spike should be eliminated by median filter
    assert result.hardened_inputs[:, :, 4, 4].max().item() < 10.0

    logits = result.hardened_model(inputs)
    assert logits.shape == (2, 3)


def test_image_denoising_defense():
    model = DummyConvNet()
    inputs = torch.rand(2, 3, 16, 16)

    for method in ("tv", "bilateral", "local_mean"):
        defense = ImageDenoisingDefense(method=method, strength=0.05, kernel_size=3)
        assert defense.requires_training is False
        result = defense.apply(model=model, inputs=inputs)
        assert result.success is True
        assert result.hardened_inputs.shape == inputs.shape
        logits = result.hardened_model(inputs)
        assert logits.shape == (2, 3)

    with pytest.raises(HardeningConfigurationError):
        ImageDenoisingDefense(method="unsupported_method")


def test_feature_squeezing_defense():
    model = DummyConvNet()
    inputs = torch.rand(3, 3, 16, 16)
    orig_clone = inputs.clone()

    defense = FeatureSqueezingDefense(bit_depth=3, clip_min=0.0, clip_max=1.0)
    assert defense.requires_training is False
    result = defense.apply(model=model, inputs=inputs)
    assert result.success is True
    assert result.hardened_inputs.shape == inputs.shape
    assert torch.equal(inputs, orig_clone)

    # Output values should belong to discrete 2^3 = 8 levels
    unique_vals = torch.unique(result.hardened_inputs)
    assert len(unique_vals) <= 8

    logits = result.hardened_model(inputs)
    assert logits.shape == (3, 3)


# ============================================================================
# 2. SMOOTHING FAMILY TESTS
# ============================================================================

def test_randomized_smoothing_defense():
    model = DummyConvNet()
    inputs = torch.rand(3, 3, 16, 16)

    defense = RandomizedSmoothingDefense(sigma=0.1, num_samples=5)
    assert defense.requires_training is False
    assert "smoothing" == defense.defense_family

    result = defense.apply(model=model, inputs=inputs)
    assert result.success is True
    # Verify metadata distinguishes empirical from certified
    assert result.metadata.parameters["is_certified"] is False
    assert result.metadata.parameters["smoothing_mode"] == "empirical"
    assert "certification_notice" in result.metadata.extra_metadata

    # Verify it uses the common HardenedModelWrapper / StochasticInferenceWrapper
    from app.hardening.defenses.wrapper import HardenedModelWrapper, StochasticInferenceWrapper
    assert isinstance(result.hardened_model, HardenedModelWrapper)
    assert isinstance(result.hardened_model, StochasticInferenceWrapper)

    # Check forward pass on smoothed wrapper
    logits = result.hardened_model(inputs)
    assert logits.shape == (3, 3)


def test_randomized_smoothing_clip_validation():
    # clip_min >= clip_max should raise HardeningConfigurationError
    with pytest.raises(HardeningConfigurationError, match="clip_min must be strictly less than clip_max"):
        RandomizedSmoothingDefense(clip_min=1.0, clip_max=0.5)

    with pytest.raises(HardeningConfigurationError, match="clip_min must be strictly less than clip_max"):
        RandomizedSmoothingDefense(clip_min=0.5, clip_max=0.5)

    with pytest.raises(HardeningConfigurationError, match="batch_size must be at least 1"):
        RandomizedSmoothingDefense(batch_size=0)


def test_randomized_smoothing_output_contract_and_batching():
    model = DummyConvNet()
    inputs = torch.rand(4, 3, 16, 16)

    # Test "prob" output contract: probabilities must sum to 1.0 and be in [0, 1]
    defense_prob = RandomizedSmoothingDefense(
        sigma=0.1,
        num_samples=8,
        batch_size=3,  # non-divisible batch size to test chunking
        output_type="prob",
    )
    result_prob = defense_prob.apply(model=model, inputs=inputs)
    smoothed_prob = result_prob.hardened_model
    smoothed_prob.eval()

    probs = smoothed_prob(inputs)
    assert probs.shape == (4, 3)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)

    # Test "log_prob" output contract
    defense_log = RandomizedSmoothingDefense(
        sigma=0.1,
        num_samples=6,
        batch_size=4,
        output_type="log_prob",
    )
    result_log = defense_log.apply(model=model, inputs=inputs)
    smoothed_log = result_log.hardened_model
    smoothed_log.eval()

    log_probs = smoothed_log(inputs)
    assert log_probs.shape == (4, 3)
    assert torch.all(log_probs <= 0.0 + 1e-6)  # log of probabilities in [0, 1] is <= 0

    # Test latency notice in metadata
    assert "latency_notice" in result_log.metadata.extra_metadata
    assert "setup_time_seconds" in result_log.metadata.parameters


def test_randomized_smoothing_training_mode_vs_eval_mode():
    model = DummyConvNet()
    inputs = torch.rand(2, 3, 16, 16, requires_grad=True)

    defense = RandomizedSmoothingDefense(sigma=0.1, num_samples=5, batch_size=4)
    result = defense.apply(model=model, inputs=inputs)
    smoothed_model = result.hardened_model

    # In training mode: gradients should flow back to model parameters
    smoothed_model.train()
    out_train = smoothed_model(inputs)
    loss = out_train.sum()
    loss.backward()

    # Verify gradients were computed on base_model parameters
    param = next(smoothed_model.base_model.parameters())
    assert param.grad is not None

    # In eval mode: inference executes under torch.no_grad()
    smoothed_model.eval()
    inputs_eval = torch.rand(2, 3, 16, 16, requires_grad=True)
    out_eval = smoothed_model(inputs_eval)
    assert out_eval.grad_fn is None  # no autograd graph built in eval mode



# ============================================================================
# 3. DETECTION FAMILY TESTS
# ============================================================================

def test_confidence_rejection_defense_and_coverage():
    model = DummyConvNet()
    inputs = torch.rand(10, 3, 16, 16)

    defense = ConfidenceRejectionDefense(threshold=0.7)
    assert defense.requires_training is False
    assert "detection" == defense.defense_family

    result = defense.apply(model=model, inputs=inputs)
    assert result.success is True
    meta = result.metadata.extra_metadata

    assert "coverage" in meta
    assert "rejection_rate" in meta
    assert meta["coverage"] + meta["rejection_rate"] == pytest.approx(1.0, abs=1e-5)
    assert meta["total_samples"] == 10
    assert meta["accepted_count"] + meta["rejected_count"] == 10

    # Predict with rejection API
    wrapper = result.hardened_model
    pred_res = wrapper.predict_with_rejection(inputs)
    assert "is_rejected" in pred_res
    assert len(pred_res["is_rejected"]) == 10


def test_adversarial_detection_strategies():
    model = DummyConvNet()
    inputs = torch.rand(6, 3, 16, 16)

    for method in ("sensitivity", "margin", "entropy"):
        detector = AdversarialDetectionDefense(threshold=0.5, method=method, num_samples=3)
        assert detector.requires_training is False

        result = detector.apply(model=model, inputs=inputs)
        assert result.success is True
        meta = result.metadata.extra_metadata
        assert meta["total_samples"] == 6
        assert "adversarial_detected_count" in meta
        assert "detection_mask" in meta
        assert len(meta["detection_scores"]) == 6

        # Check wrapped predict_with_detection
        det_res = result.hardened_model.predict_with_detection(inputs)
        assert "is_adversarial" in det_res
        assert "scores" in det_res


# ============================================================================
# 4. INPUT TRANSFORMATION FAMILY TESTS
# ============================================================================

def test_random_resize_defense():
    model = DummyConvNet()
    inputs = torch.rand(2, 3, 16, 16)
    orig_clone = inputs.clone()

    defense = RandomResizeDefense(min_scale=0.9, max_scale=1.1)
    assert defense.requires_training is False

    result = defense.apply(model=model, inputs=inputs)
    assert result.success is True
    assert result.hardened_inputs.shape == inputs.shape
    assert torch.equal(inputs, orig_clone)

    logits = result.hardened_model(inputs)
    assert logits.shape == (2, 3)


def test_random_crop_defense():
    model = DummyConvNet()
    inputs = torch.rand(2, 3, 16, 16)
    orig_clone = inputs.clone()

    defense = RandomCropDefense(pad_size=2, padding_mode="reflect")
    assert defense.requires_training is False

    result = defense.apply(model=model, inputs=inputs)
    assert result.success is True
    assert result.hardened_inputs.shape == inputs.shape
    assert torch.equal(inputs, orig_clone)

    logits = result.hardened_model(inputs)
    assert logits.shape == (2, 3)

    # Verify that crops are independent across batch items
    # Create a batch of identical patterned images
    identical_batch = torch.arange(16 * 16, dtype=torch.float32).view(1, 1, 16, 16).repeat(16, 1, 1, 1)
    crop_def = RandomCropDefense(pad_size=4)
    transformed_batch = crop_def.transform(identical_batch)
    assert transformed_batch.shape == (16, 1, 16, 16)
    # Not all samples should have identical crops if offsets are drawn independently
    diffs = [not torch.equal(transformed_batch[0], transformed_batch[i]) for i in range(1, 16)]
    assert any(diffs), "Expected stochastic variation across images in the batch"



def test_padding_defense():
    model = DummyConvNet()
    inputs = torch.rand(2, 3, 16, 16)
    orig_clone = inputs.clone()

    defense = PaddingDefense(pad_size=3, padding_mode="reflect")
    assert defense.requires_training is False

    result = defense.apply(model=model, inputs=inputs)
    assert result.success is True
    assert result.hardened_inputs.shape == inputs.shape
    assert torch.equal(inputs, orig_clone)

    logits = result.hardened_model(inputs)
    assert logits.shape == (2, 3)


def test_rotation_defense():
    model = DummyConvNet()
    inputs = torch.rand(2, 3, 16, 16)
    orig_clone = inputs.clone()

    defense = RotationDefense(max_angle=12.0)
    assert defense.requires_training is False

    result = defense.apply(model=model, inputs=inputs)
    assert result.success is True
    assert result.hardened_inputs.shape == inputs.shape
    assert torch.equal(inputs, orig_clone)

    logits = result.hardened_model(inputs)
    assert logits.shape == (2, 3)


def test_translation_defense():
    model = DummyConvNet()
    inputs = torch.rand(2, 3, 16, 16)
    orig_clone = inputs.clone()

    defense = TranslationDefense(max_dx=0.1, max_dy=0.1)
    assert defense.requires_training is False

    result = defense.apply(model=model, inputs=inputs)
    assert result.success is True
    assert result.hardened_inputs.shape == inputs.shape
    assert torch.equal(inputs, orig_clone)

    logits = result.hardened_model(inputs)
    assert logits.shape == (2, 3)


# ============================================================================
# 5. FEATURE-LEVEL DEFENSES TESTS
# ============================================================================

def test_feature_denoising_defense():
    model = DummyConvNet()
    inputs = torch.rand(2, 3, 16, 16)

    defense = FeatureDenoisingDefense(method="mean", strength=0.2)
    assert defense.requires_training is False

    result = defense.apply(model=model, inputs=inputs)
    assert result.success is True
    assert result.hardened_inputs.shape == inputs.shape

    logits = result.hardened_model(inputs)
    assert logits.shape == (2, 3)

    # Detach hook cleanly
    result.hardened_model.remove_hook()


def test_feature_alignment_defense():
    model = DummyConvNet()
    inputs = torch.rand(4, 3, 16, 16)

    defense = FeatureAlignmentDefense(noise_std=0.03, alignment_weight=0.6)
    assert defense.requires_training is False

    result = defense.apply(model=model, inputs=inputs)
    assert result.success is True
    assert "mean_consistency_score" in result.metadata.extra_metadata

    logits = result.hardened_model(inputs)
    assert logits.shape == (4, 3)

    # Evaluate consistency API
    cons_info = result.hardened_model.evaluate_consistency(inputs)
    assert "consistency_scores" in cons_info
    assert "mean_consistency" in cons_info


# ============================================================================
# 6. ENSEMBLE DEFENSES TESTS
# ============================================================================

def test_model_ensemble_defense():
    model1 = DummyConvNet(num_classes=3)
    model2 = DummyConvNet(num_classes=3)
    inputs = torch.rand(3, 3, 16, 16)

    for agg in ("mean", "soft_voting", "hard_voting", "median"):
        defense = ModelEnsembleDefense(models=[model1, model2], aggregation=agg)
        assert defense.requires_training is False

        result = defense.apply(model=model1, inputs=inputs)
        assert result.success is True
        assert result.metadata.parameters["num_models"] == 2

        logits = result.hardened_model(inputs)
        assert logits.shape == (3, 3)


def test_model_ensemble_hard_voting_distribution():
    # Construct models with deterministic class outputs to verify vote share calculation
    class FixedPredictor(nn.Module):
        def __init__(self, pred_class: int, num_classes: int = 3):
            super().__init__()
            self.pred_class = pred_class
            self.num_classes = num_classes

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b = x.size(0)
            logits = torch.zeros(b, self.num_classes)
            logits[:, self.pred_class] = 10.0
            return logits

    # 3 models: 2 vote for class 0, 1 votes for class 1
    m1 = FixedPredictor(pred_class=0)
    m2 = FixedPredictor(pred_class=0)
    m3 = FixedPredictor(pred_class=1)

    inputs = torch.rand(2, 3, 8, 8)
    defense = ModelEnsembleDefense(models=[m2, m3], aggregation="hard_voting")
    result = defense.apply(model=m1, inputs=inputs)
    out = result.hardened_model(inputs)

    # Softmax of hard voting output should accurately reflect vote proportions: 2/3, 1/3, 0
    probs = F.softmax(out, dim=-1)
    expected_probs = torch.tensor([[2.0 / 3.0, 1.0 / 3.0, 0.0], [2.0 / 3.0, 1.0 / 3.0, 0.0]])
    assert torch.allclose(probs, expected_probs, atol=1e-4)

    # Argmax should select majority class 0
    assert torch.equal(torch.argmax(out, dim=-1), torch.zeros(2, dtype=torch.long))


def test_model_ensemble_weight_validation():
    m1 = DummyConvNet(num_classes=2)
    m2 = DummyConvNet(num_classes=2)

    # Mismatched weights count
    defense_bad_count = ModelEnsembleDefense(models=[m2], weights=[0.5, 0.3, 0.2])
    with pytest.raises(HardeningConfigurationError, match="Number of weights .* must match number of ensemble models"):
        defense_bad_count.apply(model=m1)

    # Negative weights
    with pytest.raises(HardeningConfigurationError, match="weights must be non-negative"):
        ModelEnsembleDefense(weights=[-0.5, 1.5])

    # Zero sum weights
    with pytest.raises(HardeningConfigurationError, match="Sum of ensemble weights must be strictly positive"):
        ModelEnsembleDefense(weights=[0.0, 0.0])


def test_model_ensemble_binary_classification():
    class BinaryModel(nn.Module):
        def __init__(self, offset: float):
            super().__init__()
            self.offset = offset

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Returns (B, 1) logits
            return torch.full((x.size(0), 1), self.offset)

    bm1 = BinaryModel(offset=1.5)   # positive
    bm2 = BinaryModel(offset=-1.0)  # negative
    bm3 = BinaryModel(offset=2.0)   # positive

    inputs = torch.rand(4, 5)

    for agg in ("soft_voting", "hard_voting", "median", "mean"):
        defense = ModelEnsembleDefense(models=[bm2, bm3], aggregation=agg)
        result = defense.apply(model=bm1, inputs=inputs)
        out = result.hardened_model(inputs)

        # Shape should be preserved as (B, 1)
        assert out.shape == (4, 1)
        # 2 positive models vs 1 negative model -> overall positive logit (> 0)
        assert torch.all(out > 0.0)



def test_prediction_ensemble_defense():
    model = DummyConvNet(num_classes=3)
    inputs = torch.rand(3, 3, 16, 16)

    defense = PredictionEnsembleDefense(voting="soft", num_views=3)
    assert defense.requires_training is False

    result = defense.apply(model=model, inputs=inputs)
    assert result.success is True
    assert result.metadata.parameters["num_views"] == 3

    logits = result.hardened_model(inputs)
    assert logits.shape == (3, 3)

    pred_meta = result.hardened_model.predict_with_metadata(inputs)
    assert "consensus_rate" in pred_meta
    assert "mean_consensus" in pred_meta


def test_prediction_ensemble_num_views_and_validation():
    # num_views=5 generates exactly 5 views
    defense_5 = PredictionEnsembleDefense(num_views=5)
    transforms_5 = defense_5._build_default_transforms()
    assert len(transforms_5) == 5

    # num_views=0 raises HardeningConfigurationError
    with pytest.raises(HardeningConfigurationError, match="num_views must be at least 1"):
        PredictionEnsembleDefense(num_views=0)

    # clip_min >= clip_max raises HardeningConfigurationError
    with pytest.raises(HardeningConfigurationError, match="clip_min .* must be strictly less than clip_max"):
        PredictionEnsembleDefense(clip_min=1.0, clip_max=0.5)


def test_prediction_ensemble_binary_and_hard_voting():
    class BinaryModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Returns (B, 1) logits
            return torch.full((x.size(0), 1), 1.5)

    bm = BinaryModel()
    inputs = torch.rand(4, 5)

    # Test hard voting with binary model preserves (B, 1) shape
    defense_hard = PredictionEnsembleDefense(voting="hard", num_views=3)
    res_hard = defense_hard.apply(model=bm, inputs=inputs)
    out_hard = res_hard.hardened_model(inputs)
    assert out_hard.shape == (4, 1)
    assert torch.all(out_hard > 0.0)

    # Test predict_with_metadata on binary model
    meta = res_hard.hardened_model.predict_with_metadata(inputs)
    assert meta["consensus_rate"].shape == (4,)
    assert meta["mean_consensus"] == 1.0


def test_prediction_ensemble_standardized_negative_inputs():
    # Input with standardized negative values (e.g. ImageNet normalization: -2.0)
    std_inputs = torch.full((2, 3, 8, 8), -2.0)
    # Defense configured for standardized inputs (clip_min=None, clip_max=None)
    defense = PredictionEnsembleDefense(clip_min=None, clip_max=None, num_views=3)
    transforms = defense._build_default_transforms()
    noisy_view = transforms[1](std_inputs)
    # Values should remain around -2.0 without being clamped to [0, 1]
    assert torch.all(noisy_view < 0.0)


def test_prediction_ensemble_metadata_no_redundant_compute():
    call_counts = {"count": 0}

    def counting_transform(x: torch.Tensor) -> torch.Tensor:
        call_counts["count"] += 1
        return x

    model = DummyConvNet(num_classes=3)
    wrapper = PredictionEnsembleModelWrapper(
        base_model=model,
        transforms=[counting_transform, counting_transform],
        voting="soft",
    )

    inputs = torch.rand(2, 3, 8, 8)
    # predict_with_metadata should call each transform once (2 calls total), NOT 4 calls
    _ = wrapper.predict_with_metadata(inputs)
    assert call_counts["count"] == 2



# ============================================================================
# 7. REGISTRY & ENGINE INTEGRATION TESTS
# ============================================================================

def test_all_defenses_registered_and_metadata():
    expected_keys = [
        "normalization",
        "gaussian_filter",
        "median_filter",
        "image_denoising",
        "feature_squeezing",
        "randomized_smoothing",
        "confidence_rejection",
        "adversarial_detection",
        "random_resize",
        "random_crop",
        "padding",
        "rotation",
        "translation",
        "feature_denoising",
        "feature_alignment",
        "model_ensemble",
        "prediction_ensemble",
    ]

    for key in expected_keys:
        defense_cls = get_defense_class(key)
        assert issubclass(defense_cls, BaseDefense)
        inst = defense_cls()
        assert inst.requires_training is False
        assert inst.requires_retraining is False
        meta = inst.get_metadata()
        assert meta["name"] == key or key in DEFENSE_REGISTRY
        assert isinstance(meta["supported_domains"], list)
        assert len(meta["supported_domains"]) > 0


def test_hardening_engine_with_new_defenses():
    engine = HardeningEngine()
    model = DummyConvNet()
    inputs = torch.rand(2, 3, 16, 16)

    test_defense_keys = [
        ("normalization", {"norm_type": "min_max"}),
        ("gaussian_filter", {"kernel_size": 3, "sigma": 1.0}),
        ("median_filter", {"kernel_size": 3}),
        ("random_resize", {"min_scale": 0.9, "max_scale": 1.1}),
        ("feature_denoising", {"method": "mean", "strength": 0.2}),
        ("prediction_ensemble", {"voting": "soft", "num_views": 2}),
    ]

    for def_name, cfg in test_defense_keys:
        res = engine.harden(
            model=model,
            defense=def_name,
            inputs=inputs,
            defense_config=cfg,
        )
        assert res.success is True
        assert res.hardened_model is not None
        preds = res.hardened_model(inputs)
        assert preds.shape == (2, 3)
