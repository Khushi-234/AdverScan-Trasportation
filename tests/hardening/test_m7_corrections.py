"""
Comprehensive M7 Hardening & Defense Implementation Verification Tests.

Verifies:
1. PyTorch and Scikit-learn model compatibility.
2. All four modalities (Vision, Tabular, Text, Time-Series).
3. Parameter validation and strict constraint enforcement.
4. No-op defense detection and elimination.
5. Model training/eval state preservation and restoration.
6. Resource and forward hook cleanup in iterative hardening.
7. Unsupported combination rejection with informative diagnostics.
8. M7-to-M8 handoff contract and metadata preservation.
9. Frontend defense service integration with PyTorch and Sklearn models.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from app.hardening.hardening_engine import HardeningEngine, HardeningResult
from app.hardening.defense_selector import DefenseSelector
from app.hardening.defenses import (
    NormalizationDefense,
    GaussianFilterDefense,
    MedianFilterDefense,
    ImageDenoisingDefense,
    FeatureSqueezingDefense,
    RandomizedSmoothingDefense,
    ConfidenceRejectionDefense,
    AdversarialDetectionDefense,
    RotationDefense,
    RandomCropDefense,
    RandomResizeDefense,
    PaddingDefense,
    TranslationDefense,
    FeatureDenoisingDefense,
    FeatureAlignmentDefense,
    ModelEnsembleDefense,
    PredictionEnsembleDefense,
    AdversarialTrainingDefense,
    HardenedModelWrapper,
)
from app.hardening.exceptions import HardeningConfigurationError, DefenseExecutionError
from app.retest.threshold import RetestThresholds
from frontend.services.defense_service import (
    execute_configured_defense,
    get_available_defenses_for_domain,
)


# Simple PyTorch test models
class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 3)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class SimpleMLP(nn.Module):
    def __init__(self, in_features=4, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 8)
        self.fc2 = nn.Linear(8, num_classes)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# ============================================================================
# 1. SCIKIT-LEARN & PYTORCH COMPATIBILITY
# ============================================================================

def test_sklearn_model_wrapped_with_preprocessing_defense():
    """Verify Scikit-learn estimator can be hardened, providing predict, predict_proba, and classes_."""
    X_train = np.random.randn(20, 4).astype(np.float32)
    y_train = np.random.randint(0, 2, size=20)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    defense = FeatureSqueezingDefense(bit_depth=4)
    result = defense.apply(model=clf)

    assert result.success is True
    wrapped = result.hardened_model
    assert isinstance(wrapped, HardenedModelWrapper)

    # Scikit-learn predict & predict_proba interfaces
    preds = wrapped.predict(X_train)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (20,)

    probs = wrapped.predict_proba(X_train)
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (20, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    # classes_ attribute forwarding
    assert np.array_equal(wrapped.classes_, clf.classes_)

    # PyTorch callable interface for downstream evaluation harnesses
    tensor_inputs = torch.from_numpy(X_train).float()
    tensor_outputs = wrapped(tensor_inputs)
    assert isinstance(tensor_outputs, torch.Tensor)
    assert tensor_outputs.shape == (20, 2)


def test_hardening_engine_supports_sklearn_model():
    """Verify HardeningEngine accepts Scikit-learn estimators without PyTorch-only rejection."""
    X_train = np.random.randn(30, 6).astype(np.float32)
    y_train = np.random.randint(0, 3, size=30)
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X_train, y_train)

    engine = HardeningEngine()
    test_inputs = torch.from_numpy(X_train[:10]).float()
    test_labels = torch.from_numpy(y_train[:10]).long()

    res = engine.harden(
        model=clf,
        defense="normalization",
        inputs=test_inputs,
        labels=test_labels,
        defense_config={"norm_type": "min_max"},
    )
    assert res.success is True
    assert res.hardened_model is not None


# ============================================================================
# 2. ALL FOUR MODALITIES
# ============================================================================

def test_modality_vision_4d():
    """Verify vision defenses operate correctly on 4D image tensors (B, C, H, W)."""
    model = SimpleConvNet()
    inputs = torch.rand(4, 3, 16, 16)

    defenses = [
        RotationDefense(max_angle=10.0),
        RandomCropDefense(pad_size=2),
        RandomResizeDefense(min_scale=0.9, max_scale=1.1),
        PaddingDefense(pad_size=2),
        TranslationDefense(max_dx=0.05, max_dy=0.05),
        GaussianFilterDefense(kernel_size=3, sigma=1.0),
        MedianFilterDefense(kernel_size=3),
        ImageDenoisingDefense(method="local_mean"),
    ]

    for d in defenses:
        res = d.apply(model=model, inputs=inputs)
        assert res.success is True
        assert res.hardened_inputs is not None
        assert res.hardened_inputs.shape == (4, 3, 16, 16)
        out = res.hardened_model(inputs)
        assert out.shape == (4, 3)


def test_modality_tabular_2d():
    """Verify tabular defenses operate correctly on 2D feature tensors (B, D)."""
    model = SimpleMLP(in_features=5, num_classes=2)
    inputs = torch.randn(8, 5)

    defenses = [
        NormalizationDefense(norm_type="min_max"),
        FeatureSqueezingDefense(bit_depth=4),
        RandomizedSmoothingDefense(sigma=0.05, num_samples=3, batch_size=4),
        ConfidenceRejectionDefense(threshold=0.6),
        FeatureAlignmentDefense(noise_std=0.02, alignment_weight=0.5),
    ]

    for d in defenses:
        res = d.apply(model=model, inputs=inputs)
        assert res.success is True
        assert res.hardened_model is not None
        out = res.hardened_model(inputs)
        assert out.shape[0] == 8


class ToyTextClassifier(nn.Module):
    def __init__(self, vocab_size=3000, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 8)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        return self.fc(self.emb(x).mean(dim=1))


def test_modality_text_discrete_tokens():
    """Verify text handling: confidence rejection works; continuous noise filters reject discrete token IDs."""
    # Discrete integer token IDs (shape B, L)
    token_ids = torch.tensor([[101, 2054, 2003, 102], [101, 1045, 2066, 102]], dtype=torch.int64)

    # 1. Detection defenses with entropy and margin work on text
    det_entropy = AdversarialDetectionDefense(method="entropy")
    model = ToyTextClassifier()
    res_det = det_entropy.apply(model=model, inputs=token_ids)
    assert res_det.success is True

    # 2. Confidence rejection works on text
    conf_rej = ConfidenceRejectionDefense(threshold=0.5)
    res_rej = conf_rej.apply(model=model, inputs=token_ids)
    assert res_rej.success is True

    # 3. Continuous noise sensitivity detection rejects discrete token IDs
    det_sens = AdversarialDetectionDefense(method="sensitivity")
    with pytest.raises(HardeningConfigurationError, match="discrete integer token IDs"):
        det_sens.apply(model=model, inputs=token_ids)

    # 4. Continuous normalization rejects discrete token IDs
    norm_def = NormalizationDefense()
    with pytest.raises(HardeningConfigurationError, match="discrete integer token IDs"):
        norm_def.apply(model=model, inputs=token_ids)

    # 5. Continuous feature squeezing rejects discrete token IDs
    squeeze_def = FeatureSqueezingDefense()
    with pytest.raises(HardeningConfigurationError, match="discrete integer token IDs"):
        squeeze_def.apply(model=model, inputs=token_ids)


def test_modality_timeseries_3d():
    """Verify time-series sequence smoothing on 3D tensors (B, T, F)."""
    # 3D time-series tensor: 2 sequences, 10 timesteps, 4 features
    ts_inputs = torch.randn(2, 10, 4)

    # GaussianFilterDefense supports 1D temporal Gaussian smoothing on 3D sequences
    gf = GaussianFilterDefense(kernel_size=3, sigma=1.0)
    assert "time-series" in gf.supported_domains

    smoothed = gf.filter(ts_inputs)
    assert smoothed.shape == (2, 10, 4)
    assert not torch.allclose(ts_inputs, smoothed)


# ============================================================================
# 3. PARAMETER VALIDATION & NO-OP ELIMINATION
# ============================================================================

def test_parameter_validation_and_no_op_rejection():
    """Verify zero or invalid defense parameters raise HardeningConfigurationError."""
    # Rotation with angle <= 0
    with pytest.raises(HardeningConfigurationError, match="max_angle"):
        RotationDefense(max_angle=0.0)

    # Random crop with pad_size <= 0
    with pytest.raises(HardeningConfigurationError, match="pad_size"):
        RandomCropDefense(pad_size=0)

    # Padding with pad_size <= 0
    with pytest.raises(HardeningConfigurationError, match="pad_size"):
        PaddingDefense(pad_size=0)

    # Translation with zero shifts
    with pytest.raises(HardeningConfigurationError, match="max_dx"):
        TranslationDefense(max_dx=0.0, max_dy=0.0)

    # Random resize with min_scale >= max_scale
    with pytest.raises(HardeningConfigurationError, match="min_scale"):
        RandomResizeDefense(min_scale=1.1, max_scale=0.9)

    # Feature denoising with zero strength
    with pytest.raises(HardeningConfigurationError, match="strength"):
        FeatureDenoisingDefense(strength=0.0)


def test_no_op_defense_elimination_in_normalization():
    """Verify default NormalizationDefense actively normalizes instead of doing a silent no-op."""
    norm_def = NormalizationDefense()
    inputs = torch.tensor([[10.0, 20.0, 30.0], [100.0, 200.0, 300.0]])
    hardened = norm_def.normalize(inputs)

    # Must be actively transformed (not identical to raw inputs)
    assert not torch.allclose(inputs, hardened)
    # Check that sample means are approximately 0.0
    dims = tuple(range(1, inputs.ndim))
    means = hardened.mean(dim=dims)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)


def test_image_defense_on_incompatible_dimensions_raises():
    """Verify applying image-only spatial defenses to 2D tabular inputs raises clear configuration errors."""
    model = SimpleMLP()
    tabular_inputs = torch.randn(4, 4)

    image_defenses = [
        MedianFilterDefense(),
        ImageDenoisingDefense(),
        RotationDefense(max_angle=15.0),
        RandomCropDefense(pad_size=2),
        PaddingDefense(pad_size=2),
        TranslationDefense(max_dx=0.1, max_dy=0.1),
        RandomResizeDefense(min_scale=0.9, max_scale=1.1),
    ]

    for d in image_defenses:
        with pytest.raises(HardeningConfigurationError, match="requires 4D image"):
            d.apply(model=model, inputs=tabular_inputs)


# ============================================================================
# 4. MODEL STATE PRESERVATION & RESTORATION
# ============================================================================

def test_model_training_state_preserved_after_defense():
    """Verify model training mode is preserved and restored after defense application and evaluation."""
    model = SimpleConvNet()
    model.train()
    assert model.training is True

    inputs = torch.rand(2, 3, 16, 16)
    labels = torch.tensor([0, 1])

    conf_def = ConfidenceRejectionDefense(threshold=0.6)
    _ = conf_def.apply(model=model, inputs=inputs, labels=labels)

    # Original model's training state must be restored to True
    assert model.training is True

    # When model is in eval mode, it remains in eval mode
    model.eval()
    _ = conf_def.apply(model=model, inputs=inputs, labels=labels)
    assert model.training is False


# ============================================================================
# 5. RESOURCE AND HOOK CLEANUP
# ============================================================================

def test_hook_cleanup_in_iterative_hardening():
    """Verify intermediate forward hooks attached during candidate evaluations are cleaned up on rejection."""
    model = SimpleConvNet()
    inputs = torch.rand(2, 3, 16, 16)
    labels = torch.tensor([0, 1])

    engine = HardeningEngine()

    # Define an eval_fn where no candidate meets thresholds
    def eval_fn_failing(candidate_model):
        return {
            "clean_accuracy": 0.5,
            "asr": 0.9,
            "vulnerability_score": 85.0,
            "latency_ms": 10.0,
        }

    thresholds = RetestThresholds(min_asr_reduction=0.8)  # Strict threshold

    res = engine.harden_iterative(
        model=model,
        inputs=inputs,
        labels=labels,
        candidate_defenses=["feature_denoising", "feature_squeezing"],
        candidate_params={"feature_denoising": {"strength": 0.2}},
        eval_fn=eval_fn_failing,
        thresholds=thresholds,
        fallback_to_best=False,
    )

    assert res.selected_defense is None

    # Check that conv2 (target layer for feature denoising) has zero forward hooks on baseline model
    conv2_hooks = getattr(model.conv2, "_forward_hooks", {})
    assert len(conv2_hooks) == 0


# ============================================================================
# 6. UNSUPPORTED COMBINATION REJECTIONS
# ============================================================================

def test_adversarial_training_on_sklearn_raises():
    """Verify adversarial training fails clearly when attempted on non-PyTorch models."""
    clf = LogisticRegression()
    adv_train = AdversarialTrainingDefense()
    with pytest.raises(HardeningConfigurationError, match="differentiable PyTorch nn.Module"):
        adv_train.apply(model=clf, inputs=torch.randn(5, 4), labels=torch.tensor([0, 1, 0, 1, 0]))


def test_feature_denoising_on_sklearn_raises():
    """Verify feature denoising fails clearly when attempted on non-PyTorch models."""
    clf = RandomForestClassifier()
    feat_denoise = FeatureDenoisingDefense()
    with pytest.raises(HardeningConfigurationError, match="PyTorch nn.Module with intermediate layers"):
        feat_denoise.apply(model=clf, inputs=torch.randn(5, 4))


# ============================================================================
# 7. M7-TO-M8 HANDOFF CONTRACT
# ============================================================================

def test_m7_to_m8_handoff_contract():
    """Verify HardeningResult preserves required contract fields for M8 Re-test consumption."""
    model = SimpleConvNet()
    inputs = torch.rand(2, 3, 16, 16)
    labels = torch.tensor([0, 1])

    engine = HardeningEngine()
    res = engine.harden(
        model=model,
        defense="spatial_smoothing",
        inputs=inputs,
        labels=labels,
    )

    # Core M7 contract fields required by M8
    assert res.hardened_model is not None
    assert res.metadata is not None
    assert res.metadata.defense_name == "spatial_smoothing"
    assert res.selected_defense == "spatial_smoothing"
    assert res.thresholds is not None
    assert isinstance(res.recommendations, list)
    assert res.success is True

    # Dictionary serializability for orchestration
    res_dict = res.to_dict()
    assert "metadata" in res_dict
    assert "selected_defense" in res_dict
    assert res_dict["selected_defense"] == "spatial_smoothing"


# ============================================================================
# 8. FRONTEND DEFENSE SERVICE INTEGRATION
# ============================================================================

def test_frontend_execute_defense_with_pytorch_model():
    """Verify frontend service execute_configured_defense runs seamlessly with PyTorch model."""
    model = SimpleConvNet()
    test_inputs = torch.rand(4, 3, 16, 16)
    test_labels = torch.tensor([0, 1, 2, 0])
    adv_examples = test_inputs + 0.05 * torch.randn_like(test_inputs)

    attack_result = {
        "attack_name": "fgsm",
        "clean_accuracy": 100.0,
        "adversarial_accuracy": 25.0,
        "asr_percentage": 75.0,
        "vulnerability_score": 65.0,
        "original_predictions": [0, 1, 2, 0],
        "adversarial_predictions": [1, 2, 0, 1],
    }

    result = execute_configured_defense(
        model=model,
        domain="image",
        defense_choice="spatial_smoothing",
        test_inputs=test_inputs,
        test_labels=test_labels,
        adversarial_examples=adv_examples,
        attack_result=attack_result,
        fallback_to_best=True,
    )

    assert result["success"] is True
    assert result["defense_name"] == "spatial_smoothing"
    assert result["hardened_model"] is not None
    assert "post_clean_accuracy" in result
    assert "post_asr" in result
    assert "sample_comparisons" in result
    assert len(result["sample_comparisons"]) == 4


def test_frontend_execute_defense_with_sklearn_model():
    """Verify frontend service execute_configured_defense runs with Scikit-learn estimator."""
    X_data = np.random.randn(20, 4).astype(np.float32)
    y_data = np.random.randint(0, 2, size=20)
    clf = LogisticRegression()
    clf.fit(X_data, y_data)

    test_inputs = torch.from_numpy(X_data[:8]).float()
    test_labels = torch.from_numpy(y_data[:8]).long()
    adv_examples = test_inputs + 0.02 * torch.randn_like(test_inputs)

    attack_result = {
        "attack_name": "hopskipjump",
        "clean_accuracy": 100.0,
        "adversarial_accuracy": 25.0,
        "asr_percentage": 75.0,
        "vulnerability_score": 60.0,
    }

    result = execute_configured_defense(
        model=clf,
        domain="tabular",
        defense_choice="feature_squeezing",
        test_inputs=test_inputs,
        test_labels=test_labels,
        adversarial_examples=adv_examples,
        attack_result=attack_result,
        fallback_to_best=True,
    )

    assert result["success"] is True
    assert result["hardened_model"] is not None
    assert "post_clean_accuracy" in result
