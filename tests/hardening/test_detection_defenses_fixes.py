"""
Unit tests for bug fixes and structural enforcement in Adversarial Detection and Confidence Rejection defenses.
"""

import pytest
import torch
import torch.nn as nn
from app.hardening.defenses.detection import (
    AdversarialDetectionDefense,
    ConfidenceRejectionDefense,
)
from app.hardening.defenses.detection.adversarial_detection import AdversarialDetectorModelWrapper
from app.hardening.defenses.detection.confidence_rejection import ConfidenceRejectionModelWrapper
from app.hardening.exceptions import HardeningConfigurationError


class BinaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


class MultiClassModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 5)

    def forward(self, x):
        return self.fc(x)


# ============================================================================
# 1. STRUCTURAL PROBLEM: ENFORCED IN FORWARD() TESTS
# ============================================================================

def test_adversarial_detection_enforces_defense_in_forward():
    """Verify that hardened_model(x) actively suppresses logits for detected adversarial inputs."""
    model = MultiClassModel()
    # High noise + low threshold -> inputs will be flagged as adversarial under sensitivity
    defense = AdversarialDetectionDefense(threshold=0.01, noise_std=1.0, num_samples=5, enforce_defense=True)
    res = defense.apply(model=model)
    wrapped = res.hardened_model

    inputs = torch.randn(4, 4)
    outputs = wrapped(inputs)

    # When flagged as adversarial, logits should be suppressed to rejection_value (0.0)
    det_res = wrapped.last_detection_result
    assert det_res is not None
    is_adv = det_res["is_adversarial"]

    for i in range(4):
        if is_adv[i]:
            assert torch.all(outputs[i] == 0.0)
        else:
            assert not torch.all(outputs[i] == 0.0)


def test_confidence_rejection_enforces_defense_in_forward():
    """Verify that hardened_model(x) actively suppresses logits for low-confidence inputs."""
    model = MultiClassModel()
    # High threshold (0.99) -> random inputs will be rejected
    defense = ConfidenceRejectionDefense(threshold=0.99, enforce_defense=True, rejection_value=-999.0)
    res = defense.apply(model=model)
    wrapped = res.hardened_model

    inputs = torch.randn(4, 4)
    outputs = wrapped(inputs)

    rej_res = wrapped.last_rejection_result
    assert rej_res is not None
    is_rej = rej_res["is_rejected"]

    for i in range(4):
        if is_rej[i]:
            assert torch.all(outputs[i] == -999.0)


def test_opt_out_enforce_defense_returns_raw_logits():
    """Verify that setting enforce_defense=False allows opting out of suppression."""
    model = MultiClassModel()
    defense = AdversarialDetectionDefense(threshold=0.01, enforce_defense=False)
    res = defense.apply(model=model)
    wrapped = res.hardened_model

    inputs = torch.randn(4, 4)
    outputs = wrapped(inputs)
    raw_outputs = model(inputs)
    assert torch.allclose(outputs, raw_outputs)


# ============================================================================
# 2. NUM_SAMPLES VALIDATION BUG TEST
# ============================================================================

def test_num_samples_zero_or_negative_raises_config_error():
    """Verify num_samples < 1 raises HardeningConfigurationError instead of causing silent 0/0 NaN."""
    with pytest.raises(HardeningConfigurationError, match="num_samples must be at least 1"):
        AdversarialDetectionDefense(num_samples=0)

    with pytest.raises(HardeningConfigurationError, match="num_samples must be at least 1"):
        AdversarialDetectionDefense(num_samples=-2)


# ============================================================================
# 3. DEAD CODE IN MARGIN METHOD'S BINARY BRANCH TEST
# ============================================================================

def test_margin_method_binary_and_multiclass_branches():
    """Verify margin calculation executes cleanly for both binary and multi-class without dead code."""
    # Binary model (2 classes)
    bin_model = BinaryModel()
    bin_inputs = torch.randn(6, 4)
    bin_detector = AdversarialDetectionDefense(method="margin", threshold=0.5)
    is_adv_bin, scores_bin = bin_detector.detect(bin_model, bin_inputs)
    assert scores_bin.shape == (6,)
    assert (scores_bin >= 0.0).all() and (scores_bin <= 1.0).all()
    assert not torch.isnan(scores_bin).any()

    # Multi-class model (> 2 classes)
    mc_model = MultiClassModel()
    mc_inputs = torch.randn(6, 4)
    mc_detector = AdversarialDetectionDefense(method="margin", threshold=0.5)
    is_adv_mc, scores_mc = mc_detector.detect(mc_model, mc_inputs)
    assert scores_mc.shape == (6,)
    assert (scores_mc >= 0.0).all() and (scores_mc <= 1.0).all()
    assert not torch.isnan(scores_mc).any()


# ============================================================================
# 4. HARDENED INPUTS AND LABELS ALIGNMENT TEST
# ============================================================================

def test_adversarial_detection_hardened_inputs_and_labels_alignment():
    """Verify labels are filtered synchronously with inputs to preserve index alignment."""
    model = MultiClassModel()
    defense = AdversarialDetectionDefense(threshold=0.5, method="sensitivity", num_samples=3)

    inputs = torch.randn(10, 4)
    labels = torch.arange(10)  # [0, 1, 2, ..., 9]

    res = defense.apply(model=model, inputs=inputs, labels=labels)
    assert res.success is True
    assert res.hardened_inputs is not None
    assert res.hardened_labels is not None

    # Lengths must match exactly
    assert res.hardened_inputs.shape[0] == res.hardened_labels.shape[0]

    # Check is_accepted_mask alignment
    is_accepted_mask = res.metadata.extra_metadata["is_accepted_mask"]
    expected_labels = labels[torch.tensor(is_accepted_mask)]
    assert torch.equal(res.hardened_labels, expected_labels)


def test_confidence_rejection_hardened_inputs_and_labels_alignment():
    """Verify labels are filtered synchronously in confidence rejection."""
    model = MultiClassModel()
    defense = ConfidenceRejectionDefense(threshold=0.5)

    inputs = torch.randn(8, 4)
    labels = torch.arange(8)

    res = defense.apply(model=model, inputs=inputs, labels=labels)
    assert res.success is True
    assert res.hardened_inputs is not None
    assert res.hardened_labels is not None
    assert res.hardened_inputs.shape[0] == res.hardened_labels.shape[0]

    is_accepted_mask = res.metadata.extra_metadata["is_accepted_mask"]
    expected_labels = labels[torch.tensor(is_accepted_mask)]
    assert torch.equal(res.hardened_labels, expected_labels)


class MockTransformerTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 2)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(f"embedding(): argument 'indices' (position 2) must be Tensor, not {type(input_ids).__name__}")
        x = input_ids.float()
        return self.fc(x)


def test_confidence_rejection_dictionary_inputs_text_model():
    """Verify ConfidenceRejectionDefense works with transformer dictionary inputs without TypeError."""
    model = MockTransformerTextModel()
    defense = ConfidenceRejectionDefense(threshold=0.6)

    inputs = {
        "input_ids": torch.randint(0, 100, (6, 8)),
        "attention_mask": torch.ones((6, 8)),
    }
    labels = torch.tensor([0, 1, 0, 1, 0, 1])

    res = defense.apply(model=model, inputs=inputs, labels=labels)
    assert res.success is True
    assert isinstance(res.hardened_inputs, dict)
    assert "input_ids" in res.hardened_inputs
    assert res.hardened_labels is not None
    assert len(res.hardened_inputs["input_ids"]) == len(res.hardened_labels)

    # Calling wrapped model with kwargs and positional dict
    out_kwargs = res.hardened_model(**inputs)
    assert isinstance(out_kwargs, torch.Tensor)
    assert out_kwargs.shape == (6, 2)

    out_pos = res.hardened_model(inputs)
    assert isinstance(out_pos, torch.Tensor)
    assert out_pos.shape == (6, 2)


def test_adversarial_detection_dictionary_inputs_text_model():
    """Verify AdversarialDetectionDefense works with transformer dictionary inputs without TypeError."""
    model = MockTransformerTextModel()
    defense = AdversarialDetectionDefense(method="margin", threshold=0.5)

    inputs = {
        "input_ids": torch.randint(0, 100, (6, 8)),
        "attention_mask": torch.ones((6, 8)),
    }
    labels = torch.tensor([0, 1, 0, 1, 0, 1])

    res = defense.apply(model=model, inputs=inputs, labels=labels)
    assert res.success is True
    assert isinstance(res.hardened_inputs, dict)
    assert "input_ids" in res.hardened_inputs
    assert res.hardened_labels is not None
    assert len(res.hardened_inputs["input_ids"]) == len(res.hardened_labels)

    out_kwargs = res.hardened_model(**inputs)
    assert isinstance(out_kwargs, torch.Tensor)
    assert out_kwargs.shape == (6, 2)

    out_pos = res.hardened_model(inputs)
    assert isinstance(out_pos, torch.Tensor)
    assert out_pos.shape == (6, 2)
