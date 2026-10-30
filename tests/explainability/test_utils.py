"""
Unit tests for the modular explainability utils package:
- app.explainability.utils.conversions
- app.explainability.utils.validation
- app.explainability.utils.prediction
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

# Direct modular imports
from app.explainability.utils.conversions import to_numpy, to_tensor, unwrap_model
from app.explainability.utils.validation import (
    ensure_batch,
    get_shape,
    validate_matching_shapes,
    validate_single_sample,
)
from app.explainability.utils.prediction import (
    configure_cpu_threads,
    extract_prediction_and_confidence,
    predict_batched,
    predict_proba_np,
    safe_predict,
)

# Package-level imports
import app.explainability.utils as xai_utils


class SimpleDummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc(x)


class DummyAdapter:
    def __init__(self, inner):
        self._model = inner

    def get_model(self):
        return self._model


# -------------------------------------------------------------------------
# Conversions Tests
# -------------------------------------------------------------------------

def test_unwrap_model_nested():
    raw = SimpleDummyModel()
    adapter1 = DummyAdapter(raw)
    adapter2 = DummyAdapter(adapter1)
    assert unwrap_model(adapter2) is raw
    assert unwrap_model(raw) is raw
    assert unwrap_model(None) is None


def test_to_numpy_conversions():
    # Tensor
    t = torch.tensor([1.0, 2.0, 3.0])
    arr = to_numpy(t)
    assert isinstance(arr, np.ndarray)
    assert np.allclose(arr, [1.0, 2.0, 3.0])

    # List & dtype
    arr_int = to_numpy([1.2, 3.7], dtype=np.int32)
    assert arr_int.dtype == np.int32
    assert list(arr_int) == [1, 3]

    # Scalar
    arr_scalar = to_numpy(42)
    assert arr_scalar.shape == (1,)
    assert arr_scalar[0] == 42


def test_to_tensor_conversions():
    arr = np.array([1.5, 2.5], dtype=np.float32)
    t = to_tensor(arr)
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.float32
    assert torch.allclose(t, torch.tensor([1.5, 2.5]))

    # Scalar
    t_scalar = to_tensor(5.0)
    assert t_scalar.shape == (1,)
    assert t_scalar[0] == 5.0


# -------------------------------------------------------------------------
# Validation Tests
# -------------------------------------------------------------------------

def test_get_shape_and_validation():
    t = torch.zeros((2, 3, 4))
    assert get_shape(t) == (2, 3, 4)
    assert get_shape([1, 2, 3]) == (3,)
    assert get_shape(99) == ()

    # ensure_batch
    tab_1d = np.array([1.0, 2.0])
    batched_tab, was_single = ensure_batch(tab_1d, modality="tabular")
    assert was_single is True
    assert batched_tab.shape == (1, 2)

    img_3d = torch.zeros((3, 32, 32))
    batched_img, was_single_img = ensure_batch(img_3d, modality="image")
    assert was_single_img is True
    assert batched_img.shape == (1, 3, 32, 32)


def test_validate_single_sample():
    # Valid single sample
    validate_single_sample(torch.zeros(1, 4), modality="tabular")
    validate_single_sample("single text", modality="text")
    validate_single_sample(["single text in list"], modality="text")

    # Invalid batch N > 1
    with pytest.raises(ValueError, match="explain\\(\\) accepts exactly one sample"):
        validate_single_sample(torch.zeros(3, 4), modality="tabular")

    with pytest.raises(ValueError, match="explain\\(\\) accepts exactly one sample"):
        validate_single_sample(["sample 1", "sample 2"], modality="text")


def test_validate_matching_shapes():
    clean = torch.zeros(1, 4)
    adv = torch.zeros(1, 4)
    mismatch = torch.zeros(1, 5)

    assert validate_matching_shapes(clean, adv) is True
    assert validate_matching_shapes(clean, mismatch) is False
    assert validate_matching_shapes(["text A"], ["text B"]) is True
    assert validate_matching_shapes(["text A"], ["text B", "text C"]) is False


# -------------------------------------------------------------------------
# Prediction Tests
# -------------------------------------------------------------------------

def test_prediction_utilities():
    model = SimpleDummyModel()
    x = torch.randn(10, 3)

    # safe_predict
    res_tensor = safe_predict(model, x, output_mode="classification")
    assert isinstance(res_tensor, torch.Tensor)
    assert res_tensor.shape == (10, 2)

    # extract_prediction_and_confidence
    probs = np.array([[0.1, 0.9]])
    pred, conf = extract_prediction_and_confidence(probs, output_mode="classification")
    assert pred == 1
    assert pytest.approx(conf, 0.01) == 0.9

    # regression
    reg_val, reg_conf = extract_prediction_and_confidence(np.array([[3.14]]), output_mode="regression")
    assert pytest.approx(reg_val, 0.01) == 3.14
    assert reg_conf == 1.0


# -------------------------------------------------------------------------
# Package and Facade Equivalence Tests
# -------------------------------------------------------------------------

def test_utils_package_exports():
    expected_exports = [
        "unwrap_model",
        "to_numpy",
        "to_tensor",
        "ensure_batch",
        "get_shape",
        "validate_single_sample",
        "validate_matching_shapes",
        "configure_cpu_threads",
        "predict_batched",
        "predict_proba_np",
        "safe_predict",
        "extract_prediction_and_confidence",
    ]
    for name in expected_exports:
        assert hasattr(xai_utils, name), f"app.explainability.utils missing {name}"
        assert callable(getattr(xai_utils, name))
