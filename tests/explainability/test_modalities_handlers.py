"""
Unit tests for domain modality handlers (image, text, tabular, timeseries).
"""

import numpy as np
import pandas as pd
import pytest
import torch

from app.explainability.modalities import (
    ImageModalityHandler,
    TabularModalityHandler,
    TextModalityHandler,
    TimeSeriesModalityHandler,
    get_modality_handler,
)
from app.explainability.modality import Modality


def test_modality_handler_factory():
    assert isinstance(get_modality_handler("image"), ImageModalityHandler)
    assert isinstance(get_modality_handler("text"), TextModalityHandler)
    assert isinstance(get_modality_handler("tabular"), TabularModalityHandler)
    assert isinstance(get_modality_handler("timeseries"), TimeSeriesModalityHandler)


def test_image_modality_handler():
    handler = ImageModalityHandler()
    assert handler.modality == Modality.IMAGE

    # Test 3D (C, H, W) -> 4D (1, C, H, W)
    img_3d = torch.randn(3, 16, 16)
    prep = handler.prepare_input(img_3d)
    assert prep.ndim == 4
    assert prep.shape == (1, 3, 16, 16)

    # Test prepare_batch
    adv_3d = img_3d + 0.05
    c_prep, a_prep = handler.prepare_batch(img_3d, adv_3d)
    assert c_prep.shape == a_prep.shape

    # Test postprocessing attribution
    dummy_attr = np.random.randn(1, 3, 16, 16)
    res = handler.postprocess_attribution(dummy_attr, img_3d, top_k=3)
    assert "heatmap" in res
    assert "top_features" in res
    assert len(res["top_features"]) == 3
    assert "summary" in res
    assert "coordinates" in res["top_features"][0]


def test_image_modality_shape_mismatch():
    handler = ImageModalityHandler()
    img_a = torch.randn(3, 16, 16)
    img_b = torch.randn(3, 32, 32)
    with pytest.raises(ValueError, match="Image shape mismatch"):
        handler.prepare_batch(img_a, img_b)


def test_text_modality_handler():
    handler = TextModalityHandler()
    assert handler.modality == Modality.TEXT

    text = "The quick brown fox jumps over the lazy dog"
    prep = handler.prepare_input(text)
    assert prep == text

    adv_text = "The quick brown fox jumps over a lazy dog"
    c_p, a_p = handler.prepare_batch(text, adv_text)
    assert c_p == text
    assert a_p == adv_text

    dummy_attr = [0.1, 0.5, 0.2, 0.8, -0.3, 0.0, 0.1, -0.9, 0.4]
    res = handler.postprocess_attribution(dummy_attr, text, top_k=3)
    assert "tokens" in res
    assert "token_importance" in res
    assert len(res["top_features"]) == 3
    assert res["top_features"][0]["feature"] in text.split()


def test_tabular_modality_handler():
    handler = TabularModalityHandler()
    assert handler.modality == Modality.TABULAR

    df = pd.DataFrame({
        "age": [25.0, 30.0],
        "income": [50000.0, 60000.0],
        "score": [700.0, 750.0],
    })
    prep = handler.prepare_input(df)
    assert prep.ndim == 2
    assert prep.shape == (2, 3)
    assert handler.feature_names == ["age", "income", "score"]

    dummy_attr = [0.2, -0.8, 0.5]
    res = handler.postprocess_attribution(dummy_attr, prep, top_k=2)
    assert "feature_importance_map" in res
    assert abs(res["feature_importance_map"]["income"] - (-0.8)) < 1e-5
    assert res["top_features"][0]["feature"] == "income"


def test_timeseries_modality_handler():
    handler = TimeSeriesModalityHandler(variable_names=["sensor_1", "sensor_2"])
    assert handler.modality == Modality.TIMESERIES

    # 2D (T, F) -> 3D (1, T, F)
    ts_2d = torch.randn(20, 2)
    prep = handler.prepare_input(ts_2d)
    assert prep.shape == (1, 20, 2)

    dummy_attr = np.random.randn(20, 2)
    res = handler.postprocess_attribution(dummy_attr, ts_2d, top_k=4)
    assert "step_importance" in res
    assert "variable_importance" in res
    assert "peak_window" in res
    assert len(res["top_features"]) == 4
    assert "sensor_1" in res["variable_importance"]
