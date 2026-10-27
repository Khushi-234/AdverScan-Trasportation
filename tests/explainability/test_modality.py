"""
Unit tests for modality definition and auto-detection in AdverScan XAI.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from app.explainability.modality import (
    Modality,
    detect_modality,
    is_valid_modality,
)


def test_modality_enum_and_validation():
    assert Modality.from_str("image") == Modality.IMAGE
    assert Modality.from_str("TEXT") == Modality.TEXT
    assert Modality.from_str("tabular") == Modality.TABULAR
    assert Modality.from_str("time_series") == Modality.TIMESERIES

    assert is_valid_modality("image") is True
    assert is_valid_modality("timeseries") is True
    assert is_valid_modality("unknown_domain") is False

    with pytest.raises(ValueError, match="Unsupported modality"):
        Modality.from_str("quantum")


def test_detect_modality_with_hints():
    assert detect_modality([1, 2, 3], hint="image") == Modality.IMAGE
    assert detect_modality([1, 2, 3], hint="text") == Modality.TEXT
    assert detect_modality([1, 2, 3], hint="timeseries") == Modality.TIMESERIES
    assert detect_modality([1, 2, 3], hint="tabular") == Modality.TABULAR


def test_detect_modality_text():
    assert detect_modality("This is a clean sample sentence.") == Modality.TEXT
    assert detect_modality(["first sentence", "second sentence"]) == Modality.TEXT
    assert detect_modality({"text": "Adversarial text input"}) == Modality.TEXT
    assert detect_modality({"tokens": ["apple", "orange"]}) == Modality.TEXT


def test_detect_modality_tabular():
    df = pd.DataFrame({"col_a": [1.0, 2.0], "col_b": [3.0, 4.0]})
    assert detect_modality(df) == Modality.TABULAR

    arr_2d = np.random.randn(10, 5)
    assert detect_modality(arr_2d) == Modality.TABULAR

    tensor_2d = torch.randn(8, 4)
    assert detect_modality(tensor_2d) == Modality.TABULAR

    tensor_1d = torch.randn(6)
    assert detect_modality(tensor_1d) == Modality.TABULAR


def test_detect_modality_image():
    # 4D Image batch (B, C, H, W)
    t_4d = torch.randn(4, 3, 32, 32)
    assert detect_modality(t_4d) == Modality.IMAGE

    # 3D Single Image (C, H, W) where C in (1, 3, 4) and spatial >= 8
    t_3d = torch.randn(3, 28, 28)
    assert detect_modality(t_3d) == Modality.IMAGE

    # Dict with image
    assert detect_modality({"pixel_values": t_4d}) == Modality.IMAGE


def test_detect_modality_timeseries():
    # 3D sequential data (B, T, F) where spatial dims don't match typical image
    t_seq = torch.randn(16, 50, 6)
    assert detect_modality(t_seq) == Modality.TIMESERIES

    assert detect_modality({"sequence": t_seq}) == Modality.TIMESERIES


def test_detect_modality_invalid():
    class CustomObject:
        pass

    with pytest.raises(ValueError, match="Unable to auto-detect modality"):
        detect_modality(CustomObject())
