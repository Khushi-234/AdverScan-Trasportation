"""
Unit and integration tests for time-series model and dataset ingestion in AdverScan.
Verifies time-series classification model ingestion, TimeSeriesProcessor functionality,
forecasting model rejection with UnsupportedTaskError, and local PyTorch classifiers
(1D CNN, LSTM, GRU, Transformer).
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from app.ingestion import (
    HuggingFaceAdapter,
    HuggingFaceLoader,
    IngestionResult,
    ModelMetadata,
    ModelPresetRegistry,
    PyTorchAdapter,
    PyTorchLoader,
    TimeSeriesProcessor,
    UnsupportedTaskError,
    ingest_model,
)
from app.ingestion.model_registry import detect_model_domain
from app.ingestion.validation import ModelValidator


# ---------------------------------------------------------------------------
# Test Architectures: 1D CNN, LSTM, GRU, Transformer
# ---------------------------------------------------------------------------

class Dummy1DCNNClassifier(nn.Module):
    """1D CNN classifier for time-series data [batch, channels, sequence_length]."""
    def __init__(self, in_channels: int = 1, num_classes: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: [batch, channels, sequence_length]
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class DummyLSTMClassifier(nn.Module):
    """LSTM classifier for time-series data."""
    def __init__(self, in_channels: int = 1, hidden_size: int = 32, num_classes: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(in_channels, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accepts [batch, channels, seq_len] or [batch, seq_len, channels]
        if x.ndim == 3 and x.shape[1] == 1 and x.shape[2] > 1:
            x = x.transpose(1, 2)
        out, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


class DummyGRUClassifier(nn.Module):
    """GRU classifier for time-series data."""
    def __init__(self, in_channels: int = 1, hidden_size: int = 32, num_classes: int = 2):
        super().__init__()
        self.gru = nn.GRU(in_channels, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3 and x.shape[1] == 1 and x.shape[2] > 1:
            x = x.transpose(1, 2)
        out, hn = self.gru(x)
        return self.fc(hn[-1])


class DummyTransformerTSClassifier(nn.Module):
    """Transformer-based classifier for time-series data."""
    def __init__(self, in_channels: int = 1, d_model: int = 16, num_classes: int = 5, nhead: int = 2):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, d_model, kernel_size=1)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).transpose(1, 2)
        x = self.encoder(x)
        return self.fc(x.mean(dim=1))


class DummyTSForecaster(nn.Module):
    """Mock time-series forecasting model (not supported for classification)."""
    def __init__(self, in_channels: int = 1, forecast_horizon: int = 24):
        super().__init__()
        self.task_type = "forecasting"
        self.linear = nn.Linear(50, forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------------------------------------------------------------------------
# TimeSeriesProcessor Tests
# ---------------------------------------------------------------------------

def test_time_series_processor_tensor_shapes():
    """Verify processor converts 1D, 2D, and 3D inputs to [batch, channels, sequence_length]."""
    proc = TimeSeriesProcessor(num_channels=1, target_format="batch_channels_length")

    # 1D single sequence [100] -> [1, 1, 100]
    x_1d = torch.randn(100)
    out_1d = proc.process(x_1d)
    assert out_1d.shape == (1, 1, 100)
    assert out_1d.dtype == torch.float32

    # 2D batch of univariate sequences [8, 100] -> [8, 1, 100]
    x_2d = torch.randn(8, 100)
    out_2d = proc.process(x_2d)
    assert out_2d.shape == (8, 1, 100)

    # 3D already formatted [4, 1, 100] -> [4, 1, 100]
    x_3d = torch.randn(4, 1, 100)
    out_3d = proc.process(x_3d)
    assert out_3d.shape == (4, 1, 100)


def test_time_series_processor_dataframe_and_series():
    """Verify processor handles pandas DataFrame and Series inputs."""
    proc = TimeSeriesProcessor(num_channels=1)

    df = pd.DataFrame({"sensor_1": np.linspace(0, 10, 50, dtype=np.float32)})
    out = proc.process(df)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (50, 1, 1) or out.ndim == 3

    series = pd.Series(np.arange(30, dtype=np.float32))
    out_s = proc.process(series)
    assert isinstance(out_s, torch.Tensor)
    assert out_s.shape == (1, 1, 30)


def test_time_series_processor_normalization_and_standardization():
    """Verify z-score standardization and min-max normalization."""
    proc_std = TimeSeriesProcessor(standardize=True)
    x = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0]])
    out_std = proc_std.process(x)
    assert pytest.approx(out_std.mean().item(), abs=1e-5) == 0.0
    assert pytest.approx(out_std.std().item(), abs=1e-5) == 1.0

    proc_norm = TimeSeriesProcessor(normalize=True)
    out_norm = proc_norm.process(x)
    assert pytest.approx(out_norm.min().item(), abs=1e-5) == 0.0
    assert pytest.approx(out_norm.max().item(), abs=1e-5) == 1.0


def test_time_series_processor_padding_and_truncation():
    """Verify sequence_length padding and truncation."""
    proc = TimeSeriesProcessor(sequence_length=40, pad_value=0.0)

    # Truncate 60 -> 40
    x_long = torch.randn(1, 1, 60)
    assert proc.process(x_long).shape == (1, 1, 40)

    # Pad 25 -> 40
    x_short = torch.ones(1, 1, 25)
    out_padded = proc.process(x_short)
    assert out_padded.shape == (1, 1, 40)
    assert (out_padded[0, 0, 25:] == 0.0).all()


def test_time_series_processor_load_file():
    """Verify loading from CSV and NPY files on disk."""
    proc = TimeSeriesProcessor()
    with tempfile.TemporaryDirectory() as tmpdir:
        # CSV test
        csv_path = Path(tmpdir) / "sample.csv"
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        df.to_csv(csv_path, index=False)
        out_csv = proc.process(csv_path)
        assert isinstance(out_csv, torch.Tensor)
        assert out_csv.shape == (5, 1, 1) or out_csv.ndim == 3

        # NPY test
        npy_path = Path(tmpdir) / "sample.npy"
        np.save(npy_path, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
        out_npy = proc.process(npy_path)
        assert isinstance(out_npy, torch.Tensor)
        assert out_npy.shape == (2, 1, 3)


# ---------------------------------------------------------------------------
# Domain and Task Detection Tests
# ---------------------------------------------------------------------------

def test_detect_domain_time_series():
    """Verify automatic detection of data_domain='time-series'."""
    assert detect_model_domain(Dummy1DCNNClassifier()) == "time-series"
    assert detect_model_domain(DummyLSTMClassifier()) == "time-series"
    assert detect_model_domain(DummyGRUClassifier()) == "time-series"
    assert detect_model_domain(DummyTransformerTSClassifier()) == "time-series"

    # From processor name
    assert detect_model_domain("custom/model", processor_name="timeseries_processor") == "time-series"

    # From DataFrame sample input
    df_sample = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    assert detect_model_domain("custom/model", sample_input=df_sample) == "time-series"


# ---------------------------------------------------------------------------
# Forecasting Model Rejection Tests (Requirement 7 & 8)
# ---------------------------------------------------------------------------

def test_forecasting_model_raises_unsupported_task_error():
    """Verify forecasting models raise UnsupportedTaskError with explicit message."""
    # 1. Via task_type='forecasting'
    with pytest.raises(UnsupportedTaskError) as exc_info:
        ingest_model(
            model_path=Dummy1DCNNClassifier(),
            task_type="forecasting",
            data_domain="time-series",
        )
    assert "AdverScan currently supports time-series classification models only; forecasting models are not supported." in str(exc_info.value)

    # 2. Via task_type='regression'
    with pytest.raises(UnsupportedTaskError) as exc_info2:
        ingest_model(
            model_path=Dummy1DCNNClassifier(),
            task_type="regression",
            data_domain="time-series",
        )
    assert "AdverScan currently supports time-series classification models only; forecasting models are not supported." in str(exc_info2.value)


def test_forecasting_model_detected_from_architecture_and_pipeline_tag():
    """Verify forecasting models detected from HF config architecture or pipeline_tag raise error."""
    # Mock HF loader with a forecasting architecture
    mock_config = MagicMock()
    mock_config.architectures = ["PatchTSTForPrediction"]
    mock_config.pipeline_tag = "time-series-forecasting"

    loader = HuggingFaceLoader()
    with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config), \
         patch("app.ingestion.loader.huggingface_loader.hf_hub_download", side_effect=Exception("offline")):
        with pytest.raises(UnsupportedTaskError) as exc_info:
            loader.load("mock/forecasting-model", domain="time-series")
        assert "AdverScan currently supports time-series classification models only; forecasting models are not supported." in str(exc_info.value)


# ---------------------------------------------------------------------------
# Local PyTorch Time-Series Classifiers Ingestion Tests (Requirement 4 & 11)
# ---------------------------------------------------------------------------

def test_ingest_local_pytorch_1dcnn():
    """Verify successful ingestion of local 1D CNN classifier."""
    model = Dummy1DCNNClassifier(in_channels=1, num_classes=4)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "1dcnn_model.pt"
        torch.save(model, model_path)

        sample = torch.randn(2, 1, 50)
        result = ingest_model(
            model_path=model_path,
            sample_input=sample,
            data_domain="time-series",
            device="cpu",
        )

        assert isinstance(result, IngestionResult)
        adapter, meta = result
        assert isinstance(adapter, PyTorchAdapter)
        assert meta.data_domain == "time-series"
        assert meta.num_classes == 4
        assert meta.framework == "pytorch_local"
        assert meta.input_shape == (2, 1, 50)
        assert isinstance(result.processor, TimeSeriesProcessor)

        preds = adapter.predict(sample)
        assert preds.shape == (2, 4)


def test_ingest_local_pytorch_lstm():
    """Verify successful ingestion of local LSTM classifier."""
    model = DummyLSTMClassifier(in_channels=1, hidden_size=16, num_classes=3)
    sample = torch.randn(2, 1, 40)
    result = ingest_model(
        model_path=model,
        sample_input=sample,
        device="cpu",
    )

    adapter, meta = result
    assert isinstance(adapter, PyTorchAdapter)
    assert meta.data_domain == "time-series"
    assert meta.num_classes == 3
    assert isinstance(result.processor, TimeSeriesProcessor)

    preds = adapter.predict(sample)
    assert preds.shape == (2, 3)


def test_ingest_local_pytorch_gru():
    """Verify successful ingestion of local GRU classifier."""
    model = DummyGRUClassifier(in_channels=1, hidden_size=16, num_classes=2)
    sample = torch.randn(4, 1, 30)
    result = ingest_model(
        model_path=model,
        sample_input=sample,
        device="cpu",
    )

    adapter, meta = result
    assert meta.data_domain == "time-series"
    assert meta.num_classes == 2
    preds = adapter.predict(sample)
    assert preds.shape == (4, 2)


def test_ingest_local_pytorch_transformer():
    """Verify successful ingestion of local Transformer time-series classifier."""
    model = DummyTransformerTSClassifier(in_channels=1, d_model=16, num_classes=5)
    sample = torch.randn(2, 1, 32)
    result = ingest_model(
        model_path=model,
        sample_input=sample,
        device="cpu",
    )

    adapter, meta = result
    assert meta.data_domain == "time-series"
    assert meta.num_classes == 5
    preds = adapter.predict(sample)
    assert preds.shape == (2, 5)


# ---------------------------------------------------------------------------
# Hugging Face Time-Series Classification Model Tests (Requirement 1 & 11)
# ---------------------------------------------------------------------------

def test_ingest_huggingface_time_series_classifier():
    """Verify successful ingestion of Hugging Face time-series classification model."""
    class MockPatchTSTOutput:
        def __init__(self, logits):
            self.prediction_logits = logits
            self.logits = logits

    class MockPatchTSTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(14, 2)
            self.config = MagicMock()
            self.config.context_length = 14
            self.config.num_input_channels = 1
            self.config.num_classes = 2
            self.config.architectures = ["PatchTSTForClassification"]

        def forward(self, x=None, past_values=None, **kwargs):
            inp = past_values if past_values is not None else x
            if inp.ndim == 3 and inp.shape[-1] == 1:
                inp = inp.squeeze(-1)
            return MockPatchTSTOutput(self.linear(inp))

    mock_model = MockPatchTSTModel()
    with patch.object(HuggingFaceLoader, "load", return_value=mock_model):
        sample = torch.randn(2, 14, 1)
        result = ingest_model(
            model_path="mock/patchtst-classifier",
            sample_input=sample,
            data_domain="time-series",
            framework="huggingface",
            device="cpu",
        )

        adapter, meta = result
        assert isinstance(adapter, HuggingFaceAdapter)
        assert meta.data_domain == "time-series"
        assert meta.num_classes == 2
        assert meta.framework == "huggingface"
        assert isinstance(result.processor, TimeSeriesProcessor)

        preds = adapter.predict(sample)
        assert preds.shape == (2, 2)


def test_time_series_no_image_text_defaults():
    """Verify that image/text defaults (e.g. input_resolution) are not applied to time-series."""
    cfg = ModelPresetRegistry.build_runtime_config(
        model_id="custom-ts-model",
        data_domain="time-series",
    )
    assert cfg["data_domain"] == "time-series"
    assert cfg.get("input_resolution") is None
    assert "max_length" not in cfg


def test_time_series_preset_lookup():
    """Verify that the time-series model preset is present and resolvable in ModelPresetRegistry."""
    preset_id = "hf-tiny-v2/tiny-random-PatchTSTForClassification"
    assert ModelPresetRegistry.has_preset(preset_id)
    preset = ModelPresetRegistry.get_preset(preset_id)
    assert preset is not None
    assert preset["data_domain"] == "time-series"
    assert preset["task_type"] == "classification"
    assert preset["num_classes"] == 2
    assert preset["framework"] == "huggingface"
    assert preset["sequence_length"] == 14
    assert preset["num_channels"] == 1
