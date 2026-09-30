"""
Integration tests for Module 1 end-to-end ingestion pipeline and alias imports.
"""

import tempfile
from pathlib import Path
import torch
import torch.nn as nn

from app.ingestion import ingest_model
from app.ingestion.adapters.pytorch_adapter import PyTorchAdapter
from app.ingestion.metadata.model_metadata import ModelMetadata


class EndToEndModel(nn.Module):
    """Network for end-to-end integration testing."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )

    def forward(self, x):
        return self.net(x)


def test_ingest_model_from_instance_integration():
    """Test full ingest_model pipeline from an in-memory nn.Module instance."""
    model = EndToEndModel()
    sample_input = torch.randn(2, 16)

    adapter, metadata = ingest_model(
        model_path=model,
        sample_input=sample_input,
        model_name="EndToEndModelTest",
        device="cpu"
    )

    # Verify adapter state
    assert isinstance(adapter, PyTorchAdapter)
    assert adapter.get_model() is model
    assert not adapter.get_model().training

    # Verify metadata fields populated from inference
    assert isinstance(metadata, ModelMetadata)
    assert metadata.framework == "pytorch"
    assert metadata.model_name == "EndToEndModelTest"
    assert metadata.input_shape == (2, 16)
    assert metadata.output_shape == (2, 4)
    assert metadata.num_classes == 4
    assert metadata.device == "cpu"

    # Verify end-to-end prediction execution
    preds = adapter.predict(sample_input)
    assert preds.shape == (2, 4)


def test_ingest_model_from_file_integration():
    """Test full ingest_model pipeline from a saved checkpoint file on disk."""
    model = EndToEndModel()
    sample_input = torch.randn(1, 16)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "e2e_checkpoint.pt"
        torch.save(model, checkpoint_path)

        adapter, metadata = ingest_model(
            model_path=checkpoint_path,
            sample_input=sample_input,
            device="cpu"
        )

        assert isinstance(adapter, PyTorchAdapter)
        assert metadata.input_shape == (1, 16)
        assert metadata.output_shape == (1, 4)
        assert metadata.num_classes == 4


def test_app_model_ingestion_alias_integration():
    """Test importing all public module interfaces via app.model_ingestion alias."""
    from app.ingestion import (
        BaseModelAdapter,
        BaseModelLoader,
        DeviceManager,
        ModelIngestionError,
        ModelLoadError,
        ModelMetadata,
        ModelValidationError,
        ModelValidator,
        PyTorchAdapter,
        PyTorchLoader,
        UnsupportedModelError,
        ingest_model as alias_ingest_model,
    )

    model = EndToEndModel()
    sample_input = torch.randn(1, 16)

    adapter, metadata = alias_ingest_model(
        model_path=model,
        sample_input=sample_input,
        device="cpu"
    )

    assert isinstance(adapter, PyTorchAdapter)
    assert isinstance(metadata, ModelMetadata)


def test_direct_pipeline_import_integration():
    """Test importing ingest_model directly from app.ingestion.pipeline."""
    from app.ingestion.pipeline import ingest_model as direct_ingest_model

    model = EndToEndModel()
    sample_input = torch.randn(1, 16)

    adapter, metadata = direct_ingest_model(
        model_path=model,
        sample_input=sample_input,
        device="cpu"
    )

    assert isinstance(adapter, PyTorchAdapter)
    assert isinstance(metadata, ModelMetadata)
