"""
Unit tests for domain detection and preventing image fallback.
Verifies that models automatically detect 'image' or 'text', and fall back to 'unknown',
never defaulting blindly to 'image'.
"""

from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.nn as nn

from app.ingestion.adapters.huggingface_adapter import HuggingFaceAdapter
from app.ingestion.exceptions import ModelLoadError, UnsupportedDomainError
from app.ingestion.loader.huggingface_loader import HuggingFaceLoader
from app.ingestion.metadata.model_metadata import ModelMetadata
from app.ingestion.model_registry import (
    ModelPresetRegistry,
    detect_model_domain,
    setup_model_environment,
)
from app.ingestion.pipeline import ingest_model


class SimpleConvNet(nn.Module):
    """Network with 2D convolution layers."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        return self.fc(self.pool(self.conv(x)).flatten(1))


class SimpleTextNet(nn.Module):
    """Network with embedding layers for text processing."""
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(100, 16)
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        return self.fc(self.emb(x).mean(dim=1))


class UnknownLinearNet(nn.Module):
    """Generic neural network with only linear layers and no domain cues."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


def test_metadata_defaults_to_none():
    """Verify ModelMetadata does not default domain or data_domain to 'image'."""
    meta = ModelMetadata()
    assert meta.data_domain is None
    assert meta.domain is None


def test_detect_domain_from_presets():
    """Verify domain detection for known preset models."""
    assert detect_model_domain("bazyl/gtsrb-model") == "image"
    assert detect_model_domain("ashutoshsharma58/scenery3214154") == "image"


def test_detect_domain_from_model_name():
    """Verify domain detection from vision vs text model name keywords."""
    assert detect_model_domain("google/vit-base-patch16-224") == "image"
    assert detect_model_domain("microsoft/resnet-50") == "image"
    assert detect_model_domain("bert-base-uncased") == "text"
    assert detect_model_domain("roberta-large") == "text"
    assert detect_model_domain("t5-small") == "text"


def test_detect_domain_from_sample_input():
    """Verify domain detection from sample inputs."""
    # Text: string or list of strings
    assert detect_model_domain("unknown-id", sample_input="Sample text prompt") == "text"
    assert detect_model_domain("unknown-id", sample_input=["Review 1", "Review 2"]) == "text"

    # Text: token IDs tensor (integer dtype)
    assert detect_model_domain("unknown-id", sample_input=torch.tensor([[101, 2023, 102]], dtype=torch.long)) == "text"

    # Text: dictionary containing input_ids
    assert detect_model_domain("unknown-id", sample_input={"input_ids": torch.tensor([[101, 102]])}) == "text"

    # Image: 4D tensor (B, C, H, W)
    assert detect_model_domain("unknown-id", sample_input=torch.randn(1, 3, 224, 224)) == "image"

    # Image: dictionary containing pixel_values
    assert detect_model_domain("unknown-id", sample_input={"pixel_values": torch.randn(1, 3, 224, 224)}) == "image"


def test_detect_domain_from_in_memory_architecture():
    """Verify domain detection from in-memory module layer types."""
    conv_net = SimpleConvNet()
    assert detect_model_domain(conv_net) == "image"

    text_net = SimpleTextNet()
    assert detect_model_domain(text_net) == "text"

    # Net with no domain cues should return 'unknown', NEVER 'image'
    unknown_net = UnknownLinearNet()
    assert detect_model_domain(unknown_net) == "unknown"


def test_build_runtime_config_unknown_fallback():
    """Verify that build_runtime_config sets unknown and does NOT set input_resolution to (3, 224, 224)."""
    cfg = ModelPresetRegistry.build_runtime_config(model_id="arbitrary-org/custom-model")
    assert cfg["data_domain"] == "unknown"
    assert "Unknown" in cfg["domain"]
    assert cfg["input_resolution"] is None


def test_resolve_model_info_unknown_fallback():
    """Verify that resolve_model_info sets unknown domain for ambiguous models."""
    info = ModelPresetRegistry.resolve_model_info(model_id_or_path="arbitrary-unknown-model")
    assert info["data_domain"] == "unknown"
    assert "Unknown" in info["domain"]
    assert info["input_resolution"] is None


def test_setup_model_environment_unknown_domain():
    """Verify setup_model_environment returns 'unknown' domain for unregistered ambiguous models."""
    env = setup_model_environment("unregistered-generic-model", log_log_event=False)
    assert env["data_domain"] == "unknown"
    assert env["expected_input_resolution"] is None


def test_huggingface_loader_rejects_unknown_domain():
    """Verify HuggingFaceLoader does not assume image and raises error if domain is unknown."""
    loader = HuggingFaceLoader()
    with pytest.raises(ModelLoadError, match="Could not automatically determine model domain"):
        loader.load("arbitrary-org/unknown-model", domain="unknown")


def test_huggingface_adapter_infers_domain_from_processor():
    """Verify HuggingFaceAdapter infers domain from processor when not explicitly provided."""
    dummy_model = UnknownLinearNet()
    mock_processor = MagicMock()
    mock_processor.data_domain = "text"

    adapter = HuggingFaceAdapter(
        model=dummy_model,
        processor=mock_processor,
        data_domain=None,
    )
    assert adapter.data_domain == "text"


def test_ingest_model_unknown_domain_does_not_create_image_processor():
    """Verify ingest_model with unknown domain creates no processor and sets data_domain='unknown'."""
    unknown_model = UnknownLinearNet()
    sample_input = torch.randn(2, 10)

    adapter, metadata = ingest_model(
        model_path=unknown_model,
        sample_input=sample_input,
        model_name="AmbiguousNet",
        device="cpu",
    )

    assert metadata.data_domain == "unknown"
    assert metadata.domain == "unknown"
    assert adapter.processor is None
