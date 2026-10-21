"""
Comprehensive tests for preset registry models and user-provided custom models in AdverScan ingestion pipeline.
"""

from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.nn as nn

from app.ingestion import (
    HuggingFaceAdapter,
    HuggingFaceLoader,
    ImageProcessor,
    IngestionResult,
    ModelMetadata,
    ModelPresetRegistry,
    PyTorchAdapter,
    PyTorchLoader,
    TextProcessor,
    ingest_model,
)
from app.ingestion.exceptions import (
    ModelValidationError,
    UnsupportedDomainError,
    UnsupportedModelError,
)


class MockVisionModel(nn.Module):
    """Mock vision classifier returning ImageClassifierOutput with logits."""

    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(3, num_classes)

    def forward(self, pixel_values=None, *args, **kwargs):
        x = pixel_values if pixel_values is not None else (args[0] if args else None)
        pooled = self.pool(x).view(x.size(0), -1)
        logits = self.classifier(pooled)

        class Output:
            def __init__(self, l):
                self.logits = l
        return Output(logits)



class MockTextModel(nn.Module):
    """Mock text sequence classifier returning SequenceClassifierOutput with logits."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, input_ids=None, attention_mask=None, *args, **kwargs):
        x = input_ids if input_ids is not None else (args[0] if args else None)
        logits = self.classifier(x.float())

        class Output:
            def __init__(self, l):
                self.logits = l
        return Output(logits)


def test_registry_preset_lookup_and_ingestion():
    """Test that a preset model in model_registry.py is automatically resolved."""
    mock_model = MockVisionModel(num_classes=43)

    with patch.object(HuggingFaceLoader, "load", return_value=mock_model), \
         patch("transformers.AutoImageProcessor.from_pretrained", return_value=MagicMock()):
        sample_img = torch.randn(1, 3, 224, 224)
        result = ingest_model(
            model_path="bazyl/gtsrb-model",
            sample_input=sample_img,
            device="cpu",
        )


        # Verify IngestionResult container
        assert isinstance(result, IngestionResult)
        assert isinstance(result, tuple)
        assert len(result) == 2

        # Verify tuple unpacking
        adapter, metadata = result
        assert isinstance(adapter, HuggingFaceAdapter)
        assert isinstance(metadata, ModelMetadata)

        # Verify preset fields populated
        assert metadata.model_id == "bazyl/gtsrb-model"
        assert metadata.framework == "huggingface"
        assert metadata.data_domain == "image"
        assert metadata.num_classes == 43
        assert isinstance(result.processor, ImageProcessor)


def test_custom_user_provided_model_not_in_registry():
    """Test that an unregistered custom model does NOT fail and uses user runtime configuration."""
    assert not ModelPresetRegistry.has_preset("custom-org/my-private-model")

    mock_model = MockVisionModel(num_classes=8)

    with patch.object(HuggingFaceLoader, "load", return_value=mock_model), \
         patch("transformers.AutoImageProcessor.from_pretrained", return_value=MagicMock()):
        sample_img = torch.randn(2, 3, 224, 224)
        result = ingest_model(
            model_path="custom-org/my-private-model",
            dataset_name="custom_satellite_dataset",
            data_domain="image",
            task_type="classification",
            framework="huggingface",
            processor_name="custom-org/my-private-model",
            sample_input=sample_img,
            device="cpu",
        )

        adapter, metadata = result
        assert isinstance(adapter, HuggingFaceAdapter)
        assert isinstance(metadata, ModelMetadata)

        # Verify user-provided configuration is normalized into ModelMetadata
        assert metadata.model_id == "custom-org/my-private-model"
        assert metadata.dataset_name == "custom_satellite_dataset"
        assert metadata.data_domain == "image"
        assert metadata.framework == "huggingface"
        assert metadata.processor_name == "custom-org/my-private-model"
        assert metadata.num_classes == 8
        assert isinstance(result.processor, ImageProcessor)

        # Verify adapter interface works uniformly on custom model
        preds = adapter.predict(sample_img)
        assert isinstance(preds, torch.Tensor)
        assert preds.shape == (2, 8)

        logits = adapter.get_logits(sample_img)
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (2, 8)


def test_custom_text_domain_model_ingestion():
    """Test ingesting custom text NLP model selecting TextProcessor and HuggingFaceAdapter."""
    mock_model = MockTextModel(num_classes=3)

    with patch.object(HuggingFaceLoader, "load", return_value=mock_model), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()):
        sample_tokens = torch.randn(2, 16)
        result = ingest_model(
            model_path="custom-nlp/sentiment-roberta",
            data_domain="text",
            task_type="classification",
            framework="huggingface",
            sample_input=sample_tokens,
            device="cpu",
        )


        adapter, metadata = result
        assert isinstance(adapter, HuggingFaceAdapter)
        assert metadata.data_domain == "text"
        assert isinstance(result.processor, TextProcessor)
        assert result.processor.data_domain == "text"

        # Verify inference
        out = adapter.predict(sample_tokens)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 3)


def test_custom_pytorch_framework_selection():
    """Test that framework='pytorch' selects PyTorchLoader and PyTorchAdapter."""
    mock_net = MockVisionModel(num_classes=4)

    with patch.object(PyTorchLoader, "load", return_value=mock_net):
        sample_in = torch.randn(1, 3, 32, 32)
        result = ingest_model(
            model_path="checkpoints/my_custom_weights.pt",
            framework="pytorch",
            data_domain="image",
            sample_input=sample_in,
            device="cpu",
        )

        adapter, metadata = result
        assert isinstance(adapter, PyTorchAdapter)
        assert metadata.framework == "pytorch"
        assert isinstance(result.processor, ImageProcessor)


def test_unsupported_domain_rejection():
    """Test that non-image and non-text domains are strictly rejected during validation."""
    with pytest.raises(UnsupportedDomainError, match="Unsupported data domain"):
        ingest_model(
            model_path="audio-model-1",
            data_domain="audio",
            framework="huggingface",
        )

    with pytest.raises(UnsupportedDomainError, match="Unsupported data domain"):
        ingest_model(
            model_path="graph-model-1",
            data_domain="graph",
            framework="pytorch",
        )


def test_unsupported_framework_rejection():
    """Test that unsupported frameworks are strictly rejected during validation."""
    with pytest.raises(UnsupportedModelError, match="Unsupported framework"):
        ingest_model(
            model_path="model-1",
            data_domain="image",
            framework="jax",
        )


def test_missing_model_id_rejection():
    """Test that empty or None model_id is rejected."""
    with pytest.raises(ModelValidationError, match="'model_id' is required"):
        ingest_model(
            model_path="",
            data_domain="image",
            framework="huggingface",
        )


def test_adapter_common_interface_duck_typing():
    """
    Test that both PyTorchAdapter and HuggingFaceAdapter expose the required common interface:
    predict(), predict_batch(), get_logits(), eval(), to()
    """
    py_model = MockVisionModel(num_classes=2)
    hf_model = MockVisionModel(num_classes=2)

    py_adapter = PyTorchAdapter(py_model, device="cpu")
    hf_adapter = HuggingFaceAdapter(hf_model, device="cpu", data_domain="image")

    sample = torch.randn(2, 3, 32, 32)

    for adapter in [py_adapter, hf_adapter]:
        # predict
        p = adapter.predict(sample)
        assert isinstance(p, torch.Tensor)
        assert p.shape == (2, 2)

        # predict_batch
        pb = adapter.predict_batch(sample)
        assert isinstance(pb, torch.Tensor)
        assert pb.shape == (2, 2)

        # get_logits
        l = adapter.get_logits(sample)
        assert isinstance(l, torch.Tensor)
        assert l.shape == (2, 2)

        # eval
        adapter.eval()
        assert not adapter.get_model().training

        # to
        adapter.to("cpu")
        assert str(adapter.device) == "cpu"
