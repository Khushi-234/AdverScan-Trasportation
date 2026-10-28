"""
Unit tests for ModelMetadata dataclass and containers.
"""

from app.ingestion.metadata.model_metadata import ModelMetadata


def test_model_metadata_defaults():
    """Test default values of ModelMetadata container."""
    metadata = ModelMetadata()

    assert metadata.framework == "pytorch"
    assert metadata.model_name is None
    assert metadata.input_shape is None
    assert metadata.output_shape is None
    assert metadata.num_classes is None
    assert metadata.task_type == "classification"
    assert metadata.device == "cpu"
    assert isinstance(metadata.extra_info, dict)


def test_model_metadata_custom():
    """Test custom metadata values and to_dict conversion."""
    metadata = ModelMetadata(
        framework="pytorch",
        model_name="ResNet18",
        input_shape=(1, 3, 224, 224),
        output_shape=(1, 1000),
        num_classes=1000,
        task_type="classification",
        device="cuda",
        extra_info={"author": "AdverScan"},
    )

    data = metadata.to_dict()

    assert data["framework"] == "pytorch"
    assert data["model_name"] == "ResNet18"
    assert data["input_shape"] == (1, 3, 224, 224)
    assert data["output_shape"] == (1, 1000)
    assert data["num_classes"] == 1000
    assert data["task_type"] == "classification"
    assert data["device"] == "cuda"
    assert data["extra_info"] == {"author": "AdverScan"}
