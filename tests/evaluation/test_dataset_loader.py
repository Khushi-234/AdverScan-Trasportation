"""
Unit tests for GTSRBDatasetLoader image decoding and sample processing.
"""

import io
from PIL import Image
import pytest
from app.evaluation.dataset_loader import GTSRBDatasetLoader


def test_gtsrb_dataset_loader_decode_image_bytes():
    """Test decoding PIL Image from raw PNG bytes dict."""
    img = Image.new("RGB", (32, 32), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    sample = {"Path": {"bytes": raw_bytes}, "ClassId": 14}

    loader = GTSRBDatasetLoader.__new__(GTSRBDatasetLoader)
    decoded_img = loader._decode_image(sample)

    assert isinstance(decoded_img, Image.Image)
    assert decoded_img.size == (32, 32)
    assert decoded_img.mode == "RGB"


def test_gtsrb_dataset_loader_rgb_conversion():
    """Test explicit conversion of RGBA and grayscale images to RGB."""
    loader = GTSRBDatasetLoader.__new__(GTSRBDatasetLoader)

    # Test RGBA image
    rgba_img = Image.new("RGBA", (32, 32), color=(255, 0, 0, 128))
    sample_rgba = {"image": rgba_img}
    decoded_rgba = loader._decode_image(sample_rgba)
    assert decoded_rgba.mode == "RGB"

    # Test Grayscale image
    gray_img = Image.new("L", (32, 32), color=128)
    sample_gray = {"image": gray_img}
    decoded_gray = loader._decode_image(sample_gray)
    assert decoded_gray.mode == "RGB"


def test_gtsrb_dataset_loader_processor_output_shape(mocker=None):
    """Test image processor transforms PIL image into expected (N, 3, 224, 224) tensor shape."""
    from transformers import AutoImageProcessor
    import torch
    
    processor = AutoImageProcessor.from_pretrained("bazyl/gtsrb-model")
    img = Image.new("RGB", (32, 32), color="blue")
    processed = processor(images=[img, img], return_tensors="pt")
    
    assert "pixel_values" in processed
    tensor = processed["pixel_values"]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (2, 3, 224, 224)


def test_gtsrb_dataset_loader_decode_invalid_sample():
    """Test error handling when decoding sample with missing keys."""
    loader = GTSRBDatasetLoader.__new__(GTSRBDatasetLoader)
    with pytest.raises(KeyError, match="does not contain a valid image"):
        loader._decode_image({"invalid": "dict"})


def test_time_series_dataset_loader():
    """Test TimeSeriesDatasetLoader with 3D numerical sequence tensor inputs."""
    import torch
    from app.evaluation.dataset_loader import TimeSeriesDatasetLoader

    inputs = torch.randn(10, 24, 5)  # 10 samples, 24 time steps, 5 features
    targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    loader = TimeSeriesDatasetLoader(inputs=inputs, targets=targets, batch_size=4)
    assert len(loader) == 10

    batches = list(loader.iterate_batches())
    assert len(batches) == 3  # 4 + 4 + 2 samples
    b_in, b_tgt, t_list = batches[0]
    assert b_in.shape == (4, 24, 5)
    assert b_tgt.shape == (4,)
    assert t_list == [0, 1, 0, 1]


def test_tabular_dataset_loader_tuple():
    """Test TabularDatasetLoader with feature matrix and target vector tuple."""
    import torch
    from app.evaluation.dataset_loader import TabularDatasetLoader

    X = torch.randn(12, 8)  # 12 samples, 8 tabular features
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])

    loader = TabularDatasetLoader(data=(X, y), batch_size=5)
    assert len(loader) == 12

    batches = list(loader.iterate_batches())
    assert len(batches) == 3
    b_in, b_tgt, t_list = batches[0]
    assert b_in.shape == (5, 8)
    assert b_tgt.shape == (5,)
    assert t_list == [0, 1, 2, 0, 1]


def test_get_dataset_loader_factory_domains():
    """Test get_dataset_loader factory resolving correct domain classes."""
    import torch
    from app.evaluation.dataset_loader import (
        HFVisionDatasetLoader,
        TimeSeriesDatasetLoader,
        TabularDatasetLoader,
        get_dataset_loader,
    )

    # Time series domain
    ts_loader = get_dataset_loader(
        dataset_name="custom_ts",
        data_domain="time-series",
        inputs=torch.randn(4, 10),
        targets=torch.tensor([0, 1, 0, 1]),
    )
    assert isinstance(ts_loader, TimeSeriesDatasetLoader)

    # Tabular domain
    tab_loader = get_dataset_loader(
        dataset_name="custom_tab",
        data_domain="tabular",
        data=(torch.randn(4, 5), torch.tensor([0, 1, 0, 1])),
    )
    assert isinstance(tab_loader, TabularDatasetLoader)

    # Generic domain
    from app.evaluation.dataset_loader import GenericDatasetLoader
    gen_loader = get_dataset_loader(
        dataset_name="custom_gen",
        data_domain="generic",
        inputs=torch.randn(4, 5),
        targets=torch.tensor([0, 1, 0, 1]),
    )
    assert isinstance(gen_loader, GenericDatasetLoader)


def test_get_dataset_loader_validation_and_unsupported_domains():
    """Test get_dataset_loader error handling for missing args and unsupported domains."""
    import torch
    from app.evaluation.dataset_loader import get_dataset_loader

    # Unsupported domain
    with pytest.raises(ValueError, match="Unsupported data domain"):
        get_dataset_loader(dataset_name="unknown", data_domain="quantum_computing")

    # Time series missing inputs or targets
    with pytest.raises(ValueError, match="TimeSeriesDatasetLoader requires"):
        get_dataset_loader(dataset_name="ts", data_domain="time-series")

    # Tabular missing data
    with pytest.raises(ValueError, match="TabularDatasetLoader requires"):
        get_dataset_loader(dataset_name="tab", data_domain="tabular")

    # Generic missing inputs or targets
    with pytest.raises(ValueError, match="GenericDatasetLoader requires"):
        get_dataset_loader(dataset_name="gen", data_domain="generic")


def test_hf_vision_loader_invalid_processor_raises_runtime_error(monkeypatch):
    """Test that specifying an invalid/unresolvable processor_name raises a RuntimeError."""
    from app.evaluation.dataset_loader import HFVisionDatasetLoader

    class DummyDataset:
        column_names = ["image", "label"]
        features = {}
        def __len__(self):
            return 1
        def __getitem__(self, idx):
            return {"image": Image.new("RGB", (10, 10)), "label": 0}

    import app.evaluation.dataset_loader as dl_module
    monkeypatch.setattr(dl_module, "load_dataset", lambda *args, **kwargs: DummyDataset())

    with pytest.raises(RuntimeError, match="Failed to load AutoImageProcessor"):
        HFVisionDatasetLoader(
            dataset_name="dummy_dataset",
            processor_name="non_existent_processor_12345XYZ",
        )


def test_hf_vision_loader_missing_label_raises_key_error(monkeypatch):
    """Test that a dataset missing all recognized label columns raises KeyError."""
    from app.evaluation.dataset_loader import HFVisionDatasetLoader

    class DummyUnlabeledDataset:
        column_names = ["image", "unrelated_meta"]
        features = {}
        def __len__(self):
            return 1
        def __getitem__(self, idx):
            return {"image": Image.new("RGB", (10, 10)), "unrelated_meta": "test"}

    import app.evaluation.dataset_loader as dl_module
    monkeypatch.setattr(dl_module, "load_dataset", lambda *args, **kwargs: DummyUnlabeledDataset())

    with pytest.raises(KeyError, match="does not contain any recognized label column"):
        HFVisionDatasetLoader(
            dataset_name="unlabeled_dataset",
            processor_name=None,
        )


