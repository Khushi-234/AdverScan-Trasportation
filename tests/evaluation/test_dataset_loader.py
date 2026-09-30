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

