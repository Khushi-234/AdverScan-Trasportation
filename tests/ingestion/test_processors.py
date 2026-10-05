"""
Unit tests for ImageProcessor, TextProcessor, and BaseProcessor implementations.
"""

import numpy as np
from PIL import Image
import pytest
import torch

from app.ingestion.processors.base_processor import BaseProcessor
from app.ingestion.processors.image_processor import ImageProcessor
from app.ingestion.processors.text_processor import TextProcessor


def test_base_processor_abstract():
    """Ensure BaseProcessor cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseProcessor()  # type: ignore[abstract]


def test_image_processor_tensor_input():
    """Test ImageProcessor with already prepared torch.Tensor input."""
    processor = ImageProcessor()
    assert processor.data_domain == "image"

    # 4D tensor input
    tensor_4d = torch.randn(2, 3, 224, 224)
    out = processor.process(tensor_4d)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 3, 224, 224)

    # 3D tensor input unsqueezed
    tensor_3d = torch.randn(3, 224, 224)
    out_unsqueezed = processor.process(tensor_3d)
    assert out_unsqueezed.shape == (1, 3, 224, 224)


def test_image_processor_pil_and_numpy_input():
    """Test ImageProcessor with PIL Image and numpy array inputs."""
    processor = ImageProcessor(input_resolution=(3, 64, 64))

    # PIL Image
    pil_img = Image.new("RGB", (32, 32), color="red")
    out = processor(pil_img)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 3, 64, 64)

    # List of PIL Images
    img1 = Image.new("RGB", (32, 32), color="blue")
    img2 = Image.new("RGB", (32, 32), color="green")
    out_batch = processor([img1, img2])
    assert out_batch.shape == (2, 3, 64, 64)

    # NumPy array (H, W, C)
    np_img = np.zeros((32, 32, 3), dtype=np.uint8)
    out_np = processor(np_img)
    assert out_np.shape == (1, 3, 64, 64)


def test_text_processor_string_and_list_input():
    """Test TextProcessor with single string and batch text inputs."""
    processor = TextProcessor(max_length=16)
    assert processor.data_domain == "text"

    # Single string
    res = processor("AdverScan robust test")
    assert isinstance(res, dict)
    assert "input_ids" in res
    assert "attention_mask" in res
    assert res["input_ids"].shape[0] == 1
    assert res["input_ids"].shape[1] == 16

    # Batch of strings
    batch_res = processor(["Hello world", "Adversarial testing with AdverScan"])
    assert batch_res["input_ids"].shape == (2, 16)
    assert batch_res["attention_mask"].shape == (2, 16)


def test_text_processor_dict_and_tensor_input():
    """Test TextProcessor with pre-tokenized dictionary and tensor inputs."""
    processor = TextProcessor(max_length=10)

    # Tensor input
    t = torch.tensor([[101, 2023, 102]])
    res_t = processor(t)
    assert isinstance(res_t, dict)
    assert "input_ids" in res_t
    assert torch.equal(res_t["input_ids"], t)

    # Dict input
    d = {"input_ids": t, "attention_mask": torch.ones_like(t)}
    res_d = processor(d)
    assert isinstance(res_d, dict)
    assert "input_ids" in res_d
    assert torch.equal(res_d["input_ids"], t)


def test_text_processor_userdict_mapping_input():
    """Test TextProcessor with collections.UserDict (like transformers.BatchEncoding)."""
    from collections import UserDict
    processor = TextProcessor(max_length=10)
    t = torch.tensor([[101, 102], [103, 104]])
    mapping_input = UserDict({"input_ids": t, "attention_mask": torch.ones_like(t)})
    res = processor(mapping_input)
    assert isinstance(res, dict)
    assert "input_ids" in res
    assert res["input_ids"].shape == (2, 2)
    assert torch.equal(res["input_ids"], t)
