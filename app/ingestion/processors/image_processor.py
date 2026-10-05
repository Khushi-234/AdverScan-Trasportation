"""
Image processor implementation for vision model inputs.
"""

from typing import Any, List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image

from app.ingestion.processors.base_processor import BaseProcessor


class ImageProcessor(BaseProcessor):
    """
    Standard image input processor.
    Prepares vision model inputs (tensors with shape [B, C, H, W]) using
    AutoImageProcessor or fallback transforms.
    """

    def __init__(
        self,
        processor_name_or_path: Optional[str] = None,
        processor: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
        input_resolution: Optional[Tuple[int, int, int]] = None,
    ):
        self.processor_name = processor_name_or_path
        self._device = torch.device(device) if device is not None else None
        self.input_resolution = input_resolution or (3, 224, 224)
        self._processor = processor

        if self._processor is None and processor_name_or_path:
            p_str = str(processor_name_or_path).strip()
            if not p_str.endswith((".pt", ".pth", ".bin", ".ckpt", ".tar", ".gz")):
                try:
                    from transformers import AutoImageProcessor
                    try:
                        self._processor = AutoImageProcessor.from_pretrained(
                            processor_name_or_path, local_files_only=True
                        )
                    except Exception:
                        if "/" in p_str:
                            self._processor = AutoImageProcessor.from_pretrained(
                                processor_name_or_path
                            )
                except Exception:
                    self._processor = None


    @property
    def data_domain(self) -> str:
        return "image"

    def get_processor(self) -> Any:
        return self._processor

    def _convert_single_image(self, img: Any) -> Image.Image:
        """Convert a single image object to RGB PIL Image."""
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        elif isinstance(img, np.ndarray):
            if img.dtype != np.uint8 and np.max(img) <= 1.0:
                img = (img * 255).astype(np.uint8)
            return Image.fromarray(img).convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

    def process(self, inputs: Any, **kwargs: Any) -> torch.Tensor:
        """
        Process raw image inputs into PyTorch tensor of shape (B, C, H, W).
        """
        from collections.abc import Mapping

        # If already a torch Tensor, ensure 4D (batch, C, H, W) and correct device
        if isinstance(inputs, torch.Tensor):
            tensor = inputs
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if self._device is not None:
                tensor = tensor.to(self._device)
            return tensor

        # If inputs is a mapping / dict containing pixel_values
        if isinstance(inputs, Mapping) or hasattr(inputs, "items"):
            if "pixel_values" in inputs:
                tensor = inputs["pixel_values"]
                if isinstance(tensor, torch.Tensor):
                    if tensor.ndim == 3:
                        tensor = tensor.unsqueeze(0)
                    if self._device is not None:
                        tensor = tensor.to(self._device)
                    return tensor

        # If AutoImageProcessor is configured and available
        if self._processor is not None:
            try:
                processed = self._processor(images=inputs, return_tensors="pt", **kwargs)
                if isinstance(processed, dict) and "pixel_values" in processed:
                    tensor = processed["pixel_values"]
                elif hasattr(processed, "pixel_values"):
                    tensor = processed.pixel_values
                else:
                    tensor = processed
                if isinstance(tensor, torch.Tensor) and self._device is not None:
                    tensor = tensor.to(self._device)
                return tensor
            except Exception:
                pass  # Fall back to manual PIL transform

        # Fallback transform using PIL
        images: List[Any] = inputs if isinstance(inputs, (list, tuple)) else [inputs]
        pil_images = [self._convert_single_image(im) for im in images]

        target_h = self.input_resolution[1] if len(self.input_resolution) == 3 else 224
        target_w = self.input_resolution[2] if len(self.input_resolution) == 3 else 224

        tensors = []
        for pim in pil_images:
            resized = pim.resize((target_w, target_h), Image.Resampling.BILINEAR)
            arr = np.array(resized, dtype=np.float32) / 255.0  # (H, W, C)
            arr = np.transpose(arr, (2, 0, 1))  # (C, H, W)
            # Standard ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
            norm_arr = (arr - mean) / std
            tensors.append(torch.from_numpy(norm_arr))

        batch_tensor = torch.stack(tensors, dim=0)
        if self._device is not None:
            batch_tensor = batch_tensor.to(self._device)
        return batch_tensor
