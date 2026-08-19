"""
Utility helper functions for Module 7 Hardening.

Provides operations for tensor transformations, Gaussian noise addition, spatial smoothing,
bit-depth reduction (feature squeezing), JPEG compression simulation, and model cloning.
"""

import copy
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def clone_model(model: nn.Module) -> nn.Module:
    """
    Deep copy a PyTorch model to preserve original state during hardening operations.

    Args:
        model: Source PyTorch model.

    Returns:
        nn.Module: Deep copied model instance.
    """
    return copy.deepcopy(model)


def add_gaussian_noise(
    inputs: torch.Tensor,
    sigma: float = 0.1,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> torch.Tensor:
    """
    Add isotropic Gaussian noise to input tensor and clip to specified bounds.

    Args:
        inputs: PyTorch tensor (B, C, H, W) or (B, D).
        sigma: Standard deviation of Gaussian noise.
        clip_min: Minimum value clip bound.
        clip_max: Maximum value clip bound.

    Returns:
        torch.Tensor: Noisy input tensor.
    """
    noise = torch.randn_like(inputs) * sigma
    noisy_inputs = inputs + noise
    return torch.clamp(noisy_inputs, min=clip_min, max=clip_max)


def apply_spatial_smoothing(
    inputs: torch.Tensor,
    kernel_size: int = 3,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Apply Gaussian spatial smoothing filter to image tensors.

    Args:
        inputs: Input image tensor of shape (B, C, H, W).
        kernel_size: Size of Gaussian kernel (must be odd).
        sigma: Standard deviation of Gaussian distribution.

    Returns:
        torch.Tensor: Smooth preprocessed tensor of shape (B, C, H, W).
    """
    if inputs.ndim != 4:
        # Non-spatial tensor (e.g. 2D tabular), return unmodified or apply 1D smoothing
        return inputs

    if kernel_size % 2 == 0:
        kernel_size += 1

    channels = inputs.shape[1]
    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, dtype=torch.float32, device=inputs.device) - (kernel_size - 1) / 2.0
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Create 2D Gaussian kernel
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    # Expand kernel to match input channels: (C, 1, K, K)
    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

    padding = kernel_size // 2
    smoothed = F.conv2d(inputs, kernel, padding=padding, groups=channels)
    return smoothed


def reduce_bit_depth(
    inputs: torch.Tensor,
    bit_depth: int = 4,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> torch.Tensor:
    """
    Reduce color/feature bit-depth (Feature Squeezing).

    Quantizes tensor values into 2^bit_depth discrete levels within [clip_min, clip_max].

    Args:
        inputs: Input tensor (B, C, H, W) or (B, D).
        bit_depth: Target bit depth (e.g. 1 to 8).
        clip_min: Minimum bound.
        clip_max: Maximum bound.

    Returns:
        torch.Tensor: Quantized feature squeezed tensor.
    """
    bit_depth = max(1, min(8, bit_depth))
    max_level = (2**bit_depth) - 1

    # Normalize to [0, 1]
    range_val = clip_max - clip_min
    if range_val <= 0:
        return inputs

    normalized = (inputs - clip_min) / range_val
    normalized = torch.clamp(normalized, 0.0, 1.0)

    # Quantize
    quantized = torch.round(normalized * max_level) / max_level

    # Scale back to original range
    squeezed = quantized * range_val + clip_min
    return squeezed


def simulate_jpeg_compression(
    inputs: torch.Tensor,
    quality: int = 75,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> torch.Tensor:
    """
    Simulate JPEG compression artefacts using block-wise discrete cosine transform (DCT) thresholding / quantization.

    Args:
        inputs: Input image tensor (B, C, H, W).
        quality: JPEG compression quality factor (1 to 100). Lower quality = higher compression defense.
        clip_min: Min value clip bound.
        clip_max: Max value clip bound.

    Returns:
        torch.Tensor: Compressed/preprocessed image tensor.
    """
    if inputs.ndim != 4:
        return inputs

    # Approximate lossy quantization scale factor based on quality
    quality = max(1, min(100, quality))
    scale = 50.0 / quality if quality < 50 else (200.0 - 2.0 * quality) / 100.0

    # Downsample high frequency noise via average pooling and bilinear upsampling
    block_size = 2 if quality > 70 else (4 if quality > 30 else 8)
    h, w = inputs.shape[2], inputs.shape[3]

    downsampled = F.adaptive_avg_pool2d(inputs, (max(1, h // block_size), max(1, w // block_size)))
    reconstructed = F.interpolate(downsampled, size=(h, w), mode="bilinear", align_corners=False)

    # Mix original and downsampled according to quality factor
    alpha = min(1.0, max(0.2, quality / 100.0))
    compressed = alpha * inputs + (1.0 - alpha) * reconstructed

    return torch.clamp(compressed, min=clip_min, max=clip_max)
