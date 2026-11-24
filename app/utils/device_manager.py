"""
Device management module for handling PyTorch execution devices (CPU, CUDA, MPS).
Provides strict GPU binding and safe CPU fallback workload adjustment.
"""

from typing import Any, Optional, Tuple, Union
import torch

from app.utils.device import DeviceManager

__all__ = ["DeviceManager"]
