"""
Base attack interface for standardized adversarial attack generators in AdverScan.
"""

from abc import ABC, abstractmethod
from typing import Any
import torch


class BaseAttack(ABC):
    """
    Abstract base class for adversarial attack generators.
    Wraps standardized M1 BaseModelAdapter instances or raw PyTorch models.
    """

    def __init__(self, model_adapter: Any, device: str = "cpu"):
        """
        Initialize BaseAttack.

        Args:
            model_adapter: Module 1 BaseModelAdapter instance or PyTorch model.
            device: Target execution device ('cuda' or 'cpu').
        """
        self.adapter = model_adapter
        self.device = device
        
        # Extract underlying raw model if adapter is provided
        if hasattr(model_adapter, "get_model"):
            self.raw_model = model_adapter.get_model()
        else:
            self.raw_model = model_adapter

    @abstractmethod
    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Generate adversarial perturbations for input tensor x given target labels y.

        Args:
            x: Input image tensor on target device.
            y: Target ground truth labels tensor on target device.

        Returns:
            torch.Tensor: Perturbed adversarial image tensor.
        """
        pass
