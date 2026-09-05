"""
Reproducibility Manager module for AdverScan (Module 10 / Utilities).
Provides random seed control, deterministic execution configuration, and environment/reproducibility metadata collection.
"""

import os
import platform
import random
import sys
from typing import Any, Dict, Optional
import numpy as np
import torch


class ReproducibilityManager:
    """
    Manager for setting reproducible seeds across Python random, NumPy, PyTorch CPU/CUDA,
    configuring PyTorch deterministic flags, and gathering environment reproducibility metadata.
    """

    @staticmethod
    def set_seed(seed: int = 42, deterministic: bool = False) -> None:
        """
        Set global random seed across Python random, NumPy, PyTorch CPU, and PyTorch CUDA (if available).

        Args:
            seed: Integer random seed value.
            deterministic: Whether to enforce PyTorch deterministic execution algorithms.
        """
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError(f"Invalid seed '{seed}'. Must be a non-negative integer.")

        # Python random
        random.seed(seed)

        # NumPy
        np.random.seed(seed)

        # PyTorch CPU
        torch.manual_seed(seed)

        # PyTorch CUDA (if available)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Python hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)

        # Configure deterministic flags
        ReproducibilityManager.set_deterministic(deterministic)

    @staticmethod
    def set_deterministic(deterministic: bool = False) -> None:
        """
        Configure PyTorch deterministic behavior.

        Args:
            deterministic: If True, enables PyTorch deterministic algorithms and disables CUDNN benchmarking.
        """
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except Exception:
                    pass
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(False)
                except Exception:
                    pass

    @staticmethod
    def collect_environment_metadata(
        seed: Optional[int] = None,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """
        Collect reproducibility and hardware environment metadata.

        Args:
            seed: Configured seed value.
            deterministic: Configured deterministic flag.

        Returns:
            Dict containing Python, platform, PyTorch, CUDA/GPU, CPU, and reproducibility settings.
        """
        cuda_available = torch.cuda.is_available()
        cuda_version = getattr(torch.version, "cuda", None) if cuda_available else None
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"

        cpu_info = platform.processor() or platform.machine() or "N/A"

        return {
            "python_version": platform.python_version(),
            "operating_system": platform.platform(),
            "pytorch_version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_version": cuda_version if cuda_version else "N/A",
            "gpu_name": gpu_name,
            "cpu_info": cpu_info,
            "reproducibility_settings": {
                "seed": seed,
                "deterministic": deterministic,
            },
        }

    @staticmethod
    def collect_identity_metadata(
        config: Optional[Any] = None,
        adapter: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Collect model and dataset identity metadata where available.

        Args:
            config: PipelineConfig instance or object.
            adapter: BaseModelAdapter instance or object.

        Returns:
            Dict containing model/dataset identity metadata and hash availability notes.
        """
        model_name = getattr(config, "model_name", "N/A") if config else "N/A"
        model_path = str(getattr(config, "model_path", "N/A")) if config else "N/A"
        dataset_name = getattr(config, "dataset_name", "N/A") if config else "N/A"
        split = getattr(config, "split", "N/A") if config else "N/A"

        return {
            "model_identity": {
                "model_name": model_name,
                "model_path": model_path,
                "model_hash": "N/A (model checksum hashing not implemented for dynamic adapter/hub models)",
            },
            "dataset_identity": {
                "dataset_name": dataset_name,
                "split": split,
                "dataset_checksum": "N/A (huggingface dataset hash dynamically resolved)",
            },
        }
