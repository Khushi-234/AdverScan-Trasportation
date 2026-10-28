"""
Device management module for hardware execution (CPU, CUDA, MPS) across AdverScan.
"""

from typing import Any, Optional, Tuple, Union
import torch


class DeviceManager:
    """
    Manages hardware execution devices for machine learning models.
    Enforces strict GPU binding when CUDA is available and provides
    safe CPU workload adjustment for large models.
    """

    @staticmethod
    def resolve_device(
        requested_device: Optional[Union[str, torch.device]] = None,
        model_param_count: int = 0,
    ) -> torch.device:
        """
        Resolve target execution device.
        When CUDA is available, strictly binds to GPU.
        When CPU is used, ensures fallback safety.
        """
        has_cuda = torch.cuda.is_available()
        dev_str = str(requested_device).lower() if requested_device is not None else ""

        if has_cuda:
            # If user explicitly requested CPU and model is small (< 80M params), allow CPU
            if dev_str == "cpu" and model_param_count < 80_000_000:
                return torch.device("cpu")
            # Strictly bind to CUDA when GPU is present
            device = torch.device("cuda:0" if dev_str.startswith("cuda:") else "cuda")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return device
        else:
            # Enforce CPU execution
            return torch.device("cpu")

    @staticmethod
    def get_device(
        preferred_device: Optional[Union[str, torch.device]] = None,
        model_param_count: int = 0,
    ) -> torch.device:
        """
        Determine and return the appropriate execution device.
        """
        return DeviceManager.resolve_device(preferred_device, model_param_count)

    @staticmethod
    def adjust_workload_for_hardware(
        sample_count: int,
        model_param_count: int,
        device: Union[str, torch.device],
    ) -> Tuple[int, int]:
        """
        If running on CPU and model is large (>= 80M), restrict sample count to prevent hangs and OOMs.
        Returns:
            Tuple[int, int]: (safe_sample_count, safe_batch_size)
        """
        dev_type = device.type if isinstance(device, torch.device) else str(device).lower()
        if dev_type == "cpu" and model_param_count >= 80_000_000:
            safe_limit = min(sample_count, 10) if sample_count > 0 else 10
            if safe_limit < sample_count or sample_count == 0:
                print(f"⚠️ [Hardware Guard] Large model ({model_param_count:,} params) detected on CPU.")
                print(f"⚠️ Automatically clamping evaluation samples from {sample_count} to {safe_limit} to prevent CPU memory freeze.")
            return safe_limit, 8  # safe_samples, safe_batch_size
        return sample_count, 32

    @staticmethod
    def to_device(obj: Any, device: Union[str, torch.device]) -> Any:
        """
        Move a model or tensor object to the specified target device.
        """
        target_device = device if isinstance(device, torch.device) else DeviceManager.get_device(device)
        if hasattr(obj, "to"):
            return obj.to(target_device)
        return obj
