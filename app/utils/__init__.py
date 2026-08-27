"""
AdverScan Utilities Module.
Provides centralized hardware device resolution, model patch loading, and helper routines.
"""

from app.utils.device import (
    resolve_device,
    get_device,
    get_execution_device_info,
    to_device,
    DeviceManager,
)
from app.utils.model_utils import (
    patch_hf_config,
    load_gtsrb_vit_model,
)

__all__ = [
    "resolve_device",
    "get_device",
    "get_execution_device_info",
    "to_device",
    "DeviceManager",
    "patch_hf_config",
    "load_gtsrb_vit_model",
]
