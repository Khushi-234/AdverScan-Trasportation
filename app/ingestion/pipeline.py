"""
Ingestion pipeline logic for AdverScan framework.
"""

from typing import Any, Optional, Tuple, Union
import torch

from app.ingestion.adapters.pytorch_adapter import PyTorchAdapter
from app.ingestion.loader.pytorch_loader import PyTorchLoader
from app.ingestion.metadata.model_metadata import ModelMetadata
from app.ingestion.runtime.device_manager import DeviceManager
from app.ingestion.validation.validator import ModelValidator


def ingest_model(
    model_path: Union[str, Any],
    sample_input: Optional[Any] = None,
    device: Optional[Union[str, torch.device]] = None,
    model_class: Optional[Any] = None,
    model_name: Optional[str] = None,
    task_type: Optional[str] = "classification",
    **kwargs: Any
) -> Tuple[PyTorchAdapter, ModelMetadata]:
    """
    Standard ingestion pipeline function for loading, adapting, validating,
    and generating metadata for PyTorch models.

    Args:
        model_path: Path to model checkpoint file or an existing PyTorch model instance.
        sample_input: Optional sample input for validation and shape inference.
        device: Target execution device (e.g., 'cuda', 'cpu').
        model_class: Model class factory or structure when loading state_dict.
        model_name: Optional custom model name.
        task_type: Machine learning task type (default 'classification').
        **kwargs: Extra parameters passed to model loader.

    Returns:
        Tuple containing (PyTorchAdapter, ModelMetadata).
    """
    # 1. Device selection
    target_device = DeviceManager.get_device(device)

    # 2. Model loading
    loader = PyTorchLoader()
    raw_model = loader.load(model_path, model_class=model_class, device=target_device, **kwargs)

    # 3. Model adaptation
    adapter = PyTorchAdapter(raw_model, device=target_device)

    # 4. Validation
    ModelValidator.validate(adapter, sample_input=sample_input)

    # 5. Metadata extraction
    input_shape = None
    output_shape = None
    num_classes = None

    if sample_input is not None:
        if isinstance(sample_input, torch.Tensor):
            input_shape = tuple(sample_input.shape)
        elif hasattr(sample_input, "shape"):
            input_shape = tuple(sample_input.shape)

        out = adapter.predict(sample_input)
        if isinstance(out, torch.Tensor):
            output_shape = tuple(out.shape)
            if len(output_shape) > 1:
                num_classes = output_shape[-1]

    metadata = ModelMetadata(
        framework="pytorch",
        model_name=model_name or (raw_model.__class__.__name__ if hasattr(raw_model, "__class__") else "PyTorchModel"),
        input_shape=input_shape,
        output_shape=output_shape,
        num_classes=num_classes,
        task_type=task_type,
        device=str(target_device),
    )

    return adapter, metadata
