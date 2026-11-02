"""
Model Ingestion Service for AdverScan Frontend.

Bridges the Streamlit UI to the existing backend ingestion engine (Module 1)
and evaluation dataset loader without duplicating backend logic.
Extracts comprehensive telemetry matching scripts/run_pipeline_for_multimodel.py.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple
import torch

from app.evaluation.dataset_loader.factory import get_dataset_loader
from app.ingestion.exceptions import (
    ConfigurationError,
    DatasetFormatError,
    ModelIngestionError,
    ModelLoadError,
    ModelValidationError,
    UnsupportedDomainError,
    UnsupportedModelError,
    UnsupportedTaskError,
)
from app.ingestion.pipeline import IngestionResult, ingest_model
from app.ingestion.presets import MODEL_PRESETS


def get_available_presets() -> Dict[str, Dict[str, Any]]:
    """Retrieve existing model presets configured in the backend."""
    return MODEL_PRESETS


def load_model_and_dataset(
    model_identifier: Any,
    dataset_name: Optional[str] = None,
    data_domain: Optional[str] = None,
    framework: Optional[str] = None,
    device: Optional[str] = None,
    sample_size: int = 10,
) -> Dict[str, Any]:
    """
    Ingest a model using the existing backend pipeline, extract complete telemetry,
    and prepare an evaluation sample batch.

    Returns:
        Structured dictionary matching the telemetry in run_pipeline_for_multimodel.py:
            - "adapter": The BaseModelAdapter wrapping the loaded model.
            - "raw_model": The underlying model instance.
            - "metadata": ModelMetadata dataclass instance.
            - "telemetry": Comprehensive model & architecture telemetry dictionary.
            - "test_inputs": Sample inputs tensor for attack execution.
            - "test_labels": Sample labels tensor for attack execution.
            - "dataset_name": Resolved dataset name used.
            - "domain": Resolved data domain.
            - "ingestion_time": Exact duration in seconds.
    """
    if model_identifier is None or (
        isinstance(model_identifier, str) and not model_identifier.strip()
    ):
        raise ConfigurationError("Model identifier or path must not be empty.")

    clean_id = (
        model_identifier.strip()
        if isinstance(model_identifier, str)
        else getattr(model_identifier, "__name__", type(model_identifier).__name__)
    )

    # Determine execution device
    exec_device = device if device and device != "Auto-detect" else None

    # Step 1: Ingest model using existing backend ingestion pipeline and measure time
    t0 = time.time()
    try:
        ingestion_res: IngestionResult = ingest_model(
            model_path=model_identifier,
            dataset_name=dataset_name,
            data_domain=data_domain,
            framework=framework,
            device=exec_device,
        )
    except (
        ModelLoadError,
        ModelValidationError,
        ConfigurationError,
        DatasetFormatError,
        UnsupportedDomainError,
        UnsupportedModelError,
        UnsupportedTaskError,
    ) as known_err:
        raise known_err
    except Exception as exc:
        raise ModelIngestionError(
            f"Failed to ingest model '{clean_id}': {str(exc)}"
        ) from exc

    ingestion_time = time.time() - t0

    adapter = ingestion_res.adapter
    metadata = ingestion_res.metadata
    raw_model = getattr(adapter, "get_model", lambda: adapter)()
    preset_info = MODEL_PRESETS.get(clean_id, {})
    model_name = (
        preset_info.get("name") or getattr(metadata, "model_name", None) or clean_id
    )

    # Step 2: Compute parameter counts & estimated memory size from actual tensors
    total_params = None
    trainable_params = None
    trainable_pct = 100.0
    param_bytes = 0
    buffer_bytes = 0

    target_for_params = raw_model if hasattr(raw_model, "parameters") else adapter
    if hasattr(target_for_params, "parameters") and callable(
        getattr(target_for_params, "parameters")
    ):
        try:
            params_list = list(target_for_params.parameters())
            if params_list:
                total_params = sum(p.numel() for p in params_list)
                trainable_params = sum(
                    p.numel() for p in params_list if getattr(p, "requires_grad", False)
                )
                param_bytes = sum(
                    p.numel() * p.element_size()
                    for p in params_list
                    if hasattr(p, "element_size")
                )
                if total_params > 0:
                    trainable_pct = (trainable_params / total_params) * 100.0
        except Exception:
            pass

    if hasattr(target_for_params, "buffers") and callable(
        getattr(target_for_params, "buffers")
    ):
        try:
            buffers_list = list(target_for_params.buffers())
            if buffers_list:
                buffer_bytes = sum(
                    b.numel() * b.element_size()
                    for b in buffers_list
                    if hasattr(b, "element_size")
                )
        except Exception:
            pass

    if (total_params is None or total_params == 0) and preset_info.get("parameters"):
        total_params = preset_info.get("parameters")
        trainable_params = total_params

    # Calculate actual stored parameter and buffer memory
    if param_bytes + buffer_bytes > 0:
        estimated_size_mb = (param_bytes + buffer_bytes) / (1024.0 * 1024.0)
    elif total_params is not None and total_params > 0:
        estimated_size_mb = (total_params * 4.0) / (1024.0 * 1024.0)
    else:
        estimated_size_mb = None

    # Architecture discovery: check preset, config architectures, and class hierarchy
    arch_name = getattr(metadata, "model_architecture", None) or preset_info.get(
        "model_type"
    )
    if not arch_name and hasattr(raw_model, "config"):
        cfg = getattr(raw_model, "config")
        architectures = getattr(cfg, "architectures", None)
        if architectures and isinstance(architectures, list) and architectures[0]:
            arch_name = str(architectures[0])
        elif getattr(cfg, "model_type", None):
            arch_name = f"{str(cfg.model_type).capitalize()} Architecture"

    if not arch_name and raw_model is not None:
        cls_name = getattr(raw_model, "__class__", type(raw_model)).__name__
        if cls_name not in ("NoneType", "object", "BaseModelAdapter"):
            arch_name = cls_name

    if not arch_name:
        arch_name = "Unknown"

    # Step 3: Determine resolved domain and dataset
    resolved_domain = (
        (metadata.data_domain or data_domain or "image").lower().replace("_", "-")
    )

    # Execution device
    if hasattr(adapter, "device"):
        device_name = str(adapter.device).upper()
    elif exec_device:
        device_name = str(exec_device).upper()
    elif torch.cuda.is_available():
        device_name = "CUDA (GPU)"
    else:
        device_name = "CPU"

    # Step 4: Load a sample evaluation batch using existing dataset loader factory
    test_inputs: Optional[torch.Tensor] = None
    test_labels: Optional[torch.Tensor] = None
    is_synthetic_dataset = False
    resolved_dataset = dataset_name or metadata.dataset_name or clean_id

    try:
        loader = get_dataset_loader(
            dataset_name=resolved_dataset,
            data_domain=resolved_domain,
            processor_name=metadata.processor_name or clean_id,
            split=f"test[:{sample_size}]",
            batch_size=sample_size,
        )

        for batch in loader.iterate_batches():
            test_inputs = batch[0]
            test_labels = batch[1]
            break
    except Exception:
        test_inputs, test_labels = _generate_domain_sample_batch(metadata, sample_size)
        is_synthetic_dataset = True

    if test_inputs is None or test_labels is None:
        test_inputs, test_labels = _generate_domain_sample_batch(metadata, sample_size)
        is_synthetic_dataset = True

    actual_evaluated_samples = (
        int(test_inputs.shape[0]) if test_inputs is not None else sample_size
    )

    # Class resolution
    num_classes = getattr(metadata, "num_classes", None) or preset_info.get(
        "num_classes"
    )
    if num_classes is None and hasattr(raw_model, "config"):
        cfg = getattr(raw_model, "config")
        num_labels = getattr(cfg, "num_labels", None)
        if num_labels is not None:
            num_classes = int(num_labels)
        elif getattr(cfg, "id2label", None) and isinstance(cfg.id2label, dict):
            num_classes = len(cfg.id2label)
    if num_classes is None and hasattr(raw_model, "num_classes"):
        num_classes = getattr(raw_model, "num_classes")

    # Capabilities & gradient access
    supports_gradients = getattr(
        getattr(adapter, "capabilities", None), "supports_gradients", True
    )
    gradient_access = (
        "Full Backprop (White-Box)"
        if supports_gradients
        else "Black-Box (Inference Only)"
    )

    dataset_display_name = (
        "In-memory dataset"
        if is_synthetic_dataset and not dataset_name
        else resolved_dataset
    )

    # Observed input tensor shape
    observed_input_shape = (
        list(test_inputs.shape[1:])
        if test_inputs is not None
        else (preset_info.get("input_resolution") or metadata.input_shape or None)
    )

    # Build rich telemetry contract (aligned with run_pipeline_for_multimodel.py)
    telemetry = {
        "model_identifier": clean_id,
        "model_name": model_name,
        "architecture": arch_name,
        "framework": str(metadata.framework or "PyTorch").upper(),
        "domain": str(resolved_domain).capitalize(),
        "task_type": str(
            metadata.task_type or preset_info.get("task_type") or "Classification"
        ).capitalize(),
        "num_classes": num_classes if num_classes is not None else "Not available",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_pct": trainable_pct,
        "estimated_size_mb": estimated_size_mb,
        "ingestion_time": ingestion_time,
        "device": device_name,
        "dataset_name": dataset_display_name,
        "input_shape": (
            str(observed_input_shape)
            if observed_input_shape is not None
            else "Not available"
        ),
        "batch_shape": (
            str(list(test_inputs.shape)) if test_inputs is not None else "Not available"
        ),
        "sample_count": actual_evaluated_samples,
        "gradient_access": gradient_access,
        "supports_gradients": supports_gradients,
    }

    metadata_dict = {
        "Model Name": model_name,
        "Architecture": arch_name,
        "Domain": str(resolved_domain).capitalize(),
        "Framework": str(getattr(metadata, "framework", None) or "PyTorch").upper(),
        "Dataset": dataset_display_name,
        "Task Type": str(
            getattr(metadata, "task_type", None)
            or preset_info.get("task_type")
            or "Classification"
        ).capitalize(),
        "Input Shape": telemetry["input_shape"],
        "Total Classes": str(telemetry["num_classes"]),
        "Device": device_name,
        "Gradient Access": gradient_access,
    }

    return {
        "adapter": adapter,
        "raw_model": raw_model,
        "metadata": metadata,
        "metadata_dict": metadata_dict,
        "telemetry": telemetry,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_pct": trainable_pct,
        "estimated_size_mb": estimated_size_mb,
        "model_architecture": arch_name,
        "num_classes": metadata_dict["Total Classes"],
        "input_shape": metadata_dict["Input Shape"],
        "batch_tensor_shape": str(
            list(test_inputs.shape)
            if test_inputs is not None
            else f"({sample_size}, ...)"
        ),
        "sample_size": actual_evaluated_samples,
        "device": device_name,
        "test_inputs": test_inputs,
        "test_labels": test_labels,
        "dataset_name": dataset_display_name,
        "domain": resolved_domain,
        "ingestion_time": ingestion_time,
        "gradient_access": gradient_access,
    }


def _generate_domain_sample_batch(
    metadata: Any, sample_size: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a sample input tensor conforming to the model's expected shape."""
    domain = str(getattr(metadata, "data_domain", "image")).lower().replace("_", "-")
    num_classes = getattr(metadata, "num_classes", 2) or 2

    if domain in ("image", "vision"):
        inputs = torch.rand(sample_size, 3, 224, 224)
        labels = torch.randint(0, num_classes, (sample_size,))
    elif domain in ("tabular", "csv"):
        num_features = getattr(metadata, "num_features", 14) or 14
        inputs = torch.randn(sample_size, num_features)
        labels = torch.randint(0, num_classes, (sample_size,))
    elif domain in ("time-series", "timeseries"):
        inputs = torch.randn(sample_size, 1, 14)
        labels = torch.randint(0, num_classes, (sample_size,))
    else:
        inputs = torch.rand(sample_size, 3, 224, 224)
        labels = torch.randint(0, num_classes, (sample_size,))

    return inputs, labels
