"""
Model validation module for testing model health and compatibility.
"""

from typing import Any, Dict, Optional, Union
import numpy as np
import torch

from app.ingestion.adapters.base_adapter import BaseModelAdapter
from app.ingestion.exceptions import (
    ModelValidationError,
    UnsupportedDomainError,
    UnsupportedModelError,
    UnsupportedTaskError,
)


class ModelValidator:
    """
    Validates model state, prediction capabilities, basic integrity,
    and runtime configuration before loading.
    """

    SUPPORTED_FRAMEWORKS = {"huggingface", "pytorch", "pytorch_local", "sklearn"}
    SUPPORTED_DOMAINS = {"image", "text", "time-series", "tabular"}

    @classmethod
    def validate_configuration(
        cls,
        config: Union[Dict[str, Any], Any],
    ) -> bool:
        """
        Validate runtime configuration prior to model ingestion.

        Args:
            config: Dictionary or ModelMetadata containing model configuration.

        Returns:
            bool: True if configuration is valid.

        Raises:
            ModelValidationError: If required fields are missing or invalid.
            UnsupportedDomainError: If domain is not image, text, time-series, or tabular.
            UnsupportedModelError: If framework is unsupported.
            UnsupportedTaskError: If task is unsupported (e.g. forecasting).
        """
        if config is None:
            raise ModelValidationError("Model configuration cannot be None.")

        # Extract configuration attributes
        if isinstance(config, dict):
            model_id = config.get("model_id")
            framework = config.get("framework")
            data_domain = config.get("data_domain") or config.get("domain")
            task_type = config.get("task_type")
            processor_name = config.get("processor_name")
            resolution = config.get("input_resolution")
            max_length = config.get("max_length")
        else:
            model_id = getattr(config, "model_id", None)
            framework = getattr(config, "framework", None)
            data_domain = getattr(config, "data_domain", None) or getattr(config, "domain", None)
            task_type = getattr(config, "task_type", None)
            processor_name = getattr(config, "processor_name", None)
            resolution = getattr(config, "input_resolution", None)
            max_length = getattr(config, "max_length", None)

        # 1. Validate required model_id
        if model_id is None or not str(model_id).strip():
            raise ModelValidationError("Validation error: 'model_id' is required but was not provided.")

        # 2. Validate framework
        if framework is None or not str(framework).strip():
            raise ModelValidationError("Validation error: 'framework' is required but was not provided.")

        clean_framework = str(framework).strip().lower()
        if clean_framework not in cls.SUPPORTED_FRAMEWORKS:
            raise UnsupportedModelError(
                f"Unsupported framework '{framework}'. Supported frameworks are: {sorted(list(cls.SUPPORTED_FRAMEWORKS))}"
            )

        # 3. Validate data_domain
        if data_domain is None or not str(data_domain).strip():
            raise ModelValidationError("Validation error: 'data_domain' is required. Supported domains are: ['image', 'text', 'time-series', 'tabular'].")

        clean_domain = str(data_domain).strip().lower().replace("_", "-")
        if clean_domain == "timeseries":
            clean_domain = "time-series"

        if clean_domain not in cls.SUPPORTED_DOMAINS and clean_domain != "unknown":
            raise UnsupportedDomainError(
                f"Unsupported data domain '{data_domain}'. Current scope strictly supports ONLY: {sorted(list(cls.SUPPORTED_DOMAINS))}"
            )

        # 4. Validate task_type
        if task_type is None or not str(task_type).strip():
            raise ModelValidationError("Validation error: 'task_type' is required (e.g. 'classification').")

        clean_task = str(task_type).strip().lower()
        if clean_task in ("forecasting", "regression", "time-series-forecasting", "ts-forecasting", "time_series_forecasting"):
            raise UnsupportedTaskError(
                "AdverScan currently supports time-series classification models only; forecasting models are not supported."
            )

        # 5. Validate processor/model compatibility
        if processor_name is not None:
            if not isinstance(processor_name, str) or not processor_name.strip():
                raise ModelValidationError("Validation error: 'processor_name' must be a non-empty string when provided.")
            lower_proc = processor_name.lower()
            if clean_domain == "image" and any(k in lower_proc for k in ("bert", "gpt", "roberta", "t5", "tokenizer", "timeseries", "time-series", "tabular")):
                raise ModelValidationError(f"Incompatible processor '{processor_name}' specified for image domain model.")
            if clean_domain == "text" and any(k in lower_proc for k in ("vit", "resnet", "convnext", "imageprocessor", "timeseries", "time-series", "tabular")):
                raise ModelValidationError(f"Incompatible processor '{processor_name}' specified for text domain model.")
            if clean_domain == "time-series" and any(k in lower_proc for k in ("bert", "gpt", "roberta", "t5", "tokenizer", "vit", "resnet", "convnext", "imageprocessor", "tabular")):
                raise ModelValidationError(f"Incompatible processor '{processor_name}' specified for time-series domain model.")
            if clean_domain == "tabular" and any(k in lower_proc for k in ("bert", "gpt", "roberta", "t5", "tokenizer", "vit", "resnet", "convnext", "imageprocessor", "timeseries", "time-series")):
                raise ModelValidationError(f"Incompatible processor '{processor_name}' specified for tabular domain model.")

        # 6. Supported image/text configuration
        if clean_domain == "image" and resolution is not None:
            if not (isinstance(resolution, (tuple, list)) and len(resolution) in (2, 3)):
                raise ModelValidationError(
                    f"Invalid input_resolution '{resolution}' for image domain. Expected 2D or 3D tuple (e.g. (3, 224, 224))."
                )
        if clean_domain == "text" and max_length is not None:
            if not (isinstance(max_length, int) and max_length > 0):
                raise ModelValidationError(
                    f"Invalid max_length '{max_length}' for text domain. Expected positive integer."
                )

        return True


    @staticmethod
    def validate(
        adapter: BaseModelAdapter,
        sample_input: Optional[Any] = None
    ) -> bool:
        """
        Validate standard model adapter.

        Args:
            adapter: Standardized model adapter instance.
            sample_input: Optional sample input tensor or array for forward pass verification.

        Returns:
            bool: True if validation passes.

        Raises:
            ModelValidationError: If any validation condition is violated.
        """
        if not isinstance(adapter, BaseModelAdapter):
            raise ModelValidationError(
                f"Expected adapter instance of BaseModelAdapter, got {type(adapter)}"
            )

        model = adapter.get_model()
        if model is None:
            raise ModelValidationError("Adapter returns empty/None underlying model instance.")

        if sample_input is not None:
            try:
                outputs = adapter.predict(sample_input)
                if outputs is None:
                    raise ModelValidationError("Model output is None during sample inference.")

                # Check for NaN or Inf values if outputs are tensors or numpy arrays
                if isinstance(outputs, torch.Tensor):
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        raise ModelValidationError("Model output contains NaN or Inf values.")
                elif isinstance(outputs, np.ndarray):
                    if np.isnan(outputs).any() or np.isinf(outputs).any():
                        raise ModelValidationError("Model output contains NaN or Inf values.")

            except ModelValidationError:
                raise
            except Exception as e:
                raise ModelValidationError(
                    f"Sample inference failed during validation: {str(e)}"
                ) from e

        return True
