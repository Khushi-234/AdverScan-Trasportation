"""
Model validation module for testing model health and compatibility.
"""

from typing import Any, Optional
import numpy as np
import torch

from app.ingestion.adapters.base_adapter import BaseModelAdapter
from app.ingestion.exceptions import ModelValidationError


class ModelValidator:
    """
    Validates model state, prediction capabilities, and basic integrity.
    """

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
