"""
Hugging Face adapter implementation for standardizing Hugging Face models.
"""

from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn

from app.ingestion.adapters.base_adapter import AdapterCapabilities, BaseModelAdapter


class HuggingFaceAdapter(BaseModelAdapter):
    """
    Standardized adapter for Hugging Face transformer models across image and text domains.
    Provides uniform inference, logits extraction, device management, and attack engine compatibility.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Union[str, torch.device] = "cpu",
        processor: Optional[Any] = None,
        data_domain: Optional[str] = None,
    ):
        """
        Initialize HuggingFaceAdapter with a model and optional processor.

        Args:
            model: Hugging Face PreTrainedModel or PyTorch nn.Module.
            device: Target execution device.
            processor: Optional input processor (ImageProcessor or TextProcessor).
            data_domain: Target data domain ('image' or 'text').
        """
        if not isinstance(model, nn.Module):
            raise TypeError(f"Expected nn.Module or PreTrainedModel, got {type(model)}")

        self._model = model
        self._device = torch.device(device) if isinstance(device, str) else device
        self.processor = processor

        resolved_domain = str(data_domain).lower() if data_domain else ""
        if not resolved_domain or resolved_domain == "unknown":
            if processor is not None and hasattr(processor, "data_domain"):
                resolved_domain = str(processor.data_domain).lower()
            elif processor is not None and hasattr(processor, "domain"):
                resolved_domain = str(processor.domain).lower()
            else:
                resolved_domain = "unknown"

        self.data_domain = resolved_domain
        self._model.to(self._device)

    @property
    def capabilities(self) -> AdapterCapabilities:
        """Return Hugging Face adapter operational capabilities (supports gradients)."""
        return AdapterCapabilities(
            supports_gradients=True,
            supports_predict=True,
            supports_batch_predict=True,
            supports_logits=True,
        )

    @property
    def device(self) -> torch.device:
        """Get current model device."""
        return self._device

    def get_model(self) -> nn.Module:
        """Return the underlying Hugging Face model."""
        return self._model

    def to(self, device: Union[str, torch.device]) -> "HuggingFaceAdapter":
        """Move wrapped model and adapter to target device."""
        self._device = torch.device(device) if isinstance(device, str) else device
        self._model.to(self._device)
        return self

    def eval(self) -> "HuggingFaceAdapter":
        """Set wrapped model to evaluation mode."""
        self._model.eval()
        return self

    def train(self, mode: bool = True) -> "HuggingFaceAdapter":
        """Set wrapped model to training or evaluation mode."""
        self._model.train(mode)
        return self

    def _prepare_inputs(self, inputs: Any) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        from collections.abc import Mapping

        # Already a tensor
        if isinstance(inputs, torch.Tensor):
            return inputs.to(self._device)

        # Dictionary or Mapping of tensors (e.g. from tokenizer, BatchEncoding, or AutoImageProcessor)
        if isinstance(inputs, Mapping) or hasattr(inputs, "items"):
            return {
                k: (v.to(self._device) if isinstance(v, torch.Tensor) else v)
                for k, v in inputs.items()
            }

        # Process via attached processor if available
        if self.processor is not None:
            processed = self.processor(inputs)
            if isinstance(processed, torch.Tensor):
                return processed.to(self._device)
            elif isinstance(processed, Mapping) or hasattr(processed, "items"):
                return {
                    k: (v.to(self._device) if isinstance(v, torch.Tensor) else v)
                    for k, v in processed.items()
                }

        # NumPy array
        if isinstance(inputs, np.ndarray):
            return torch.from_numpy(inputs).to(self._device)

        # Handle Pandas DataFrame or Series
        if hasattr(inputs, "values") and isinstance(getattr(inputs, "values"), np.ndarray):
            return torch.from_numpy(inputs.values.astype(np.float32)).to(self._device)

        # Nested numeric list
        if isinstance(inputs, (list, tuple)) and inputs and isinstance(inputs[0], (int, float, list)):
            return torch.tensor(inputs).to(self._device)

        raise TypeError(f"Unsupported input type for HuggingFaceAdapter: {type(inputs)}")

    def predict(
        self,
        inputs: Any,
        return_numpy: bool = False,
        **kwargs: Any
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Perform forward inference on input data.

        Args:
            inputs: Raw data, tensor, numpy array, or dictionary of inputs.
            return_numpy: If True, returns predictions as numpy array.
            **kwargs: Extra parameters passed to model forward call.

        Returns:
            Model logits or prediction tensor / numpy array.
        """
        self._model.eval()
        prepared = self._prepare_inputs(inputs)

        with torch.no_grad():
            if isinstance(prepared, dict):
                outputs = self._model(**prepared, **kwargs)
            else:
                outputs = self._model(prepared, **kwargs)

            logits = getattr(
                outputs,
                "logits",
                getattr(
                    outputs,
                    "prediction_logits",
                    getattr(outputs, "prediction_outputs", outputs),
                ),
            )
            if hasattr(outputs, "get") and callable(getattr(outputs, "get")):
                if "prediction_outputs" in outputs and logits is outputs:
                    logits = outputs["prediction_outputs"]
                elif "logits" in outputs and logits is outputs:
                    logits = outputs["logits"]
            if hasattr(logits, "last_hidden_state"):
                logits = logits.last_hidden_state
            if isinstance(logits, torch.Tensor) and logits.ndim == 3 and self.data_domain != "time-series":
                logits = logits[:, 0, :]  # Pool CLS representation

        if return_numpy:
            return logits.cpu().numpy()
        return logits

    def predict_batch(
        self,
        inputs: Any,
        return_numpy: bool = False,
        **kwargs: Any
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Perform batch prediction on input data.

        Args:
            inputs: Batch of inputs.
            return_numpy: If True, returns predictions as numpy array.
            **kwargs: Extra parameters.

        Returns:
            Batch model prediction tensor or numpy array.
        """
        return self.predict(inputs, return_numpy=return_numpy, **kwargs)

    def get_logits(
        self,
        inputs: Any,
        return_numpy: bool = False,
        **kwargs: Any
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Compute and return raw unnormalized model logits.

        Args:
            inputs: Input data.
            return_numpy: If True, returns logits as numpy array.
            **kwargs: Extra parameters.

        Returns:
            Logits tensor or numpy array.
        """
        self._model.eval()
        prepared = self._prepare_inputs(inputs)

        with torch.no_grad():
            if isinstance(prepared, dict):
                outputs = self._model(**prepared, **kwargs)
            else:
                outputs = self._model(prepared, **kwargs)

            logits = getattr(outputs, "logits", getattr(outputs, "prediction_logits", outputs))

        if return_numpy:
            return logits.cpu().numpy()
        return logits

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Direct callable invocation of adapter, preserving gradients for attacks.
        """
        if args:
            first_arg = args[0]
            if isinstance(first_arg, torch.Tensor):
                prepared_first = first_arg.to(self._device)
                return self._model(prepared_first, *args[1:], **kwargs)
            elif isinstance(first_arg, dict):
                prepared_dict = {
                    k: (v.to(self._device) if isinstance(v, torch.Tensor) else v)
                    for k, v in first_arg.items()
                }
                return self._model(**prepared_dict, **kwargs)
        return self._model(*args, **kwargs)
