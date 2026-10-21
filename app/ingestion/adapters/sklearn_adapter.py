"""
Scikit-learn adapter implementation for standardizing sklearn tabular models in AdverScan.
"""

from typing import Any, Optional, Sequence, Union
import numpy as np
import torch

from app.ingestion.adapters.base_adapter import AdapterCapabilities, BaseModelAdapter


class SklearnAdapter(BaseModelAdapter):
    """
    Standardized adapter for scikit-learn models and duck-typed tabular estimators.
    Exposes non-gradient capabilities, batch inference, calibrated probabilities,
    and decision functions.
    """

    def __init__(
        self,
        model: Any,
        device: Union[str, torch.device] = "cpu",
        processor: Optional[Any] = None,
        data_domain: str = "tabular",
    ):
        """
        Initialize SklearnAdapter with an estimator.

        Args:
            model: Scikit-learn estimator instance (e.g. RandomForestClassifier, LogisticRegression).
            device: Execution device (sklearn executes on CPU; device recorded for pipeline uniformity).
            processor: Optional domain preprocessor instance (e.g. TabularProcessor).
            data_domain: Target domain ('tabular').
        """
        if not (hasattr(model, "predict") and (hasattr(model, "fit") or hasattr(model, "classes_"))):
            raise TypeError(f"Expected scikit-learn compatible estimator with predict(), got {type(model)}")

        self._model = model
        self._device = device if isinstance(device, str) else str(device)
        self.processor = processor
        self.data_domain = data_domain

    @property
    def capabilities(self) -> AdapterCapabilities:
        """
        Return operational capabilities for sklearn models.
        Sklearn estimators are non-differentiable (supports_gradients=False).
        """
        has_logits = hasattr(self._model, "decision_function") or hasattr(self._model, "predict_proba")
        return AdapterCapabilities(
            supports_gradients=False,
            supports_predict=True,
            supports_batch_predict=True,
            supports_logits=has_logits,
        )

    @property
    def device(self) -> str:
        """Get current model device (CPU for sklearn)."""
        return self._device

    @property
    def classes_(self) -> Optional[np.ndarray]:
        """Return classes_ attribute if exposed by estimator."""
        return getattr(self._model, "classes_", None)

    @property
    def n_features_in_(self) -> Optional[int]:
        """Return number of features seen during fit if exposed by estimator."""
        return getattr(self._model, "n_features_in_", None)

    def get_model(self) -> Any:
        """Return the wrapped scikit-learn estimator."""
        return self._model

    def to(self, device: Any) -> "SklearnAdapter":
        """Record target execution device."""
        self._device = str(device)
        return self

    def eval(self) -> "SklearnAdapter":
        """Sklearn models are always in evaluation/inference mode."""
        return self

    def train(self, mode: bool = True) -> "SklearnAdapter":
        """No-op for sklearn models."""
        return self

    def _prepare_numpy_input(self, inputs: Any) -> np.ndarray:
        """Convert input data into a clean 2D float numpy array for sklearn."""
        if isinstance(inputs, torch.Tensor):
            return inputs.detach().cpu().numpy()
        elif hasattr(inputs, "values") and hasattr(inputs, "columns"):
            # pandas DataFrame
            return inputs.values
        elif hasattr(inputs, "values") and not hasattr(inputs, "columns"):
            # pandas Series
            vals = inputs.values
            if vals.ndim == 1:
                return vals.reshape(1, -1)
            return vals
        elif isinstance(inputs, np.ndarray):
            if inputs.ndim == 1:
                return inputs.reshape(1, -1)
            return inputs
        elif isinstance(inputs, (list, tuple)):
            arr = np.asarray(inputs)
            if arr.ndim == 1:
                return arr.reshape(1, -1)
            return arr
        return np.asarray(inputs)

    def predict(
        self,
        inputs: Any,
        return_numpy: bool = False,
        return_classes: bool = False,
        **kwargs: Any,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Perform model prediction on inputs.
        For classification pipelines (AdverScan evaluation/attacks), returns 2D logits
        or log-probabilities matching [batch_size, num_classes] for softmax compatibility,
        or discrete class labels if return_classes=True.
        """
        x_np = self._prepare_numpy_input(inputs)

        if return_classes:
            preds = self._model.predict(x_np)
            if return_numpy:
                return preds
            return torch.from_numpy(preds)

        # Standard forward pass for AdverScan evaluator and attacks
        if hasattr(self._model, "predict_proba"):
            probs = self._model.predict_proba(x_np)
            # Convert to log-probabilities (pseudo-logits) so torch.softmax(logits) returns exact probs
            logits = np.log(np.clip(probs, 1e-12, 1.0))
            if return_numpy:
                return logits
            return torch.from_numpy(logits).float()

        elif hasattr(self._model, "decision_function"):
            dec = self._model.decision_function(x_np)
            if dec.ndim == 1:
                # Binary classification: expand to [N, 2] (-dec, dec)
                dec = np.column_stack([-dec, dec])
            if return_numpy:
                return dec
            return torch.from_numpy(dec).float()

        else:
            preds = self._model.predict(x_np)
            if return_numpy:
                return preds
            return torch.from_numpy(preds).float()

    def predict_proba(self, inputs: Any, **kwargs: Any) -> np.ndarray:
        """Return calibrated class probabilities."""
        if not hasattr(self._model, "predict_proba"):
            raise NotImplementedError(
                f"Model '{self._model.__class__.__name__}' does not support predict_proba()."
            )
        x_np = self._prepare_numpy_input(inputs)
        return self._model.predict_proba(x_np)

    def predict_classes(self, inputs: Any, **kwargs: Any) -> np.ndarray:
        """Return 1D discrete class labels."""
        x_np = self._prepare_numpy_input(inputs)
        return self._model.predict(x_np)

    def predict_batch(
        self,
        inputs: Any,
        return_numpy: bool = False,
        return_classes: bool = False,
        **kwargs: Any,
    ) -> Union[torch.Tensor, np.ndarray]:
        """Perform batch prediction."""
        return self.predict(inputs, return_numpy=return_numpy, return_classes=return_classes, **kwargs)

    def get_logits(self, inputs: Any, return_numpy: bool = False, **kwargs: Any) -> Union[torch.Tensor, np.ndarray]:
        """Return unnormalized logits or decision values."""
        return self.predict(inputs, return_numpy=return_numpy, return_classes=False, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Enable direct calling."""
        return self.predict(*args, **kwargs)
