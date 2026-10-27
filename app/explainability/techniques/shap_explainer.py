"""
SHAP-based XAI technique implementations for AdverScan.
Includes generic SHAPExplainer and specialized TreeSHAPExplainer for tree ensembles.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    shap = None
    _SHAP_AVAILABLE = False

from app.explainability.input_utils import predict_proba_np, to_numpy, to_tensor, unwrap_model
from app.explainability.modality import Modality
from app.explainability.techniques.base import BaseExplainer


class TreeSHAPExplainer(BaseExplainer):
    """
    Optimized TreeSHAP explainer for decision trees, Random Forests, and XGBoost/LGBM models.
    """

    name: str = "treeshap"
    supported_modalities: List[Modality] = [Modality.TABULAR]

    def __init__(self, **kwargs: Any):
        """Initialize TreeSHAPExplainer."""
        self.kwargs = kwargs
        self.is_available = _SHAP_AVAILABLE

    def is_compatible(self, modality: Modality, model: Any) -> bool:
        """Check if model is a tree ensemble and modality is tabular."""
        if modality != Modality.TABULAR:
            return False
        unwrapped = unwrap_model(model)
        # Check for tree attributes
        return hasattr(unwrapped, "estimators_") or hasattr(unwrapped, "get_booster") or hasattr(unwrapped, "tree_")

    def explain(
        self,
        model: Any,
        inputs: Any,
        target_class: Optional[int] = None,
        background_inputs: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate TreeSHAP attributions for tree-based models.

        Args:
            model: Tree-based model (RandomForest, XGBoost, etc.) or adapter.
            inputs: Tabular input of shape (1, F) or (F,).
            target_class: Class index to explain.
            background_inputs: Optional background dataset.

        Returns:
            Structured attribution dictionary.
        """
        if not self.is_available:
            return {
                "status": "unavailable",
                "executed": False,
                "technique": "treeshap",
                "message": "SHAP library is not installed in the current environment.",
                "attribution": None,
            }

        try:
            unwrapped = unwrap_model(model)
            inputs_np = to_numpy(inputs)
            if inputs_np.ndim == 1:
                inputs_np = inputs_np.reshape(1, -1)

            # Build TreeExplainer
            bg_np = to_numpy(background_inputs) if background_inputs is not None else None
            explainer = shap.TreeExplainer(unwrapped, data=bg_np)
            shap_values = explainer.shap_values(inputs_np)

            # Slicing for target_class:
            # Case 1: list of arrays (one per class)
            if isinstance(shap_values, list):
                idx = target_class if (target_class is not None and 0 <= target_class < len(shap_values)) else 0
                values = shap_values[idx]
            # Case 2: 3D array (N, F, C)
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                idx = target_class if (target_class is not None and 0 <= target_class < shap_values.shape[-1]) else 0
                values = shap_values[..., idx]
            # Case 3: 2D array (N, F)
            elif isinstance(shap_values, np.ndarray):
                values = shap_values
            elif hasattr(shap_values, "values"):
                v = shap_values.values
                if v.ndim == 3:
                    idx = target_class if (target_class is not None and 0 <= target_class < v.shape[-1]) else 0
                    values = v[..., idx]
                else:
                    values = v
            else:
                values = np.asarray(shap_values)

            return {
                "status": "success",
                "executed": True,
                "technique": "treeshap",
                "message": "TreeSHAP explanation computed successfully.",
                "attribution": values.tolist() if isinstance(values, np.ndarray) else values,
            }
        except Exception as e:
            return {
                "status": "error",
                "executed": False,
                "technique": "treeshap",
                "message": f"TreeSHAP computation error: {str(e)}",
                "attribution": None,
            }


class SHAPExplainer(BaseExplainer):
    """
    SHAP-based feature/image/tabular attribution explainer.
    Gracefully handles optional availability of the `shap` package and diverse modalities.
    """

    name: str = "shap"
    supported_modalities: List[Modality] = [
        Modality.TABULAR,
        Modality.IMAGE,
        Modality.TIMESERIES,
        Modality.TEXT,
    ]

    def __init__(self, **kwargs: Any):
        """Initialize SHAPExplainer."""
        self.kwargs = kwargs
        self.is_available = _SHAP_AVAILABLE

    def explain(
        self,
        model: Any,
        inputs: Any,
        target_class: Optional[int] = None,
        background_inputs: Optional[Any] = None,
        output_mode: str = "auto",
        modality: Optional[Union[str, Modality]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate SHAP attribution for model predictions on inputs.

        Args:
            model: PyTorch model, scikit-learn estimator, or prediction function.
            inputs: Input tensor, numpy array, or strings.
            target_class: Target class index to explain (optional).
            background_inputs: Background dataset samples for SHAP baseline reference (optional).
            output_mode: 'classification', 'regression', 'anomaly_score', or 'auto'.
            modality: Explicit modality hint.

        Returns:
            Structured attribution dictionary indicating execution status and result.
        """
        if not self.is_available:
            return {
                "status": "unavailable",
                "executed": False,
                "technique": "shap",
                "message": "SHAP library is not installed in the current environment.",
                "attribution": None,
            }

        try:
            # 1. Text input handling
            if (modality == Modality.TEXT) or (isinstance(inputs, (list, tuple, str)) and (isinstance(inputs, str) or (inputs and isinstance(inputs[0], str)))):
                text_list = [inputs] if isinstance(inputs, str) else list(inputs)

                def text_predict_fn(texts: List[str]) -> np.ndarray:
                    return predict_proba_np(model, texts, output_mode=output_mode)

                explainer = shap.Explainer(text_predict_fn)
                shap_values = explainer(text_list)
                values = shap_values.values if hasattr(shap_values, "values") else np.asarray(shap_values)

                return {
                    "status": "success",
                    "executed": True,
                    "technique": "shap",
                    "message": "SHAP text explanation computed successfully.",
                    "attribution": values.tolist() if isinstance(values, np.ndarray) else values,
                }

            # 2. Tensor / array inputs (tabular, image, timeseries)
            inputs_np = to_numpy(inputs)
            if inputs_np.ndim == 1:
                inputs_np = inputs_np.reshape(1, -1)

            # Universal prediction function wrapper utilizing predict_proba_np
            def predict_fn(x: np.ndarray) -> np.ndarray:
                return predict_proba_np(model, x, output_mode=output_mode)

            if background_inputs is not None:
                bg_np = to_numpy(background_inputs)
            else:
                bg_np = np.zeros_like(inputs_np)

            # Build Explainer
            explainer = shap.Explainer(predict_fn, bg_np)
            shap_values = explainer(inputs_np)

            values = shap_values.values if hasattr(shap_values, "values") else np.asarray(shap_values)

            # Filter by target class if multidimensional outputs
            if values is not None and isinstance(values, np.ndarray):
                # When output is (N, F, C) where C is class dimension
                if values.ndim == inputs_np.ndim + 1 and target_class is not None:
                    c_idx = target_class if target_class < values.shape[-1] else 0
                    values = values[..., c_idx]
                elif values.ndim >= 3 and target_class is not None and values.shape[-1] > target_class:
                    values = values[..., target_class]

            return {
                "status": "success",
                "executed": True,
                "technique": "shap",
                "message": "SHAP explanation computed successfully.",
                "attribution": values.tolist() if isinstance(values, np.ndarray) else values,
            }
        except Exception as e:
            return {
                "status": "error",
                "executed": False,
                "technique": "shap",
                "message": f"SHAP computation error: {str(e)}",
                "attribution": None,
            }
