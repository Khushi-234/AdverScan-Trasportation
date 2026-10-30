"""
TreeSHAP explanation technique implementation for tree-based models (XGBoost, Random Forest, LightGBM).
Provides tree-specific exact/approximate Shapley values for tabular inputs.
"""

from typing import Any, Dict, List, Optional
import numpy as np

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    shap = None
    _SHAP_AVAILABLE = False

from app.explainability.utils import to_numpy, unwrap_model
from app.explainability.modality import Modality
from app.explainability.techniques.base import BaseExplainer


class TreeSHAPExplainer(BaseExplainer):
    """
    Optimized TreeSHAP explainer for decision trees, Random Forests, and XGBoost/LightGBM models.
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
        return (
            hasattr(unwrapped, "estimators_")
            or hasattr(unwrapped, "get_booster")
            or hasattr(unwrapped, "tree_")
            or "Forest" in type(unwrapped).__name__
            or "Tree" in type(unwrapped).__name__
            or "Booster" in type(unwrapped).__name__
        )

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
            orig_shape = None
            if isinstance(shap_values, list):
                orig_shape = [tuple(v.shape) for v in shap_values]
                idx = target_class if (target_class is not None and 0 <= target_class < len(shap_values)) else 0
                values = shap_values[idx]
            elif isinstance(shap_values, np.ndarray):
                orig_shape = tuple(shap_values.shape)
                if shap_values.ndim == inputs_np.ndim + 1 and target_class is not None:
                    idx = target_class if (0 <= target_class < shap_values.shape[-1]) else 0
                    values = shap_values[..., idx]
                else:
                    values = shap_values
            elif hasattr(shap_values, "values"):
                v = shap_values.values
                orig_shape = tuple(v.shape)
                if v.ndim == inputs_np.ndim + 1 and target_class is not None:
                    idx = target_class if (0 <= target_class < v.shape[-1]) else 0
                    values = v[..., idx]
                else:
                    values = v
            else:
                values = np.asarray(shap_values)
                orig_shape = tuple(values.shape)

            return {
                "status": "success",
                "executed": True,
                "technique": "treeshap",
                "message": "TreeSHAP explanation computed successfully.",
                "attribution": values.tolist() if isinstance(values, np.ndarray) else values,
                "unsliced_shape": orig_shape,
            }
        except Exception as e:
            return {
                "status": "error",
                "executed": False,
                "technique": "treeshap",
                "message": f"TreeSHAP computation error: {str(e)}",
                "attribution": None,
            }
