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

from app.explainability.utils import predict_proba_np, to_numpy, to_tensor, unwrap_model
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
            batch_size = int(kwargs.get("batch_size", 64))
            max_evals = int(kwargs.get("shap_max_evals", kwargs.get("max_evals", 100)))
            max_text_length = kwargs.get("max_text_length", 128)
            tokenizer = kwargs.get("tokenizer", getattr(model, "tokenizer", None))

            # 1. Text input handling
            if (modality == Modality.TEXT) or (isinstance(inputs, (list, tuple, str)) and (isinstance(inputs, str) or (inputs and isinstance(inputs[0], str)))):
                text_list = [inputs] if isinstance(inputs, str) else list(inputs)

                def text_predict_fn(texts: List[str]) -> np.ndarray:
                    return predict_proba_np(
                        model,
                        texts,
                        output_mode=output_mode,
                        batch_size=batch_size,
                        max_text_length=max_text_length,
                        tokenizer=tokenizer,
                    )

                explainer = shap.Explainer(text_predict_fn)
                try:
                    shap_values = explainer(text_list, max_evals=max_evals)
                except Exception:
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
                return predict_proba_np(model, x, output_mode=output_mode, batch_size=batch_size)

            # Prepare background dataset respecting baseline_strategy
            baseline_strategy = kwargs.get("baseline_strategy", "mean")
            if background_inputs is not None:
                bg_np = to_numpy(background_inputs)
                if len(bg_np) > 100:
                    bg_np = bg_np[:100]
                if baseline_strategy == "mean" and len(bg_np) > 1:
                    bg_np = np.mean(bg_np, axis=0, keepdims=True)
            else:
                bg_np = np.zeros_like(inputs_np)

            # For 3D or 4D data (timeseries or images), handle flattening for black-box or GradientExplainer for torch
            is_multidim = inputs_np.ndim > 2
            unwrapped = unwrap_model(model)
            use_grad_explainer = (
                is_multidim
                and hasattr(shap, "GradientExplainer")
                and hasattr(unwrapped, "parameters")
                and callable(getattr(unwrapped, "parameters", None))
            )

            values = None
            orig_shape = None

            if use_grad_explainer:
                try:
                    import torch
                    import torch.nn as nn
                    if isinstance(unwrapped, nn.Module):
                        device = next(unwrapped.parameters()).device if list(unwrapped.parameters()) else torch.device("cpu")
                        bg_tensor = to_tensor(bg_np, device=device)
                        inp_tensor = to_tensor(inputs_np, device=device)
                        explainer = shap.GradientExplainer(unwrapped, bg_tensor)
                        shap_vals = explainer.shap_values(inp_tensor)

                        if isinstance(shap_vals, list):
                            orig_shape = [tuple(v.shape) for v in shap_vals]
                            idx = target_class if (target_class is not None and 0 <= target_class < len(shap_vals)) else 0
                            values = shap_vals[idx]
                        else:
                            values = to_numpy(shap_vals)
                            orig_shape = tuple(values.shape)
                            if values.ndim == inputs_np.ndim + 1 and target_class is not None:
                                c_idx = target_class if target_class < values.shape[-1] else 0
                                values = values[..., c_idx]
                except Exception:
                    values = None

            if values is None:
                if is_multidim:
                    # Flatten to 2D with a reshaping predict_fn
                    flat_inputs = inputs_np.reshape(inputs_np.shape[0], -1)
                    flat_bg = bg_np.reshape(bg_np.shape[0], -1)
                    input_shape = inputs_np.shape

                    def flat_predict_fn(x_flat: np.ndarray) -> np.ndarray:
                        x_reshaped = x_flat.reshape((-1,) + input_shape[1:])
                        return predict_fn(x_reshaped)

                    explainer = shap.Explainer(flat_predict_fn, flat_bg)
                    try:
                        shap_values = explainer(flat_inputs, max_evals=max_evals)
                    except Exception:
                        shap_values = explainer(flat_inputs)
                    raw_vals = shap_values.values if hasattr(shap_values, "values") else np.asarray(shap_values)

                    # Reshape back to input dimensions
                    if isinstance(raw_vals, np.ndarray):
                        if raw_vals.ndim == 2:  # (N, D_flat)
                            values = raw_vals.reshape(input_shape)
                            orig_shape = tuple(values.shape)
                        elif raw_vals.ndim == 3:  # (N, D_flat, C)
                            c_dim = raw_vals.shape[-1]
                            reshaped = raw_vals.reshape(input_shape + (c_dim,))
                            orig_shape = tuple(reshaped.shape)
                            if target_class is not None:
                                c_idx = target_class if target_class < c_dim else 0
                                values = reshaped[..., c_idx]
                            else:
                                values = reshaped[..., 0]
                        else:
                            values = raw_vals
                            orig_shape = tuple(values.shape)
                else:
                    # Tabular 2D data
                    explainer = shap.Explainer(predict_fn, bg_np)
                    try:
                        shap_values = explainer(inputs_np, max_evals=max_evals)
                    except Exception:
                        shap_values = explainer(inputs_np)
                    values = shap_values.values if hasattr(shap_values, "values") else np.asarray(shap_values)

                    if isinstance(values, list):
                        orig_shape = [tuple(v.shape) for v in values]
                        idx = target_class if (target_class is not None and 0 <= target_class < len(values)) else 0
                        values = values[idx]
                    elif isinstance(values, np.ndarray):
                        orig_shape = tuple(values.shape)
                        # Slice ONLY when values.ndim == inputs_np.ndim + 1
                        if values.ndim == inputs_np.ndim + 1 and target_class is not None:
                            c_idx = target_class if target_class < values.shape[-1] else 0
                            values = values[..., c_idx]

            return {
                "status": "success",
                "executed": True,
                "technique": "shap",
                "message": "SHAP explanation computed successfully.",
                "attribution": values.tolist() if isinstance(values, np.ndarray) else values,
                "unsliced_shape": orig_shape,
            }
        except Exception as e:
            return {
                "status": "error",
                "executed": False,
                "technique": "shap",
                "message": f"SHAP computation error: {str(e)}",
                "attribution": None,
            }
