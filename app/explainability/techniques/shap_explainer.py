"""
SHAP-based XAI technique implementations for AdverScan.
Includes generic SHAPExplainer and specialized TreeSHAPExplainer for tree ensembles.
"""

from typing import Any, Dict, List, Optional, Union, cast
import numpy as np

shap: Any = None
try:
    import shap  # type: ignore
    import shap.maskers  # type: ignore
    _SHAP_AVAILABLE = True
except (ImportError, Exception):
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
        if not self.is_available or shap is None:
            return {
                "status": "unavailable",
                "executed": False,
                "technique": "treeshap",
                "message": "SHAP library is not installed in the current environment.",
                "attribution": None,
            }

        assert shap is not None

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

    def _eval_model(self, model: Any, x_tensor: Any) -> np.ndarray:
        """Helper to safely evaluate model and return 2D numpy array (batch_size, num_classes)."""
        import torch

        if hasattr(model, "predict") and callable(getattr(model, "predict")):
            res = model.predict(x_tensor, return_numpy=True)
        elif callable(model):
            model_eval = model.eval() if hasattr(model, "eval") else model
            with torch.no_grad():
                res = model_eval(x_tensor)
        else:
            raise TypeError("Model must be callable or possess a callable predict method.")

        if hasattr(res, "logits"):
            res = res.logits

        if isinstance(res, torch.Tensor):
            res = res.detach().cpu().numpy()

        res_np = np.asarray(res, dtype=np.float32)
        if res_np.ndim == 1:
            res_np = res_np.reshape(1, -1)
        return res_np

    def _extract_target_attribution(
        self, shap_values: Any, target_class: Optional[int] = None, batch_size: int = 1
    ) -> np.ndarray:
        """Extract numpy attribution array for the target class from SHAP output."""
        if hasattr(shap_values, "values"):
            vals = shap_values.values
        else:
            vals = shap_values

        tc_idx = target_class if target_class is not None else 0

        if isinstance(vals, list):
            tc = max(0, min(tc_idx, len(vals) - 1))
            arr = vals[tc]
        elif isinstance(vals, np.ndarray):
            if vals.ndim == 3:
                s = vals.shape
                if s[0] == batch_size:
                    # Shape (B, F, num_classes)
                    tc = max(0, min(tc_idx, s[2] - 1))
                    arr = vals[:, :, tc]
                elif s[1] == batch_size:
                    # Shape (num_classes, B, F)
                    tc = max(0, min(tc_idx, s[0] - 1))
                    arr = vals[tc, :, :]
                else:
                    tc = max(0, min(tc_idx, s[2] - 1))
                    arr = vals[:, :, tc]
            else:
                arr = vals
        else:
            arr = np.asarray(vals, dtype=np.float64)

        return np.asarray(arr, dtype=np.float64)

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
        if not self.is_available or shap is None:
            return {
                "status": "unavailable",
                "executed": False,
                "technique": "shap",
                "message": "SHAP library is not installed in the current environment.",
                "attribution": None,
            }

        assert shap is not None

        try:
            # 1. Text input handling
            if (modality == Modality.TEXT) or (isinstance(inputs, (list, tuple, str)) and (isinstance(inputs, str) or (inputs and isinstance(inputs[0], str)))):
                text_list = [inputs] if isinstance(inputs, str) else list(inputs)

                def text_predict_fn(texts: List[str]) -> np.ndarray:
                    return predict_proba_np(model, texts, output_mode=output_mode)

                tok = kwargs.get("tokenizer")
                if tok is None:
                    proc = getattr(model, "processor", None)
                    if hasattr(proc, "get_processor") and callable(proc.get_processor):
                        tok = proc.get_processor()
                    elif hasattr(proc, "_tokenizer"):
                        tok = proc._tokenizer
                    elif hasattr(proc, "tokenizer"):
                        tok = proc.tokenizer
                    elif hasattr(model, "tokenizer"):
                        tok = model.tokenizer

                masker = None
                if tok is not None and hasattr(shap, "maskers") and hasattr(shap.maskers, "Text"):
                    try:
                        masker = shap.maskers.Text(tok)
                    except Exception:
                        masker = shap.maskers.Text()
                elif hasattr(shap, "maskers") and hasattr(shap.maskers, "Text"):
                    masker = shap.maskers.Text()

                try:
                    explainer = shap.Explainer(text_predict_fn, masker=masker) if masker is not None else shap.Explainer(text_predict_fn, masker=shap.maskers.Text())
                    shap_values = explainer(text_list)
                    values = shap_values.values if hasattr(shap_values, "values") else np.asarray(shap_values)
                except Exception:
                    fallback_masker = shap.maskers.Text() if hasattr(shap, "maskers") and hasattr(shap.maskers, "Text") else None
                    explainer = shap.Explainer(text_predict_fn, masker=fallback_masker) if fallback_masker is not None else shap.Explainer(text_predict_fn)
                    shap_values = explainer(text_list)
                    values = shap_values.values if hasattr(shap_values, "values") else np.asarray(shap_values)

                if isinstance(values, np.ndarray) and target_class is not None and values.ndim == 3:
                    c_idx = target_class if target_class < values.shape[-1] else 0
                    values = values[..., c_idx]

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

            if inputs_np.ndim == 4 and "grid_size" in kwargs:
                # 4D image inputs: (B, C, H, W) with spatial grid aggregation
                img_shape: Any = inputs_np.shape
                B, C, H, W = int(img_shape[0]), int(img_shape[1]), int(img_shape[2]), int(img_shape[3])
                grid_size = kwargs.get("grid_size", 4)
                gh = min(grid_size, H)
                gw = min(grid_size, W)

                import torch
                import torch.nn.functional as F
                t_in = torch.from_numpy(inputs_np).float()
                t_pooled = F.adaptive_avg_pool2d(t_in.mean(dim=1, keepdim=True), (gh, gw))
                flat_inputs = t_pooled.view(B, -1).numpy()
                flat_bg = np.zeros_like(flat_inputs)

                def patch_predict_fn(x_flat: np.ndarray) -> np.ndarray:
                    x_t = torch.from_numpy(x_flat).float().view(-1, 1, gh, gw)
                    x_up = F.interpolate(x_t, size=(H, W), mode="nearest")
                    x_up = x_up.repeat(1, C, 1, 1)
                    return self._eval_model(model, x_up)

                nsamples = kwargs.get("nsamples", 10)
                explainer = shap.KernelExplainer(patch_predict_fn, flat_bg)
                shap_values = explainer.shap_values(flat_inputs, nsamples=nsamples)

                attr_patch = self._extract_target_attribution(shap_values, target_class, batch_size=B)
                attr_patch_t = torch.from_numpy(attr_patch).float().view(-1, 1, gh, gw)
                attr_full_t = F.interpolate(attr_patch_t, size=(H, W), mode="nearest")
                attr_full_t = attr_full_t.repeat(1, C, 1, 1)
                attr_final = attr_full_t.numpy()

                return {
                    "status": "success",
                    "executed": True,
                    "technique": "shap",
                    "message": "SHAP explanation computed successfully.",
                    "attribution": attr_final.tolist() if isinstance(attr_final, np.ndarray) else attr_final,
                    "unsliced_shape": tuple(attr_final.shape),
                }

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
                        explainer = shap.GradientExplainer(unwrapped, cast(Any, bg_tensor))
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

                    try:
                        explainer = shap.Explainer(flat_predict_fn, flat_bg)
                        shap_values = explainer(flat_inputs)
                        raw_vals = shap_values.values if hasattr(shap_values, "values") else np.asarray(shap_values)
                    except Exception:
                        nsamples = kwargs.get("nsamples", 10)
                        k_expl = shap.KernelExplainer(flat_predict_fn, flat_bg)
                        raw_vals = k_expl.shap_values(flat_inputs, nsamples=nsamples)

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
