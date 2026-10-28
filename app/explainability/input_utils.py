"""
Input validation, model prediction, and conversion utilities for AdverScan explainability module.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


def unwrap_model(model: Any) -> Any:
    """
    Unwrap any adapter (e.g., PyTorchAdapter, M1 wrappers) to obtain the underlying model.

    Args:
        model: Model instance or adapter.

    Returns:
        Underlying PyTorch nn.Module, scikit-learn estimator, or callable.
    """
    curr = model
    visited = set()
    while curr is not None and id(curr) not in visited:
        visited.add(id(curr))
        if hasattr(curr, "get_model") and callable(curr.get_model):
            curr = curr.get_model()
        elif hasattr(curr, "_model") and curr._model is not None and curr._model is not curr:
            curr = curr._model
        elif hasattr(curr, "model") and curr.model is not None and curr.model is not curr:
            curr = curr.model
        else:
            break
    return curr


def to_numpy(x: Any, dtype: Optional[Any] = None) -> np.ndarray:
    """
    Safely convert inputs (Tensor, DataFrame, Series, list, scalar) to NumPy ndarray.

    Args:
        x: Input data.
        dtype: Optional target numpy data type.

    Returns:
        np.ndarray of converted data.
    """
    if isinstance(x, np.ndarray):
        arr = x
    elif isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    elif hasattr(x, "to_numpy"):  # pandas DataFrame or Series
        arr = x.to_numpy()
    elif isinstance(x, (list, tuple)):
        arr = np.asarray(x)
    elif np.isscalar(x):
        arr = np.array([x])
    else:
        try:
            arr = np.asarray(x)
        except Exception as e:
            raise TypeError(f"Cannot convert object of type {type(x)} to numpy array: {e}")

    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def to_tensor(
    x: Any,
    dtype: torch.dtype = torch.float32,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Safely convert inputs to PyTorch Tensor.

    Args:
        x: Input array, DataFrame, scalar, or tensor.
        dtype: Target torch data type.
        device: Target torch device.

    Returns:
        torch.Tensor of converted data.
    """
    if isinstance(x, torch.Tensor):
        t = x.detach().clone().to(dtype=dtype).contiguous()
    elif isinstance(x, np.ndarray):
        arr = np.ascontiguousarray(x.copy())
        t = torch.from_numpy(arr).to(dtype=dtype).contiguous()
    elif hasattr(x, "to_numpy"):  # pandas DataFrame/Series
        t = torch.from_numpy(np.ascontiguousarray(x.to_numpy().copy())).to(dtype=dtype).contiguous()
    elif isinstance(x, (list, tuple)):
        t = torch.tensor(x, dtype=dtype).contiguous()
    elif np.isscalar(x):
        t = torch.tensor([x], dtype=dtype).contiguous()
    else:
        arr = np.ascontiguousarray(to_numpy(x).copy())
        t = torch.from_numpy(arr).to(dtype=dtype).contiguous()

    if device is not None:
        t = t.to(device)
    return t


def ensure_batch(inputs: Any, modality: str = "tabular") -> Tuple[Any, bool]:
    """
    Ensure input has a batch dimension.

    Args:
        inputs: Input tensor, array, or text.
        modality: Data modality string ('image', 'text', 'tabular', 'timeseries').

    Returns:
        Tuple of (batched_inputs, was_single: bool).
    """
    modality_lower = modality.lower()

    if isinstance(inputs, str):
        return [inputs], True

    if isinstance(inputs, (list, tuple)):
        if len(inputs) > 0 and isinstance(inputs[0], str):
            # Already list of strings
            return list(inputs), False

    arr_or_tensor = inputs
    is_tensor = isinstance(arr_or_tensor, torch.Tensor)
    is_np = isinstance(arr_or_tensor, np.ndarray)

    if is_tensor or is_np:
        ndim = arr_or_tensor.ndim
        if modality_lower == "image":
            # Image: 3D (C, H, W) or (H, W, C) -> add batch dimension (1, C, H, W)
            if ndim == 3:
                return (arr_or_tensor.unsqueeze(0) if is_tensor else np.expand_dims(arr_or_tensor, 0), True)
            return arr_or_tensor, False
        elif modality_lower in ("timeseries", "time_series"):
            # Time-series: 2D (T, F) -> (1, T, F)
            if ndim == 2:
                return (arr_or_tensor.unsqueeze(0) if is_tensor else np.expand_dims(arr_or_tensor, 0), True)
            return arr_or_tensor, False
        elif modality_lower == "tabular":
            # Tabular: 1D (D,) -> (1, D)
            if ndim == 1:
                return (arr_or_tensor.unsqueeze(0) if is_tensor else np.expand_dims(arr_or_tensor, 0), True)
            return arr_or_tensor, False

    return inputs, False


def get_shape(x: Any) -> Tuple[int, ...]:
    """Get shape of tensor, array, dataframe, list, or dict."""
    if hasattr(x, "shape"):
        return tuple(x.shape)
    if isinstance(x, (list, tuple)):
        return (len(x),)
    if isinstance(x, dict) or hasattr(x, "items"):
        for k in ("input_ids", "pixel_values", "features", "inputs", "x"):
            if hasattr(x, "get"):
                val = x.get(k)
                if val is not None and hasattr(val, "shape"):
                    return tuple(val.shape)
        if hasattr(x, "values"):
            for v in x.values():
                if v is not None and hasattr(v, "shape"):
                    return tuple(v.shape)
                if isinstance(v, (list, tuple)):
                    return (len(v),)
    return ()


def validate_single_sample(inputs: Any, modality: str = "tabular") -> None:
    """
    Ensure input contains exactly one sample. Raises ValueError if batch size N > 1.

    Args:
        inputs: Model input sample or batch.
        modality: Target data modality.

    Raises:
        ValueError: If input contains more than one sample.
    """
    modality_lower = modality.lower()
    if isinstance(inputs, (list, tuple)) and all(isinstance(s, str) for s in inputs):
        if len(inputs) > 1:
            raise ValueError(
                f"explain() accepts exactly one sample. Received batch size N={len(inputs)}. "
                f"Use explain_batch() for batches."
            )
        return

    shape = get_shape(inputs)
    if not shape:
        return

    n_samples = None
    if modality_lower == "image":
        if len(shape) == 4:
            n_samples = shape[0]
    elif modality_lower in ("timeseries", "time_series"):
        if len(shape) == 3:
            n_samples = shape[0]
    elif modality_lower == "tabular":
        if len(shape) == 2:
            n_samples = shape[0]

    if n_samples is not None and n_samples > 1:
        raise ValueError(
            f"explain() accepts exactly one sample. Received batch size N={n_samples}. "
            f"Use explain_batch() for batches."
        )


def validate_matching_shapes(clean: Any, adversarial: Any) -> bool:
    """
    Validate that clean and adversarial inputs have matching shapes and formats.
    """
    if isinstance(clean, (list, tuple)) and isinstance(adversarial, (list, tuple)):
        return len(clean) == len(adversarial)
    shape_clean = get_shape(clean)
    shape_adv = get_shape(adversarial)
    if not shape_clean or not shape_adv:
        return True
    return shape_clean == shape_adv


def predict_proba_np(
    model: Any,
    x: Any,
    output_mode: str = "auto",
) -> np.ndarray:
    """
    Universal prediction helper returning calibrated probabilities or scalar metrics.

    Order of resolution:
    1. Unwrap adapter (M1 adapter -> underlying model).
    2. Check output_mode:
       - 'regression': returns continuous values of shape (N, 1).
       - 'anomaly_score': evaluates reconstruction error MSE of shape (N, 1).
       - 'classification' / 'auto':
         a) If unwrapped model has `predict_proba`, use it.
         b) If model is PyTorch nn.Module, evaluate in eval() & no_grad().
            - If output is (N, 1) or (N,), treat values in [0, 1] as p1 or apply sigmoid,
              returning np.stack([1-p1, p1], axis=-1) of shape (N, 2).
            - If output is (N, C) with C >= 2, apply softmax ONLY if not already normalized.
         c) If model has `predict`, fallback to it.

    Args:
        model: PyTorch module, PyTorchAdapter, scikit-learn model, or callable.
        x: Input array, tensor, or list of strings with shape (N, ...).
        output_mode: 'classification', 'regression', 'anomaly_score', or 'auto'.

    Returns:
        np.ndarray:
            - Shape (N, C) for classification (probabilities summing to 1).
            - Shape (N, 1) for regression or anomaly_score.
    """
    unwrapped = unwrap_model(model)

    # 1. Text input handling (str, list/tuple of str, or numpy string arrays)
    is_text = False
    text_inputs: List[str] = []
    if isinstance(x, str):
        is_text = True
        text_inputs = [x]
    elif isinstance(x, (list, tuple)) and (len(x) == 0 or isinstance(x[0], str)):
        is_text = True
        text_inputs = list(x)
    elif isinstance(x, np.ndarray) and (x.dtype.kind in ("U", "S", "O") or (x.size > 0 and isinstance(x.flat[0], str))):
        is_text = True
        text_inputs = [str(s) for s in x.flat]

    if is_text:
        target = model if (hasattr(model, "predict_proba") or hasattr(model, "predict")) else unwrapped
        if hasattr(target, "predict_proba"):
            probs = target.predict_proba(text_inputs)
            return to_numpy(probs, dtype=np.float64)
        elif hasattr(target, "predict"):
            res = target.predict(text_inputs)
            res_arr = to_numpy(res, dtype=np.float64)
            if res_arr.ndim == 1:
                return np.stack([1.0 - res_arr, res_arr], axis=-1)
            is_prob = bool(
                np.all(res_arr >= 0.0)
                and np.allclose(np.sum(res_arr, axis=-1), 1.0, atol=1e-3)
            )
            if not is_prob:
                shifted = res_arr - np.max(res_arr, axis=-1, keepdims=True)
                exp_vals = np.exp(shifted)
                res_arr = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
            return res_arr
        elif callable(target) and not isinstance(target, nn.Module):
            res = target(text_inputs)
            res_arr = to_numpy(res, dtype=np.float64)
            if res_arr.ndim == 1:
                return np.stack([1.0 - res_arr, res_arr], axis=-1)
            return res_arr
        elif isinstance(target, nn.Module):
            tok = getattr(model, "tokenizer", None) or getattr(model, "processor", None) or getattr(target, "tokenizer", None)
            if hasattr(tok, "get_processor"):
                tok = tok.get_processor()
            if tok is not None and callable(tok):
                encoded = tok(text_inputs, return_tensors="pt", padding=True, truncation=True)
                device = next(target.parameters()).device if list(target.parameters()) else torch.device("cpu")
                encoded = {k: v.to(device) for k, v in encoded.items()}
                target.eval()
                with torch.no_grad():
                    out = target(**encoded)
                if hasattr(out, "logits"):
                    out = out.logits
                out_np = to_numpy(out, dtype=np.float64)
                if out_np.ndim == 1:
                    return np.stack([1.0 - out_np, out_np], axis=-1)
                shifted = out_np - np.max(out_np, axis=-1, keepdims=True)
                exp_vals = np.exp(shifted)
                return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
            else:
                raise TypeError(
                    f"PyTorch model {type(target).__name__} received raw text strings but no tokenizer/processor was attached."
                )
        else:
            raise TypeError("Model must be callable or possess predict/predict_proba method.")

    # 1b. Dictionary / Mapping input handling (e.g. NLP token dicts, multimodal batches)
    if isinstance(x, dict) or hasattr(x, "items"):
        target = model if (hasattr(model, "predict_proba") or hasattr(model, "predict")) else unwrapped
        if hasattr(target, "predict_proba"):
            probs = target.predict_proba(x)
            return to_numpy(probs, dtype=np.float64)
        elif hasattr(target, "predict"):
            res = target.predict(x)
            res_arr = to_numpy(res, dtype=np.float64)
            if res_arr.ndim == 1:
                return np.stack([1.0 - res_arr, res_arr], axis=-1)
            is_prob = bool(
                np.all(res_arr >= 0.0)
                and np.allclose(np.sum(res_arr, axis=-1), 1.0, atol=1e-3)
            )
            if not is_prob:
                shifted = res_arr - np.max(res_arr, axis=-1, keepdims=True)
                exp_vals = np.exp(shifted)
                res_arr = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
            return res_arr
        elif isinstance(unwrapped, nn.Module):
            unwrapped.eval()
            device = next(unwrapped.parameters()).device if list(unwrapped.parameters()) else torch.device("cpu")
            t_dict = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else torch.as_tensor(v, device=device))
                for k, v in x.items()
            }
            with torch.no_grad():
                out = unwrapped(**t_dict)
            if hasattr(out, "logits"):
                out = out.logits
            out_np = to_numpy(out, dtype=np.float64)
            if out_np.ndim == 1:
                return np.stack([1.0 - out_np, out_np], axis=-1)
            shifted = out_np - np.max(out_np, axis=-1, keepdims=True)
            exp_vals = np.exp(shifted)
            return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

    x_np = to_numpy(x, dtype=np.float32)

    # 2. Anomaly score mode (Autoencoder reconstruction error)
    if output_mode == "anomaly_score":
        t_in = to_tensor(x_np)
        if isinstance(unwrapped, nn.Module):
            unwrapped.eval()
            device = next(unwrapped.parameters()).device if list(unwrapped.parameters()) else torch.device("cpu")
            t_in = t_in.to(device)
            with torch.no_grad():
                recon = unwrapped(t_in)
            recon_np = to_numpy(recon, dtype=np.float64)
        elif hasattr(unwrapped, "predict"):
            recon_np = to_numpy(unwrapped.predict(x_np), dtype=np.float64)
        elif callable(unwrapped):
            recon_np = to_numpy(unwrapped(x_np), dtype=np.float64)
        else:
            raise TypeError("Model cannot be evaluated for anomaly score reconstruction.")

        # Compute per-sample MSE across all dimensions after batch
        diff = (x_np.astype(np.float64) - recon_np) ** 2
        reduce_axes = tuple(range(1, diff.ndim)) if diff.ndim > 1 else (0,)
        mse = np.mean(diff, axis=reduce_axes, keepdims=True)
        return mse.reshape(-1, 1)

    # 3. Regression mode
    if output_mode == "regression":
        if hasattr(unwrapped, "predict") and callable(getattr(unwrapped, "predict")) and not isinstance(unwrapped, nn.Module):
            preds = unwrapped.predict(x_np)
            return to_numpy(preds, dtype=np.float64).reshape(-1, 1)
        elif isinstance(unwrapped, nn.Module):
            unwrapped.eval()
            device = next(unwrapped.parameters()).device if list(unwrapped.parameters()) else torch.device("cpu")
            t_in = to_tensor(x_np).to(device)
            with torch.no_grad():
                out = unwrapped(t_in)
            if hasattr(out, "logits"):
                out = out.logits
            return to_numpy(out, dtype=np.float64).reshape(-1, 1)
        elif callable(unwrapped):
            preds = unwrapped(x_np)
            return to_numpy(preds, dtype=np.float64).reshape(-1, 1)

    # 4. Classification mode
    # a) If model has predict_proba (sklearn, XGBoost, etc.)
    if hasattr(unwrapped, "predict_proba") and callable(getattr(unwrapped, "predict_proba")) and not isinstance(unwrapped, nn.Module):
        probs = unwrapped.predict_proba(x_np)
        probs_np = to_numpy(probs, dtype=np.float64)
        if probs_np.ndim == 1:
            probs_np = np.stack([1.0 - probs_np, probs_np], axis=-1)
        return probs_np

    # b) PyTorch nn.Module forward evaluation
    if isinstance(unwrapped, nn.Module):
        unwrapped.eval()
        device = next(unwrapped.parameters()).device if list(unwrapped.parameters()) else torch.device("cpu")
        t_in = to_tensor(x_np).to(device)
        with torch.no_grad():
            out = unwrapped(t_in)
        if hasattr(out, "logits"):
            out = out.logits
        out_np = to_numpy(out, dtype=np.float64)

        # Handle single-output binary models (N, 1) or (N,)
        if out_np.ndim == 1 or (out_np.ndim == 2 and out_np.shape[1] == 1):
            flat = out_np.ravel()
            # If all values already in [0, 1] and not raw logits
            if np.all(flat >= 0.0) and np.all(flat <= 1.0):
                p1 = flat
            else:
                # Apply sigmoid to raw logit
                p1 = 1.0 / (1.0 + np.exp(-np.clip(flat, -50.0, 50.0)))
            p0 = 1.0 - p1
            return np.column_stack([p0, p1])

        # Multi-class output (N, C)
        if out_np.ndim >= 2:
            # Check if output is already normalized probabilities
            is_prob = bool(
                np.all(out_np >= 0.0)
                and np.allclose(np.sum(out_np, axis=-1), 1.0, atol=1e-3)
            )
            if is_prob:
                return out_np

            # Apply numerically stable softmax
            shift = out_np - np.max(out_np, axis=-1, keepdims=True)
            exp_s = np.exp(np.clip(shift, -50.0, 50.0))
            return exp_s / np.sum(exp_s, axis=-1, keepdims=True)

        return out_np

    # c) Fallback predict for other models
    predict_fn: Any = getattr(unwrapped, "predict", None)
    if callable(predict_fn):
        res = predict_fn(x_np)
        res_np = to_numpy(res, dtype=np.float64)
        if res_np.ndim == 1:
            # Hard binary labels or probabilities
            if np.all((res_np == 0) | (res_np == 1)):
                # Convert hard labels to pseudo-confidence
                p1 = np.where(res_np == 1, 0.99, 0.01)
                return np.column_stack([1.0 - p1, p1])
            elif np.all(res_np >= 0.0) and np.all(res_np <= 1.0):
                return np.column_stack([1.0 - res_np, res_np])
            else:
                return res_np.reshape(-1, 1)
        elif res_np.ndim >= 2:
            if np.allclose(np.sum(res_np, axis=-1), 1.0, atol=1e-3):
                return res_np
            shift = res_np - np.max(res_np, axis=-1, keepdims=True)
            exp_s = np.exp(np.clip(shift, -50.0, 50.0))
            return exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        return res_np

    if callable(unwrapped):
        res = unwrapped(x_np)
        res_np = to_numpy(res, dtype=np.float64)
        if res_np.ndim == 1:
            p1 = 1.0 / (1.0 + np.exp(-np.clip(res_np, -50.0, 50.0)))
            return np.column_stack([1.0 - p1, p1])
        return res_np

    raise TypeError(f"Model of type {type(model)} cannot be evaluated for predictions.")


def safe_predict(
    model: Any,
    inputs: Any,
    output_mode: str = "auto",
    **kwargs: Any,
) -> torch.Tensor:
    """
    Execute model prediction across diverse model types and return torch.Tensor.

    Args:
        model: Target model or adapter.
        inputs: Model inputs of shape (N, ...).
        output_mode: 'classification', 'regression', 'anomaly_score', or 'auto'.

    Returns:
        torch.Tensor of model predictions or probabilities.
    """
    probs_np = predict_proba_np(model, inputs, output_mode=output_mode)
    return to_tensor(probs_np)


def extract_prediction_and_confidence(
    outputs: Any,
    output_mode: str = "auto",
) -> Tuple[Any, float]:
    """
    Extract predicted class/label and confidence score from model outputs.

    Args:
        outputs: Model outputs (probabilities, logits, or scalar values) of shape (N, C) or (C,).
        output_mode: 'classification', 'regression', 'anomaly_score', or 'auto'.

    Returns:
        Tuple of (predicted_class_or_value, confidence_score).
    """
    arr = to_numpy(outputs, dtype=np.float64)

    if output_mode in ("regression", "anomaly_score"):
        val = float(arr.ravel()[0]) if arr.size > 0 else 0.0
        return val, 1.0

    if arr.ndim >= 2:
        # Check if single output (N, 1)
        if arr.shape[1] == 1:
            val = float(arr[0, 0])
            pred_class = int(val >= 0.5)
            conf = float(val if pred_class == 1 else (1.0 - val))
            return pred_class, conf

        preds = np.argmax(arr, axis=-1)
        confs = np.max(arr, axis=-1)
        if len(preds) == 1:
            return int(preds[0]), float(confs[0])
        return preds.tolist(), float(np.mean(confs))

    elif arr.ndim == 1:
        if len(arr) > 1:
            pred = int(np.argmax(arr))
            conf = float(np.max(arr))
            return pred, conf
        else:
            val = float(arr[0])
            pred_class = int(val >= 0.5)
            conf = float(val if pred_class == 1 else (1.0 - val))
            return pred_class, conf

    return float(arr), 1.0
