"""
Model prediction, chunked inference batching, and confidence extraction utilities for AdverScan.
Provides high-performance, batched evaluation across PyTorch, scikit-learn, XGBoost, and callables
with torch.inference_mode() and CPU thread controls to prevent OOM and reduce latency.
"""

from typing import Any, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from app.explainability.utils.conversions import to_numpy, to_tensor, unwrap_model


def configure_cpu_threads(num_threads: Optional[int] = None) -> None:
    """
    Explicitly configure PyTorch CPU thread count to avoid thread contention.

    Args:
        num_threads: Optional thread count override. If None, sets to min(4, os.cpu_count()).
    """
    if not torch.cuda.is_available() and hasattr(torch, "set_num_threads"):
        try:
            if num_threads is not None and num_threads > 0:
                torch.set_num_threads(num_threads)
            else:
                import os
                n_cores = min(4, os.cpu_count() or 1)
                torch.set_num_threads(n_cores)
        except Exception:
            pass


def predict_batched(
    model: Any,
    x: Any,
    batch_size: int = 64,
    device: Optional[Union[str, torch.device]] = None,
    max_text_length: int = 128,
    tokenizer: Optional[Any] = None,
    num_threads: Optional[int] = None,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Chunked forward prediction wrapper for models across modalities.
    Evaluates inputs in mini-batches using torch.inference_mode() to prevent OOM
    and accelerate execution for SHAP/LIME and large models.

    For text inputs, applies truncation=True, max_length=max_text_length.
    Sets model.eval() once and configures CPU threads explicitly.

    Args:
        model: Target PyTorch module, scikit-learn estimator, or callable.
        x: Input tensor, numpy array, or list of strings.
        batch_size: Sample chunk size (default: 64).
        device: PyTorch target device.
        max_text_length: Maximum token length for text truncation (default: 128).
        tokenizer: Optional tokenizer instance for NLP models.
        num_threads: Optional CPU thread count override.

    Returns:
        Concatenated model outputs:
            - torch.Tensor if model is nn.Module and x is torch.Tensor
            - np.ndarray otherwise.
    """
    unwrapped = unwrap_model(model)
    configure_cpu_threads(num_threads)

    if batch_size <= 0:
        batch_size = 64

    # 1. Text input handling (list of strings or single string)
    if isinstance(x, (list, tuple, str)) and (isinstance(x, str) or (len(x) > 0 and isinstance(x[0], str))):
        text_list = [x] if isinstance(x, str) else list(x)
        n_samples = len(text_list)
        tok = tokenizer or getattr(model, "tokenizer", None) or getattr(unwrapped, "tokenizer", None)

        if isinstance(unwrapped, nn.Module):
            unwrapped.eval()
            dev = device or (next(unwrapped.parameters()).device if list(unwrapped.parameters()) else torch.device("cpu"))
            outs: List[torch.Tensor] = []
            with torch.inference_mode():
                for i in range(0, n_samples, batch_size):
                    chunk = text_list[i : i + batch_size]
                    if tok is not None:
                        encoded = tok(
                            chunk,
                            padding=True,
                            truncation=True,
                            max_length=max_text_length,
                            return_tensors="pt",
                        )
                        encoded = {k: v.to(dev) for k, v in encoded.items()}
                        out = unwrapped(**encoded)
                    else:
                        out = unwrapped(chunk)

                    if hasattr(out, "logits"):
                        out = out.logits
                    outs.append(out)
            return torch.cat(outs, dim=0) if outs else torch.empty((0,), device=dev)

        elif hasattr(unwrapped, "predict_proba"):
            outs_np: List[np.ndarray] = []
            for i in range(0, n_samples, batch_size):
                chunk = text_list[i : i + batch_size]
                out = unwrapped.predict_proba(chunk)
                outs_np.append(to_numpy(out, dtype=np.float64))
            return np.concatenate(outs_np, axis=0) if outs_np else np.empty((0,))

        elif hasattr(unwrapped, "predict"):
            outs_np = []
            for i in range(0, n_samples, batch_size):
                chunk = text_list[i : i + batch_size]
                out = unwrapped.predict(chunk)
                outs_np.append(to_numpy(out, dtype=np.float64))
            return np.concatenate(outs_np, axis=0) if outs_np else np.empty((0,))

        elif callable(unwrapped):
            outs_np = []
            for i in range(0, n_samples, batch_size):
                chunk = text_list[i : i + batch_size]
                out = unwrapped(chunk)
                outs_np.append(to_numpy(out, dtype=np.float64))
            return np.concatenate(outs_np, axis=0) if outs_np else np.empty((0,))

    # 2. PyTorch nn.Module with Tensor input
    if isinstance(unwrapped, nn.Module) and isinstance(x, torch.Tensor):
        unwrapped.eval()
        dev = device or (next(unwrapped.parameters()).device if list(unwrapped.parameters()) else torch.device("cpu"))
        was_single = False
        tensor_x = x
        if tensor_x.ndim == 0:
            tensor_x = tensor_x.unsqueeze(0)
            was_single = True

        n_samples = tensor_x.shape[0]
        outs = []
        with torch.inference_mode():
            for i in range(0, n_samples, batch_size):
                chunk = tensor_x[i : i + batch_size].to(dev)
                out = unwrapped(chunk)
                if hasattr(out, "logits"):
                    out = out.logits
                outs.append(out)
        res = torch.cat(outs, dim=0) if outs else torch.empty((0,), device=dev)
        return res[0] if was_single and res.ndim > 1 and res.shape[0] == 1 else res

    # 3. PyTorch nn.Module with NumPy array input
    if isinstance(unwrapped, nn.Module):
        unwrapped.eval()
        dev = device or (next(unwrapped.parameters()).device if list(unwrapped.parameters()) else torch.device("cpu"))
        x_np = to_numpy(x, dtype=np.float32)
        n_samples = x_np.shape[0] if x_np.ndim > 0 else 1
        if x_np.ndim == 0:
            x_np = x_np.reshape(1)

        outs_np = []
        with torch.inference_mode():
            for i in range(0, n_samples, batch_size):
                chunk_np = x_np[i : i + batch_size]
                chunk_t = to_tensor(chunk_np, device=dev)
                out = unwrapped(chunk_t)
                if hasattr(out, "logits"):
                    out = out.logits
                outs_np.append(to_numpy(out, dtype=np.float64))
        return np.concatenate(outs_np, axis=0) if outs_np else np.empty((0,))

    # 4. Scikit-learn, XGBoost, or other non-torch models
    x_arr = to_numpy(x)
    n_samples = x_arr.shape[0] if x_arr.ndim > 0 else 1
    if x_arr.ndim == 0:
        x_arr = x_arr.reshape(1)

    if hasattr(unwrapped, "predict_proba"):
        outs_np = []
        for i in range(0, n_samples, batch_size):
            chunk = x_arr[i : i + batch_size]
            out = unwrapped.predict_proba(chunk)
            outs_np.append(to_numpy(out, dtype=np.float64))
        return np.concatenate(outs_np, axis=0) if outs_np else np.empty((0,))

    if hasattr(unwrapped, "predict"):
        outs_np = []
        for i in range(0, n_samples, batch_size):
            chunk = x_arr[i : i + batch_size]
            out = unwrapped.predict(chunk)
            outs_np.append(to_numpy(out, dtype=np.float64))
        return np.concatenate(outs_np, axis=0) if outs_np else np.empty((0,))

    if callable(unwrapped):
        outs_np = []
        for i in range(0, n_samples, batch_size):
            chunk = x_arr[i : i + batch_size]
            out = unwrapped(chunk)
            outs_np.append(to_numpy(out, dtype=np.float64))
        return np.concatenate(outs_np, axis=0) if outs_np else np.empty((0,))

    raise TypeError(f"Model of type {type(model)} cannot be evaluated by predict_batched.")


def predict_proba_np(
    model: Any,
    x: Any,
    output_mode: str = "auto",
    batch_size: int = 64,
    max_text_length: int = 128,
    tokenizer: Optional[Any] = None,
) -> np.ndarray:
    """
    Universal prediction helper returning calibrated probabilities or scalar metrics.
    Uses predict_batched to evaluate large inputs in chunks of batch_size.

    Args:
        model: PyTorch module, PyTorchAdapter, scikit-learn model, or callable.
        x: Input array, tensor, or list of strings with shape (N, ...).
        output_mode: 'classification', 'regression', 'anomaly_score', or 'auto'.
        batch_size: Batch chunk size for inference (default: 64).
        max_text_length: Maximum token length for text truncation (default: 128).
        tokenizer: Optional tokenizer for NLP models.

    Returns:
        np.ndarray:
            - Shape (N, C) for classification (calibrated probabilities summing to 1).
            - Shape (N, 1) or (N, K) for regression or anomaly_score.
    """
    unwrapped = unwrap_model(model)

    # 1. Anomaly score mode (Autoencoder reconstruction error)
    if output_mode == "anomaly_score":
        x_np = to_numpy(x, dtype=np.float32)
        raw_recon = predict_batched(
            unwrapped,
            x_np,
            batch_size=batch_size,
            max_text_length=max_text_length,
            tokenizer=tokenizer,
        )
        recon_np = to_numpy(raw_recon, dtype=np.float64)
        diff = (x_np.astype(np.float64) - recon_np) ** 2
        reduce_axes = tuple(range(1, diff.ndim)) if diff.ndim > 1 else (0,)
        mse = np.mean(diff, axis=reduce_axes, keepdims=True)
        return mse.reshape(-1, 1)

    # 2. Regression mode
    if output_mode == "regression":
        raw_preds = predict_batched(
            unwrapped,
            x,
            batch_size=batch_size,
            max_text_length=max_text_length,
            tokenizer=tokenizer,
        )
        preds_np = to_numpy(raw_preds, dtype=np.float64)
        if preds_np.ndim == 1:
            return preds_np.reshape(-1, 1)
        elif preds_np.ndim > 2:
            return preds_np.reshape(preds_np.shape[0], -1)
        return preds_np

    # 3. Classification mode (or 'auto')
    raw_out = predict_batched(
        unwrapped,
        x,
        batch_size=batch_size,
        max_text_length=max_text_length,
        tokenizer=tokenizer,
    )
    out_np = to_numpy(raw_out, dtype=np.float64)

    # a) Handle single-output binary models (N, 1) or (N,)
    if out_np.ndim == 1 or (out_np.ndim == 2 and out_np.shape[1] == 1):
        flat = out_np.ravel()
        if np.all(flat >= 0.0) and np.all(flat <= 1.0) and not (np.all(flat == 0) or np.all(flat == 1)):
            p1 = flat
        elif np.all((flat == 0) | (flat == 1)):
            p1 = np.where(flat == 1, 0.99, 0.01)
        else:
            p1 = 1.0 / (1.0 + np.exp(-np.clip(flat, -50.0, 50.0)))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    # b) Multi-class output (N, C)
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
