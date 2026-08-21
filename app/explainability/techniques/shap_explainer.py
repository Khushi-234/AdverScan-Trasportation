"""
SHAP-based XAI technique implementation for AdverScan.
"""

from typing import Any, Dict, Optional
import numpy as np

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    shap = None
    _SHAP_AVAILABLE = False


class SHAPExplainer:
    """
    SHAP-based feature/image attribution explainer.
    Gracefully handles optional availability of the `shap` package.
    """

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
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate SHAP attribution for model predictions on inputs.

        Args:
            model: PyTorch model or prediction function.
            inputs: Input tensor or numpy array.
            target_class: Target class index to explain (optional).
            background_inputs: Background dataset samples for SHAP background reference (optional).

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
            import torch

            # Convert inputs to numpy if tensor
            if isinstance(inputs, torch.Tensor):
                inputs_np = inputs.detach().cpu().numpy()
            else:
                inputs_np = np.asarray(inputs)

            orig_shape = inputs_np.shape
            B = orig_shape[0]

            if inputs_np.ndim > 2:
                flat_inputs = inputs_np.reshape(B, -1)
                feature_dim = flat_inputs.shape[1]

                if background_inputs is not None:
                    if isinstance(background_inputs, torch.Tensor):
                        bg_np = background_inputs.detach().cpu().numpy()
                    else:
                        bg_np = np.asarray(background_inputs)
                    flat_bg = bg_np.reshape(len(bg_np), -1)
                else:
                    flat_bg = np.zeros((1, feature_dim))

                def predict_fn(x_flat: np.ndarray) -> np.ndarray:
                    x_nd = x_flat.reshape((-1,) + orig_shape[1:])
                    t_input = torch.tensor(x_nd, dtype=torch.float32)
                    if hasattr(model, "predict"):
                        res = model.predict(t_input, return_numpy=True)
                    elif callable(model):
                        model_eval = model.eval() if hasattr(model, "eval") else model
                        with torch.no_grad():
                            out = model_eval(t_input)
                        if hasattr(out, "logits"):
                            out = out.logits
                        res = out.cpu().numpy() if isinstance(out, torch.Tensor) else np.asarray(out)
                    else:
                        raise TypeError("Model must be callable or possess a predict method.")
                    if hasattr(res, "logits"):
                        res = res.logits
                    return res

                explainer = shap.KernelExplainer(predict_fn, flat_bg)
                nsamples = kwargs.get("nsamples", 10)
                shap_values = explainer.shap_values(flat_inputs, nsamples=nsamples)
            else:
                if background_inputs is not None:
                    if isinstance(background_inputs, torch.Tensor):
                        bg_np = background_inputs.detach().cpu().numpy()
                    else:
                        bg_np = np.asarray(background_inputs)
                else:
                    bg_np = np.zeros((1, inputs_np.shape[1]))

                def predict_fn(x: np.ndarray) -> np.ndarray:
                    t_input = torch.tensor(x, dtype=torch.float32)
                    if hasattr(model, "predict"):
                        res = model.predict(t_input, return_numpy=True)
                    elif callable(model):
                        model_eval = model.eval() if hasattr(model, "eval") else model
                        with torch.no_grad():
                            out = model_eval(t_input)
                        if hasattr(out, "logits"):
                            out = out.logits
                        res = out.cpu().numpy() if isinstance(out, torch.Tensor) else np.asarray(out)
                    else:
                        raise TypeError("Model must be callable or possess a predict method.")
                    if hasattr(res, "logits"):
                        res = res.logits
                    return res

                explainer = shap.KernelExplainer(predict_fn, bg_np)
                nsamples = kwargs.get("nsamples", 10)
                shap_values = explainer.shap_values(inputs_np, nsamples=nsamples)

            values = shap_values.values if hasattr(shap_values, "values") else np.asarray(shap_values)

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
