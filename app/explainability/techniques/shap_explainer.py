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
            # Convert inputs to numpy if tensor
            import torch
            if isinstance(inputs, torch.Tensor):
                inputs_np = inputs.detach().cpu().numpy()
            else:
                inputs_np = np.asarray(inputs)

            # Build prediction function wrapper
            def predict_fn(x: np.ndarray) -> np.ndarray:
                t_input = torch.tensor(x, dtype=torch.float32)
                if hasattr(model, "predict"):
                    res = model.predict(t_input, return_numpy=True)
                elif callable(model):
                    model_eval = model.eval() if hasattr(model, "eval") else model
                    with torch.no_grad():
                        out = model_eval(t_input)
                    res = out.cpu().numpy() if isinstance(out, torch.Tensor) else np.asarray(out)
                else:
                    raise TypeError("Model must be callable or possess a predict method.")
                return res

            if background_inputs is not None:
                if isinstance(background_inputs, torch.Tensor):
                    bg_np = background_inputs.detach().cpu().numpy()
                else:
                    bg_np = np.asarray(background_inputs)
            else:
                bg_np = np.zeros_like(inputs_np)

            explainer = shap.Explainer(predict_fn, bg_np)
            shap_values = explainer(inputs_np)

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
