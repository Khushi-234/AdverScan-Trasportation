"""
SHAP-based XAI technique implementation for AdverScan.
"""

from typing import Any, Dict, Optional
import numpy as np

try:
    import shap  # type: ignore
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
            import torch.nn.functional as F

            # Convert inputs to numpy if tensor
            if isinstance(inputs, torch.Tensor):
                inputs_np = inputs.detach().cpu().numpy()
            else:
                inputs_np = np.asarray(inputs)

            orig_shape = inputs_np.shape
            B = orig_shape[0]

            # Strict upper bounds for computational safety
            user_nsamples = kwargs.get("nsamples", 8)
            nsamples = min(user_nsamples, 20)

            if inputs_np.ndim == 4:
                # 4D Image inputs: (B, C, H, W)
                _, C, H, W = orig_shape
                user_grid = kwargs.get("grid_size", 4)
                grid_size = min(max(user_grid, 2), 8)  # Default 4x4 (16 features), max 8x8 (64 features)
                gh = min(H, grid_size)
                gw = min(W, grid_size)
                num_patches = gh * gw

                flat_inputs = np.ones((B, num_patches), dtype=np.float32)
                flat_bg = np.zeros((1, num_patches), dtype=np.float32)

                t_orig = torch.from_numpy(inputs_np).float()
                # Compute channel-wise mean background reference image: (B, C, 1, 1)
                t_bg = t_orig.mean(dim=(-2, -1), keepdim=True)

                def predict_fn(x_patches: np.ndarray) -> np.ndarray:
                    N = x_patches.shape[0]
                    # Reshape patch perturbations (N, num_patches) -> (N, 1, gh, gw)
                    p_tensor = torch.from_numpy(x_patches).float().view(N, 1, gh, gw)
                    # Upsample patch mask to (N, 1, H, W)
                    mask = F.interpolate(p_tensor, size=(H, W), mode="nearest")
                    # Broadcast mask over batch items safely for any sample count N
                    repeats = (N + B - 1) // B
                    t_expanded = t_orig.repeat(repeats, 1, 1, 1)[:N]
                    t_bg_expanded = t_bg.repeat(repeats, 1, 1, 1)[:N]
                    # Blend original image with channel-wise mean background reference
                    t_in = t_expanded * mask + t_bg_expanded * (1.0 - mask)
                    return self._eval_model(model, t_in)

                explainer = shap.KernelExplainer(predict_fn, flat_bg)
                shap_values = explainer.shap_values(flat_inputs, nsamples=nsamples)

                attr_patch = self._extract_target_attribution(shap_values, target_class, batch_size=B)
                # Reshape patch attributions back to spatial dimensions (B, C, H, W)
                attr_patch_t = torch.from_numpy(attr_patch).float().view(-1, 1, gh, gw)
                attr_full_t = F.interpolate(attr_patch_t, size=(H, W), mode="nearest")
                attr_full_t = attr_full_t.repeat(1, C, 1, 1)
                attr_final = attr_full_t.numpy()

            else:
                # 1D/2D tabular or vector inputs: (B, F)
                if inputs_np.ndim > 2:
                    flat_inputs = inputs_np.reshape(B, -1)
                else:
                    flat_inputs = inputs_np

                feature_dim = flat_inputs.shape[1]

                if background_inputs is not None:
                    if isinstance(background_inputs, torch.Tensor):
                        bg_np = background_inputs.detach().cpu().numpy()
                    else:
                        bg_np = np.asarray(background_inputs)
                    flat_bg = bg_np.reshape(len(bg_np), -1)
                else:
                    if flat_inputs.shape[0] > 1:
                        flat_bg = np.mean(flat_inputs, axis=0, keepdims=True)
                    else:
                        flat_bg = np.zeros_like(flat_inputs)

                def predict_fn(x: np.ndarray) -> np.ndarray:
                    x_nd = x.reshape((-1,) + orig_shape[1:])
                    t_input = torch.tensor(x_nd, dtype=torch.float32)
                    return self._eval_model(model, t_input)

                explainer = shap.KernelExplainer(predict_fn, flat_bg)
                shap_values = explainer.shap_values(flat_inputs, nsamples=nsamples)


                attr_raw = self._extract_target_attribution(shap_values, target_class, batch_size=B)
                if attr_raw.ndim == 2:
                    attr_final = attr_raw.reshape((B,) + orig_shape[1:])
                else:
                    attr_final = attr_raw

            return {
                "status": "success",
                "executed": True,
                "technique": "shap",
                "message": "SHAP explanation computed successfully.",
                "attribution": attr_final.tolist() if isinstance(attr_final, np.ndarray) else attr_final,
            }
        except Exception as e:
            return {
                "status": "error",
                "executed": False,
                "technique": "shap",
                "message": f"SHAP computation error: {str(e)}",
                "attribution": None,
            }
