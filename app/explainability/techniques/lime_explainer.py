"""
LIME-based XAI technique implementation for AdverScan.
"""

from typing import Any, Dict, Optional
import numpy as np

try:
    import lime
    from lime import lime_image
    _LIME_AVAILABLE = True
except ImportError:
    lime = None
    lime_image = None
    _LIME_AVAILABLE = False


class LIMEExplainer:
    """
    LIME-based local feature/image-region attribution explainer.
    Gracefully handles optional availability of the `lime` package.
    """

    def __init__(self, **kwargs: Any):
        """Initialize LIMEExplainer."""
        self.kwargs = kwargs
        self.is_available = _LIME_AVAILABLE

    def explain(
        self,
        model: Any,
        inputs: Any,
        target_class: Optional[int] = None,
        num_samples: int = 100,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate LIME local image region attribution for model predictions on inputs.

        Args:
            model: PyTorch model or prediction function.
            inputs: Input tensor or numpy array (batch or single image).
            target_class: Target class index to explain (optional).
            num_samples: Number of samples for LIME perturbation.

        Returns:
            Structured attribution dictionary indicating execution status and result.
        """
        if not self.is_available:
            return {
                "status": "unavailable",
                "executed": False,
                "technique": "lime",
                "message": "LIME library is not installed in the current environment.",
                "attribution": None,
            }

        try:
            import torch

            if isinstance(inputs, torch.Tensor):
                inputs_np = inputs.detach().cpu().numpy()
            else:
                inputs_np = np.asarray(inputs)

            # LIME image explainer expects single image (H, W, C) or (C, H, W)
            single_img = inputs_np[0] if inputs_np.ndim == 4 else inputs_np
            if single_img.ndim == 3 and single_img.shape[0] in (1, 3, 4):  # (C, H, W) -> (H, W, C)
                single_img = np.transpose(single_img, (1, 2, 0))

            def predict_fn(images: np.ndarray) -> np.ndarray:
                # images shape: (B, H, W, C) -> transform back to (B, C, H, W) if PyTorch model
                if images.ndim == 4 and images.shape[-1] in (1, 3, 4):
                    images_t = np.transpose(images, (0, 3, 1, 2))
                else:
                    images_t = images

                t_input = torch.tensor(images_t, dtype=torch.float32)
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

            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                single_img,
                predict_fn,
                top_labels=5,
                hide_color=0,
                num_samples=num_samples,
                **kwargs,
            )

            top_label = target_class if target_class is not None else explanation.top_labels[0]
            temp, mask = explanation.get_image_and_mask(
                top_label, positive_only=True, num_features=5, hide_rest=False
            )

            return {
                "status": "success",
                "executed": True,
                "technique": "lime",
                "message": "LIME explanation computed successfully.",
                "target_class": top_label,
                "attribution": mask.tolist() if isinstance(mask, np.ndarray) else mask,
            }
        except Exception as e:
            return {
                "status": "error",
                "executed": False,
                "technique": "lime",
                "message": f"LIME computation error: {str(e)}",
                "attribution": None,
            }
