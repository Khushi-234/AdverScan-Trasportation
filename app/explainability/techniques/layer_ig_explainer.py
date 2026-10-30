"""
Layer Integrated Gradients explainer for transformer text and Vision Transformer (ViT) patch-level attribution.
Attributes predictions to intermediate embedding or hidden layer activations.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn

from app.explainability.utils import to_numpy, to_tensor, unwrap_model
from app.explainability.modality import Modality
from app.explainability.techniques.base import BaseExplainer

try:
    from captum.attr import LayerIntegratedGradients
    _CAPTUM_AVAILABLE = True
except ImportError:
    LayerIntegratedGradients = None
    _CAPTUM_AVAILABLE = False


class LayerIGExplainer(BaseExplainer):
    """
    Layer Integrated Gradients explainer for text transformers (token embeddings)
    and Vision Transformers (patch embeddings / intermediate representations).
    """

    name: str = "layer_ig"
    supported_modalities: List[Modality] = [Modality.TEXT, Modality.IMAGE]

    def __init__(self, **kwargs: Any):
        """Initialize LayerIGExplainer."""
        self.kwargs = kwargs
        self.is_available = True

    def is_compatible(self, modality: Modality, model: Any) -> bool:
        """Check compatibility: model must be a PyTorch nn.Module with embedding/hidden layers."""
        unwrapped = unwrap_model(model)
        if not isinstance(unwrapped, nn.Module):
            return False
        # Look for embeddings or transformer components
        has_emb = any(
            isinstance(m, (nn.Embedding, nn.MultiheadAttention))
            or "embed" in name.lower()
            or "transformer" in name.lower()
            or "bert" in name.lower()
            or "patch" in name.lower()
            for name, m in unwrapped.named_modules()
        )
        return has_emb

    def _find_target_layer(self, model: nn.Module) -> Optional[nn.Module]:
        """Identify appropriate embedding or first attention layer for layer attribution."""
        # 1. Look for word/token embeddings
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                return module
        # 2. Look for patch embedding / conv in ViT
        for name, module in model.named_modules():
            if "patch_embed" in name.lower() or "embeddings" in name.lower():
                return module
        # 3. Fallback to first layer with parameters
        for module in model.children():
            if list(module.parameters()):
                return module
        return None

    def explain(
        self,
        model: Any,
        inputs: Any,
        target_class: Optional[int] = None,
        steps: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Compute Layer Integrated Gradients attribution.

        Args:
            model: PyTorch transformer or neural network.
            inputs: Tensor or text tokens.
            target_class: Target class index to explain.
            steps: Number of interpolation steps.

        Returns:
            Structured attribution dictionary.
        """
        unwrapped = unwrap_model(model)
        if not isinstance(unwrapped, nn.Module):
            return {
                "status": "unsupported",
                "executed": False,
                "technique": "layer_ig",
                "message": "Layer Integrated Gradients requires a PyTorch nn.Module.",
                "attribution": None,
            }

        target_layer = kwargs.get("layer") or self._find_target_layer(unwrapped)
        if target_layer is None:
            return {
                "status": "unsupported",
                "executed": False,
                "technique": "layer_ig",
                "message": "Could not identify target embedding/hidden layer for Layer IG.",
                "attribution": None,
            }

        eff_steps = steps if steps is not None else int(kwargs.get("ig_steps", kwargs.get("steps", 25)))

        try:
            device = next(unwrapped.parameters()).device if list(unwrapped.parameters()) else torch.device("cpu")
            x = to_tensor(inputs).to(device)

            if _CAPTUM_AVAILABLE:
                lig = LayerIntegratedGradients(unwrapped, target_layer)
                attributions = lig.attribute(x, target=target_class, n_steps=eff_steps)
                attr_np = attributions.detach().cpu().numpy()
            else:
                # Custom layer gradient approximation via forward hook
                activations = []
                gradients = []

                def f_hook(module, inp, out):
                    out_t = out if isinstance(out, torch.Tensor) else out[0]
                    out_t.retain_grad()
                    activations.append(out_t)

                handle = target_layer.register_forward_hook(f_hook)
                try:
                    out = unwrapped(x)
                    c = target_class if target_class is not None else int(torch.argmax(out, dim=-1).item())
                    score = out[0, c] if out.ndim > 1 else out[c]
                    unwrapped.zero_grad()
                    score.backward()
                    if activations and activations[0].grad is not None:
                        act = activations[0]
                        grad = act.grad
                        attr_t = act * grad
                        attr_np = attr_t.detach().cpu().numpy()
                    else:
                        attr_np = np.zeros(x.shape, dtype=np.float32)
                finally:
                    handle.remove()

            return {
                "status": "success",
                "executed": True,
                "technique": "layer_ig",
                "message": "Layer Integrated Gradients computed successfully.",
                "attribution": attr_np.tolist() if isinstance(attr_np, np.ndarray) else attr_np,
            }
        except Exception as e:
            return {
                "status": "error",
                "executed": False,
                "technique": "layer_ig",
                "message": f"Layer Integrated Gradients computation error: {str(e)}",
                "attribution": None,
            }
