"""
Gradient-based XAI techniques for AdverScan.
Implements Saliency (Vanilla Gradient), Integrated Gradients, and Grad-CAM.
"""

import contextlib
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.explainability.input_utils import to_numpy, to_tensor, unwrap_model
from app.explainability.modality import Modality
from app.explainability.techniques.base import BaseExplainer


@contextlib.contextmanager
def disable_cudnn():
    """Context manager to temporarily disable cuDNN during backward passes to prevent LSTM eval errors."""
    orig = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        yield
    finally:
        torch.backends.cudnn.enabled = orig


class GradientExplainer(BaseExplainer):
    """
    Gradient-based explainer supporting Saliency, Integrated Gradients, and Grad-CAM.
    """

    name: str = "gradient"
    supported_modalities: List[Modality] = [
        Modality.IMAGE,
        Modality.TABULAR,
        Modality.TIMESERIES,
    ]

    def __init__(self, method: str = "saliency", **kwargs: Any):
        """
        Initialize GradientExplainer.

        Args:
            method: Gradient method: 'saliency', 'integrated_gradients', or 'gradcam'.
        """
        self.method = method.lower().replace("-", "_")
        self.kwargs = kwargs
        self.name = self.method

    def is_compatible(self, modality: Modality, model: Any) -> bool:
        """
        Check if model is a PyTorch Module and compatible with this gradient method.
        """
        unwrapped = unwrap_model(model)
        if not isinstance(unwrapped, nn.Module) and not hasattr(unwrapped, "parameters"):
            return False
        if self.method == "gradcam":
            if modality not in (Modality.IMAGE, Modality.TIMESERIES):
                return False
            has_conv = any(isinstance(m, (nn.Conv2d, nn.Conv1d)) for m in unwrapped.modules())
            return has_conv
        return True

    def explain(
        self,
        model: Any,
        inputs: Any,
        target_class: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate gradient-based attributions.
        """
        unwrapped = unwrap_model(model)
        if self.method == "gradcam":
            return self._explain_gradcam(unwrapped, inputs, target_class=target_class, **kwargs)
        elif self.method in ("integrated_gradients", "ig"):
            return self._explain_integrated_gradients(unwrapped, inputs, target_class=target_class, **kwargs)
        else:
            return self._explain_saliency(unwrapped, inputs, target_class=target_class, **kwargs)

    def _get_target_class(self, outputs: torch.Tensor, target_class: Optional[int]) -> int:
        """Infer target class if not explicitly provided."""
        if target_class is not None:
            return int(target_class)
        if outputs.ndim >= 2:
            if outputs.shape[1] == 1:
                return 1 if outputs[0, 0].item() >= 0.0 else 0
            return int(torch.argmax(outputs[0], dim=-1).item())
        elif outputs.ndim == 1 and len(outputs) > 1:
            return int(torch.argmax(outputs, dim=0).item())
        return 0

    def _get_target_score(self, outputs: torch.Tensor, c: int) -> torch.Tensor:
        """Extract scalar score for target class c, handling single-output binary models safely."""
        if outputs.ndim == 1 and len(outputs) == 1:
            raw = outputs[0]
            return raw if c == 1 else -raw
        elif outputs.ndim == 2 and outputs.shape[1] == 1:
            raw = outputs[0, 0]
            return raw if c == 1 else -raw
        elif outputs.ndim >= 2:
            return outputs[0, c]
        elif outputs.ndim == 1:
            return outputs[c]
        return outputs

    def _explain_saliency(
        self,
        model: Any,
        inputs: Any,
        target_class: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compute vanilla saliency map (gradients w.r.t input)."""
        try:
            if not isinstance(model, nn.Module):
                return {
                    "status": "unsupported",
                    "executed": False,
                    "technique": "saliency",
                    "message": "Saliency requires a differentiable PyTorch nn.Module.",
                    "attribution": None,
                }

            device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
            x = to_tensor(inputs).to(device).clone().detach().requires_grad_(True)

            model_eval = model.eval()
            outputs = model_eval(x)
            c = self._get_target_class(outputs, target_class)
            score = self._get_target_score(outputs, c)

            model.zero_grad()
            if x.grad is not None:
                x.grad.zero_()
            with disable_cudnn():
                score.backward(retain_graph=False)

            if x.grad is None:
                return {
                    "status": "error",
                    "executed": False,
                    "technique": "saliency",
                    "message": "Gradients were not computed for inputs.",
                    "attribution": None,
                }

            saliency = x.grad.detach().cpu().numpy()
            return {
                "status": "success",
                "executed": True,
                "technique": "saliency",
                "message": "Saliency attribution computed successfully.",
                "attribution": saliency.tolist() if isinstance(saliency, np.ndarray) else saliency,
            }
        except Exception as e:
            return {
                "status": "error",
                "executed": False,
                "technique": "saliency",
                "message": f"Saliency computation error: {str(e)}",
                "attribution": None,
            }

    def _explain_integrated_gradients(
        self,
        model: Any,
        inputs: Any,
        target_class: Optional[int] = None,
        steps: int = 50,
        baseline: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compute Integrated Gradients w.r.t. a baseline (or background mean)."""
        try:
            if not isinstance(model, nn.Module):
                return {
                    "status": "unsupported",
                    "executed": False,
                    "technique": "integrated_gradients",
                    "message": "Integrated Gradients requires a differentiable PyTorch nn.Module.",
                    "attribution": None,
                }

            device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
            x = to_tensor(inputs).to(device)

            if baseline is None:
                bg_data = kwargs.get("background_inputs")
                if bg_data is not None:
                    bg_np = to_numpy(bg_data)
                    mean_bg = np.mean(bg_np, axis=0, keepdims=True)
                    b = to_tensor(mean_bg, device=device)
                    if b.ndim < x.ndim:
                        b = b.expand_as(x)
                    elif b.shape != x.shape:
                        b = torch.zeros_like(x)
                else:
                    b = torch.zeros_like(x)
            else:
                b = to_tensor(baseline).to(device)
                if b.shape != x.shape:
                    b = torch.zeros_like(x)

            model_eval = model.eval()
            with torch.no_grad():
                init_out = model_eval(x)
            c = self._get_target_class(init_out, target_class)

            # Batched path interpolation
            alphas = torch.linspace(0.0, 1.0, steps, device=device).view(-1, *([1] * (x.ndim)))
            x_steps = (b.unsqueeze(0) + alphas * (x - b).unsqueeze(0)).view(-1, *x.shape[1:] if x.ndim > 1 else x.shape)
            x_steps = x_steps.clone().detach().requires_grad_(True)

            out_batch = model_eval(x_steps)
            if out_batch.ndim == 1 or (out_batch.ndim == 2 and out_batch.shape[1] == 1):
                raw = out_batch.ravel()
                scores = raw if c == 1 else -raw
            elif out_batch.ndim >= 2:
                scores = out_batch[:, c]
            else:
                scores = out_batch

            model.zero_grad()
            total_score = scores.sum()
            with disable_cudnn():
                total_score.backward()

            if x_steps.grad is None:
                return {
                    "status": "error",
                    "executed": False,
                    "technique": "integrated_gradients",
                    "message": "Failed to compute gradients along path.",
                    "attribution": None,
                }

            grads = x_steps.grad.view(steps, *x.shape)
            avg_grads = torch.mean(grads, dim=0)
            ig = (x - b) * avg_grads
            ig_np = ig.detach().cpu().numpy()

            return {
                "status": "success",
                "executed": True,
                "technique": "integrated_gradients",
                "message": "Integrated Gradients computed successfully.",
                "attribution": ig_np.tolist() if isinstance(ig_np, np.ndarray) else ig_np,
            }
        except Exception as e:
            return {
                "status": "error",
                "executed": False,
                "technique": "integrated_gradients",
                "message": f"Integrated Gradients computation error: {str(e)}",
                "attribution": None,
            }

    def _explain_gradcam(
        self,
        model: Any,
        inputs: Any,
        target_class: Optional[int] = None,
        target_layer: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compute Grad-CAM activation heatmap for convolutional networks (Conv2d or Conv1d)."""
        try:
            if not isinstance(model, nn.Module):
                return {
                    "status": "unsupported",
                    "executed": False,
                    "technique": "gradcam",
                    "message": "Grad-CAM requires a PyTorch nn.Module with convolutional layers.",
                    "attribution": None,
                }

            conv_layer = None
            for name, layer in model.named_modules():
                if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                    if target_layer is None or name == target_layer:
                        conv_layer = layer

            if conv_layer is None:
                return {
                    "status": "unsupported",
                    "executed": False,
                    "technique": "gradcam",
                    "message": "Model contains no suitable Conv2d layers (or Conv1d) for Grad-CAM.",
                    "attribution": None,
                }

            device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
            x = to_tensor(inputs).to(device).clone().detach().requires_grad_(True)
            if x.ndim == 3 and isinstance(conv_layer, nn.Conv2d):
                x = x.unsqueeze(0)
            elif x.ndim == 2 and isinstance(conv_layer, nn.Conv1d):
                x = x.unsqueeze(0)

            activations = []
            gradients = []

            def forward_hook(module, input_val, output_val):
                activations.append(output_val)
                if output_val.requires_grad:
                    output_val.register_hook(lambda grad: gradients.append(grad))

            h_fwd = conv_layer.register_forward_hook(forward_hook)

            try:
                model_eval = model.eval()
                out = model_eval(x)
                c = self._get_target_class(out, target_class)
                score = self._get_target_score(out, c)

                model.zero_grad()
                with disable_cudnn():
                    score.backward(retain_graph=True)
            finally:
                h_fwd.remove()

            if not activations or not gradients:
                return {
                    "status": "error",
                    "executed": False,
                    "technique": "gradcam",
                    "message": "Grad-CAM could not hook layer activations or gradients.",
                    "attribution": None,
                }

            act = activations[0]  # (B, C, H, W) or (B, C, L)
            grad = gradients[0]   # (B, C, H, W) or (B, C, L)

            if isinstance(conv_layer, nn.Conv1d) or act.ndim == 3:
                weights = torch.mean(grad, dim=2, keepdim=True)
                cam = torch.sum(weights * act, dim=1, keepdim=True)
                cam = F.relu(cam)
                if x.ndim == 3 and cam.shape[2] != x.shape[2]:
                    cam = F.interpolate(cam, size=x.shape[2], mode="linear", align_corners=False)
            else:
                weights = torch.mean(grad, dim=(2, 3), keepdim=True)
                cam = torch.sum(weights * act, dim=1, keepdim=True)
                cam = F.relu(cam)
                if x.ndim == 4 and cam.shape[2:] != x.shape[2:]:
                    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)

            cam_np = cam.detach().cpu().numpy()

            return {
                "status": "success",
                "executed": True,
                "technique": "gradcam",
                "message": "Grad-CAM computed successfully.",
                "attribution": cam_np.tolist() if isinstance(cam_np, np.ndarray) else cam_np,
            }
        except Exception as e:
            return {
                "status": "error",
                "executed": False,
                "technique": "gradcam",
                "message": f"Grad-CAM computation error: {str(e)}",
                "attribution": None,
            }
