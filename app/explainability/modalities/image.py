"""
Image modality handler for vision models in AdverScan XAI.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from app.explainability.input_utils import to_numpy, to_tensor
from app.explainability.modalities.base import BaseModalityHandler
from app.explainability.modality import Modality


class ImageModalityHandler(BaseModalityHandler):
    """
    Handles image inputs and prepares spatial explanations (pixel maps, salient regions).
    """

    def __init__(self):
        super().__init__(modality=Modality.IMAGE)

    def prepare_input(self, inputs: Any) -> torch.Tensor:
        """
        Validate and format image input into standard 4D (B, C, H, W) PyTorch tensor.
        """
        t = to_tensor(inputs)

        # 3D: (C, H, W) or (H, W, C) -> add batch dim
        if t.ndim == 3:
            # If channels last (H, W, C) where C in (1, 3, 4)
            if t.shape[-1] in (1, 3, 4) and t.shape[0] not in (1, 3, 4):
                t = t.permute(2, 0, 1)  # to (C, H, W)
            t = t.unsqueeze(0)  # to (1, C, H, W)

        elif t.ndim == 4:
            # If channels last (B, H, W, C)
            if t.shape[-1] in (1, 3, 4) and t.shape[1] not in (1, 3, 4):
                t = t.permute(0, 3, 1, 2)  # to (B, C, H, W)

        elif t.ndim == 2:
            # Grayscale single image (H, W)
            t = t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        return t

    def prepare_batch(self, clean: Any, adversarial: Any, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ensure clean and adversarial images match in shape and channel ordering.
        """
        clean_t = self.prepare_input(clean)
        adv_t = self.prepare_input(adversarial)
        if clean_t.shape != adv_t.shape:
            raise ValueError(
                f"Image shape mismatch between clean {tuple(clean_t.shape)} and adversarial {tuple(adv_t.shape)}"
            )
        return clean_t, adv_t

    def postprocess_attribution(
        self,
        attribution: Any,
        inputs: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Standardize raw image attribution into 2D saliency heatmap and spatial metrics.
        """
        if attribution is None:
            return {"attribution": None, "top_features": [], "summary": {}}

        arr = to_numpy(attribution, dtype=np.float32)

        # Squeeze batch dimension if single image
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]

        # Collapse channel dimension to create 2D heatmap
        if arr.ndim == 3:
            # Channels first (C, H, W)
            if arr.shape[0] in (1, 3, 4):
                heatmap = np.mean(np.abs(arr), axis=0)
            else:  # Channels last (H, W, C)
                heatmap = np.mean(np.abs(arr), axis=-1)
        elif arr.ndim == 2:
            heatmap = np.abs(arr)
        else:
            heatmap = np.abs(arr.squeeze())
            if heatmap.ndim != 2:
                heatmap = np.ones((8, 8), dtype=np.float32)

        # Normalize heatmap to [0, 1]
        max_val = float(np.max(heatmap)) if heatmap.size > 0 else 1.0
        normalized_heatmap = (heatmap / (max_val + 1e-12)).tolist()

        top_features = self.get_top_features(heatmap, inputs, top_k=kwargs.get("top_k", 5))

        return {
            "attribution": to_numpy(attribution).tolist(),
            "heatmap": normalized_heatmap,
            "top_features": top_features,
            "summary": {
                "max_attribution": float(np.max(arr)),
                "min_attribution": float(np.min(arr)),
                "mean_attribution": float(np.mean(arr)),
                "active_area_ratio": float(np.mean(heatmap > 0.5 * max_val)),
            },
        }

    def get_top_features(
        self,
        attribution: Any,
        inputs: Any,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Extract top most salient pixel patches / spatial coordinates.
        """
        arr = to_numpy(attribution, dtype=np.float32)
        if arr.ndim > 2:
            arr = np.mean(np.abs(arr), axis=0 if arr.shape[0] in (1, 3, 4) else -1)

        flat_indices = np.argsort(arr.ravel())[::-1][:top_k]
        h, w = arr.shape[-2:]
        top_features = []

        for rank, idx in enumerate(flat_indices):
            r = int(idx // w)
            c = int(idx % w)
            top_features.append({
                "feature": f"pixel_({r},{c})",
                "importance": float(arr[r, c]),
                "rank": rank + 1,
                "coordinates": [r, c],
            })

        return top_features
