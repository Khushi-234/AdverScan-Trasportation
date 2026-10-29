"""
Tabular modality handler for structured datasets in AdverScan XAI.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from app.explainability.utils import to_numpy, to_tensor
from app.explainability.modalities.base import BaseModalityHandler
from app.explainability.modality import Modality


class TabularModalityHandler(BaseModalityHandler):
    """
    Handles structured tabular data and identifies important feature columns.
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        super().__init__(modality=Modality.TABULAR)
        self.feature_names = feature_names

    def prepare_input(self, inputs: Any) -> torch.Tensor:
        """
        Validate and format tabular input into standard 2D (B, D) PyTorch tensor.
        """
        # Save feature names from DataFrame if available
        if hasattr(inputs, "columns"):
            self.feature_names = list(inputs.columns)

        t = to_tensor(inputs)
        if t.ndim == 1:
            t = t.unsqueeze(0)  # (1, D)
        elif t.ndim > 2:
            t = t.flatten(start_dim=1)
        return t

    def prepare_batch(self, clean: Any, adversarial: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare and validate clean and adversarial tabular inputs.
        """
        clean_t = self.prepare_input(clean)
        adv_t = self.prepare_input(adversarial)
        if clean_t.shape[1] != adv_t.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch between clean {clean_t.shape[1]} and adversarial {adv_t.shape[1]}"
            )
        return clean_t, adv_t

    def _get_feature_names(self, num_features: int) -> List[str]:
        """Get or generate feature names."""
        if self.feature_names and len(self.feature_names) == num_features:
            return self.feature_names
        return [f"feature_{i}" for i in range(num_features)]

    def postprocess_attribution(
        self,
        attribution: Any,
        inputs: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Format tabular attributions with column names, ranks, and contribution signs.
        """
        if attribution is None:
            return {"attribution": None, "top_features": [], "summary": {}}

        arr = to_numpy(attribution, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[0]  # Take first sample if batched single sample
        arr = arr.ravel()

        num_features = len(arr)
        feature_names = kwargs.get("feature_names") or self._get_feature_names(num_features)

        top_features = self.get_top_features(arr, feature_names, top_k=kwargs.get("top_k", 5))

        feature_importance_map = {
            name: float(score) for name, score in zip(feature_names, arr)
        }

        pos_features = [f for f in top_features if f["importance"] > 0]
        neg_features = [f for f in top_features if f["importance"] < 0]

        return {
            "attribution": arr.tolist(),
            "feature_names": feature_names,
            "feature_importance_map": feature_importance_map,
            "top_features": top_features,
            "summary": {
                "num_features": num_features,
                "top_positive_feature": pos_features[0]["feature"] if pos_features else None,
                "top_negative_feature": neg_features[0]["feature"] if neg_features else None,
                "mean_abs_importance": float(np.mean(np.abs(arr))),
            },
        }

    def get_top_features(
        self,
        attribution: Any,
        inputs: Any,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Rank tabular features by absolute attribution score.
        """
        arr = to_numpy(attribution, dtype=np.float32).ravel()
        num_features = len(arr)
        feature_names = inputs if isinstance(inputs, list) and len(inputs) == num_features else self._get_feature_names(num_features)

        sorted_indices = np.argsort(np.abs(arr))[::-1][:top_k]
        top_features = []

        for rank, idx in enumerate(sorted_indices):
            top_features.append({
                "feature": feature_names[idx],
                "importance": float(arr[idx]),
                "rank": rank + 1,
                "index": int(idx),
            })

        return top_features
