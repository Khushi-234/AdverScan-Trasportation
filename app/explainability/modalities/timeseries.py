"""
Time-series modality handler for sequential and temporal models in AdverScan XAI.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from app.explainability.input_utils import to_numpy, to_tensor
from app.explainability.modalities.base import BaseModalityHandler
from app.explainability.modality import Modality


class TimeSeriesModalityHandler(BaseModalityHandler):
    """
    Handles sequential time-series inputs and identifies important time steps,
    temporal windows, and variables.
    """

    def __init__(self, variable_names: Optional[List[str]] = None):
        super().__init__(modality=Modality.TIMESERIES)
        self.variable_names = variable_names

    def prepare_input(self, inputs: Any) -> torch.Tensor:
        """
        Validate and format time-series input into standard 3D (B, T, F) PyTorch tensor.
        """
        t = to_tensor(inputs)

        # 2D: (T, F) -> add batch dimension (1, T, F)
        if t.ndim == 2:
            t = t.unsqueeze(0)

        # 1D: single univariate sequence (T,) -> (1, T, 1)
        elif t.ndim == 1:
            t = t.unsqueeze(0).unsqueeze(-1)

        # 3D: verify whether shape is (B, F, T) or (B, T, F)
        elif t.ndim == 3:
            # If features > time steps and features <= 32 while time steps > 32, could be (B, F, T)
            # Standard is (B, T, F)
            pass

        return t

    def prepare_batch(self, clean: Any, adversarial: Any, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare and validate clean and adversarial time-series inputs.
        """
        clean_t = self.prepare_input(clean)
        adv_t = self.prepare_input(adversarial)
        if clean_t.shape != adv_t.shape:
            raise ValueError(
                f"Time-series shape mismatch between clean {tuple(clean_t.shape)} and adversarial {tuple(adv_t.shape)}"
            )
        return clean_t, adv_t

    def postprocess_attribution(
        self,
        attribution: Any,
        inputs: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Standardize raw time-series attribution into time-step and variable profiles.
        """
        if attribution is None:
            return {"attribution": None, "top_features": [], "summary": {}}

        arr = to_numpy(attribution, dtype=np.float32)

        # Squeeze batch dimension if single sequence
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]

        # arr shape: (T, F) or 1D (T,)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]

        t_steps, n_vars = arr.shape

        # Step-wise importance: aggregate across variables
        step_importance = np.mean(np.abs(arr), axis=1)

        # Variable-wise importance: aggregate across time
        var_importance = np.mean(np.abs(arr), axis=0)

        var_names = self.variable_names or [f"variable_{i}" for i in range(n_vars)]

        top_features = self.get_top_features(step_importance, inputs, top_k=kwargs.get("top_k", 5))

        # Peak impact time window
        window_size = min(kwargs.get("window_size", 5), t_steps)
        if window_size > 1 and t_steps >= window_size:
            kernel = np.ones(window_size) / window_size
            rolling_avg = np.convolve(step_importance, kernel, mode="valid")
            peak_window_start = int(np.argmax(rolling_avg))
            peak_window = [peak_window_start, peak_window_start + window_size - 1]
        else:
            peak_window = [0, t_steps - 1]

        return {
            "attribution": arr.tolist(),
            "step_importance": step_importance.tolist(),
            "variable_importance": {
                name: float(score) for name, score in zip(var_names, var_importance)
            },
            "top_features": top_features,
            "peak_window": peak_window,
            "summary": {
                "num_time_steps": t_steps,
                "num_variables": n_vars,
                "peak_time_step": int(np.argmax(step_importance)),
                "peak_window": peak_window,
                "mean_temporal_importance": float(np.mean(step_importance)),
            },
        }

    def get_top_features(
        self,
        attribution: Any,
        inputs: Any,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Extract the most critical time steps by attribution magnitude.
        """
        arr = to_numpy(attribution, dtype=np.float32)
        if arr.ndim > 1:
            arr = np.mean(np.abs(arr), axis=1 if arr.shape[-1] <= arr.shape[0] else 0)

        sorted_indices = np.argsort(arr)[::-1][:top_k]
        top_features = []

        for rank, step in enumerate(sorted_indices):
            top_features.append({
                "feature": f"timestep_{step}",
                "importance": float(arr[step]),
                "rank": rank + 1,
                "time_step": int(step),
            })

        return top_features
