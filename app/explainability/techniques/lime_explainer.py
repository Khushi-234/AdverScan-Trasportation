"""
LIME-based XAI technique implementation for AdverScan.
Supports image, tabular, text, and time-series inputs.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np

lime: Any = None
lime_image: Any = None
lime_tabular: Any = None
lime_text: Any = None

try:
    import lime  # type: ignore
    from lime import lime_image, lime_tabular, lime_text  # type: ignore
    _LIME_AVAILABLE = True
except ImportError:
    _LIME_AVAILABLE = False

from app.explainability.utils import predict_proba_np, to_numpy, to_tensor
from app.explainability.modality import Modality
from app.explainability.techniques.base import BaseExplainer


class LIMEExplainer(BaseExplainer):
    """
    LIME-based local attribution explainer for image, tabular, text, and time-series data.
    Gracefully handles optional availability of the `lime` package.
    """

    name: str = "lime"
    supported_modalities: List[Modality] = [
        Modality.IMAGE,
        Modality.TABULAR,
        Modality.TEXT,
        Modality.TIMESERIES,
    ]

    def __init__(self, **kwargs: Any):
        """Initialize LIMEExplainer."""
        self.kwargs = kwargs
        self.is_available = _LIME_AVAILABLE

    def explain(
        self,
        model: Any,
        inputs: Any,
        target_class: Optional[int] = None,
        num_samples: Optional[int] = None,
        modality: Optional[Union[str, Modality]] = None,
        output_mode: str = "auto",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate LIME local feature/region attribution for model predictions on inputs.

        Args:
            model: PyTorch model or prediction function.
            inputs: Input tensor, numpy array, or strings (image, tabular, or text).
            target_class: Target class index to explain (optional).
            num_samples: Number of samples for LIME perturbation (defaults to XAIConfig.lime_num_samples).
            modality: Explicit modality hint.
            output_mode: 'classification', 'regression', or 'auto'.

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
            eff_num_samples = (
                num_samples
                if num_samples is not None
                else int(kwargs.get("lime_num_samples", kwargs.get("num_samples", 200)))
            )
            norm_mod = Modality.from_str(modality) if modality else None

            # 1. Text input handling
            if norm_mod == Modality.TEXT or (isinstance(inputs, (list, tuple, str)) and (isinstance(inputs, str) or (inputs and isinstance(inputs[0], str)))):
                return self._explain_text(model, inputs, target_class=target_class, num_samples=eff_num_samples, output_mode=output_mode, **kwargs)

            inputs_np = to_numpy(inputs)

            # 2. Time-series inputs
            if norm_mod == Modality.TIMESERIES:
                return self._explain_timeseries(model, inputs_np, target_class=target_class, num_samples=eff_num_samples, output_mode=output_mode, **kwargs)

            # 3. Tabular inputs
            if norm_mod == Modality.TABULAR or inputs_np.ndim <= 2:
                return self._explain_tabular(model, inputs_np, target_class=target_class, num_samples=eff_num_samples, output_mode=output_mode, **kwargs)

            # 4. Image inputs
            return self._explain_image(model, inputs_np, target_class=target_class, num_samples=eff_num_samples, output_mode=output_mode, **kwargs)

        except Exception as e:
            return {
                "status": "error",
                "executed": False,
                "technique": "lime",
                "message": f"LIME computation error: {str(e)}",
                "attribution": None,
            }

    def _explain_image(
        self,
        model: Any,
        inputs_np: np.ndarray,
        target_class: Optional[int] = None,
        num_samples: Optional[int] = None,
        output_mode: str = "auto",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Explain image input via LimeImageExplainer."""
        eff_num_samples = (
            num_samples
            if num_samples is not None
            else int(kwargs.get("lime_num_samples", kwargs.get("num_samples", 200)))
        )
        batch_size = int(kwargs.get("batch_size", 64))

        single_img = inputs_np[0] if inputs_np.ndim == 4 else inputs_np
        if single_img.ndim == 3 and single_img.shape[0] in (1, 3, 4):  # (C, H, W) -> (H, W, C)
            single_img = np.transpose(single_img, (1, 2, 0))

        def predict_fn(images: np.ndarray) -> np.ndarray:
            if images.ndim == 4 and images.shape[-1] in (1, 3, 4):
                images_t = np.ascontiguousarray(np.transpose(images, (0, 3, 1, 2)))
            else:
                images_t = np.ascontiguousarray(images)
            return predict_proba_np(model, images_t, output_mode=output_mode, batch_size=batch_size)

        if lime_image is None:
            raise RuntimeError("lime.lime_image is not available.")

        explainer = lime_image.LimeImageExplainer()
        labels_to_explain = (target_class,) if target_class is not None else None
        explanation = explainer.explain_instance(
            single_img,
            predict_fn,
            labels=labels_to_explain,
            top_labels=5 if target_class is None else None,
            hide_color=0,
            num_samples=eff_num_samples,
            batch_size=batch_size,
        )

        top_label = target_class if target_class is not None else (
            explanation.top_labels[0] if hasattr(explanation, "top_labels") and explanation.top_labels else 0
        )

        # Build continuous weight map from local_exp[top_label] and segments
        # Each pixel receives the continuous linear model weight of its superpixel segment
        segments = explanation.segments  # shape (H, W)
        continuous_map = np.zeros(segments.shape, dtype=np.float32)
        local_weights = explanation.local_exp.get(top_label, [])
        for seg_id, weight in local_weights:
            continuous_map[segments == seg_id] = float(weight)

        return {
            "status": "success",
            "executed": True,
            "technique": "lime",
            "message": "LIME explanation computed successfully.",
            "target_class": top_label,
            "attribution": continuous_map.tolist() if isinstance(continuous_map, np.ndarray) else continuous_map,
        }

    def _explain_tabular(
        self,
        model: Any,
        inputs_np: np.ndarray,
        target_class: Optional[int] = None,
        num_samples: Optional[int] = None,
        output_mode: str = "auto",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Explain tabular input via LimeTabularExplainer."""
        eff_num_samples = (
            num_samples
            if num_samples is not None
            else int(kwargs.get("lime_num_samples", kwargs.get("num_samples", 200)))
        )
        batch_size = int(kwargs.get("batch_size", 64))

        single_sample = inputs_np[0] if inputs_np.ndim == 2 else inputs_np
        num_features = len(single_sample)

        bg_data = kwargs.get("background_inputs")
        if bg_data is not None:
            training_data = to_numpy(bg_data)
            if training_data.ndim == 1:
                training_data = training_data.reshape(1, -1)
        else:
            scale = np.maximum(np.abs(single_sample) * 0.1, 0.05)
            training_data = single_sample + np.random.normal(0, scale, size=(50, num_features))

        feature_names = kwargs.get("feature_names") or [f"feature_{i}" for i in range(num_features)]

        def predict_fn(data: np.ndarray) -> np.ndarray:
            return predict_proba_np(model, data, output_mode=output_mode, batch_size=batch_size)

        if lime_tabular is None:
            raise RuntimeError("lime.lime_tabular is not available.")

        explainer = lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            mode="classification" if output_mode in ("classification", "auto") else "regression",
            verbose=False,
        )

        labels_to_explain = (target_class,) if target_class is not None else None
        allowed_exp_kwargs = {"distance_metric", "model_regressor"}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_exp_kwargs}

        exp = explainer.explain_instance(
            single_sample,
            predict_fn,
            labels=labels_to_explain,
            num_features=num_features,
            top_labels=1 if target_class is None else None,
            num_samples=eff_num_samples,
            **filtered_kwargs,
        )

        top_label = target_class if target_class is not None else (
            exp.top_labels[0] if hasattr(exp, "top_labels") and exp.top_labels else 0
        )
        feature_weights = np.zeros(num_features, dtype=np.float32)

        for feat_idx, weight in exp.local_exp.get(top_label, []):
            if feat_idx < num_features:
                feature_weights[feat_idx] = float(weight)

        return {
            "status": "success",
            "executed": True,
            "technique": "lime",
            "message": "LIME tabular explanation computed successfully.",
            "target_class": top_label,
            "attribution": feature_weights.tolist(),
        }

    def _explain_timeseries(
        self,
        model: Any,
        inputs_np: np.ndarray,
        target_class: Optional[int] = None,
        num_samples: Optional[int] = None,
        output_mode: str = "auto",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Explain time-series input by unrolling (T, F) into temporal feature table."""
        eff_num_samples = (
            num_samples
            if num_samples is not None
            else int(kwargs.get("lime_num_samples", kwargs.get("num_samples", 200)))
        )
        batch_size = int(kwargs.get("batch_size", 64))

        single_ts = inputs_np[0] if inputs_np.ndim == 3 else inputs_np
        if single_ts.ndim == 1:
            single_ts = single_ts[:, np.newaxis]

        t_steps, n_vars = single_ts.shape
        flat_input = single_ts.reshape(1, -1)
        sensor_names = kwargs.get("sensor_names") or [f"sensor_{v}" for v in range(n_vars)]
        flat_names = [f"t{t}_{sensor_names[v]}" for t in range(t_steps) for v in range(n_vars)]

        def predict_fn(flat_data: np.ndarray) -> np.ndarray:
            orig_shape = flat_data.reshape(-1, t_steps, n_vars)
            return predict_proba_np(model, orig_shape, output_mode=output_mode, batch_size=batch_size)

        bg_data = kwargs.get("background_inputs")
        if bg_data is not None:
            bg_arr = to_numpy(bg_data)
            flat_bg = bg_arr.reshape(len(bg_arr), -1)
            training_data = flat_bg[:100]
        else:
            scale = np.maximum(np.abs(flat_input[0]) * 0.1, 0.05)
            training_data = flat_input + np.random.normal(0, scale, size=(50, t_steps * n_vars))

        if lime_tabular is None:
            raise RuntimeError("lime.lime_tabular is not available.")

        explainer = lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=flat_names,
            mode="classification" if output_mode in ("classification", "auto") else "regression",
            verbose=False,
        )

        labels_to_explain = (target_class,) if target_class is not None else None
        allowed_exp_kwargs = {"distance_metric", "model_regressor"}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_exp_kwargs}

        exp = explainer.explain_instance(
            flat_input[0],
            predict_fn,
            labels=labels_to_explain,
            num_features=t_steps * n_vars,
            top_labels=1 if target_class is None else None,
            num_samples=eff_num_samples,
            **filtered_kwargs,
        )

        top_label = target_class if target_class is not None else (
            exp.top_labels[0] if hasattr(exp, "top_labels") and exp.top_labels else 0
        )
        flat_weights = np.zeros(t_steps * n_vars, dtype=np.float32)
        for feat_idx, weight in exp.local_exp.get(top_label, []):
            if feat_idx < len(flat_weights):
                flat_weights[feat_idx] = float(weight)

        weights_2d = flat_weights.reshape(t_steps, n_vars)
        return {
            "status": "success",
            "executed": True,
            "technique": "lime",
            "message": "LIME time-series explanation computed successfully.",
            "target_class": top_label,
            "attribution": weights_2d.tolist(),
        }

    def _explain_text(
        self,
        model: Any,
        inputs: Any,
        target_class: Optional[int] = None,
        num_samples: Optional[int] = None,
        output_mode: str = "auto",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Explain text input via LimeTextExplainer."""
        eff_num_samples = (
            num_samples
            if num_samples is not None
            else int(kwargs.get("lime_num_samples", kwargs.get("num_samples", 200)))
        )
        batch_size = int(kwargs.get("batch_size", 64))
        max_text_length = kwargs.get("max_text_length", 128)
        tokenizer = kwargs.get("tokenizer", getattr(model, "tokenizer", None))

        text_str = inputs[0] if isinstance(inputs, (list, tuple)) else str(inputs)

        def predict_fn(texts: List[str]) -> np.ndarray:
            return predict_proba_np(
                model,
                texts,
                output_mode=output_mode,
                batch_size=batch_size,
                max_text_length=max_text_length,
                tokenizer=tokenizer,
            )

        if lime_text is None:
            raise RuntimeError("lime.lime_text is not available.")

        explainer = lime_text.LimeTextExplainer()
        labels_to_explain = (target_class,) if target_class is not None else None
        exp = explainer.explain_instance(
            text_str,
            predict_fn,
            labels=labels_to_explain,
            num_samples=eff_num_samples,
            top_labels=1 if target_class is None else None,
        )

        top_labels = getattr(exp, "top_labels", None)
        top_label = target_class if target_class is not None else (top_labels[0] if top_labels else 0)
        words = text_str.split()
        word_weights = np.zeros(len(words), dtype=np.float32)

        try:
            word_weight_map = dict(exp.as_list(label=top_label))
        except Exception:
            available = getattr(exp, "available_labels", lambda: [])()
            alt_label = available[0] if available else None
            word_weight_map = dict(exp.as_list(label=alt_label)) if alt_label is not None else {}
        for i, word in enumerate(words):
            if word in word_weight_map:
                word_weights[i] = word_weight_map[word]

        return {
            "status": "success",
            "executed": True,
            "technique": "lime",
            "message": "LIME text explanation computed successfully.",
            "target_class": top_label,
            "attribution": word_weights.tolist(),
        }
