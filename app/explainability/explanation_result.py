"""
ExplanationResult DTO for the explainability module in AdverScan.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast
import numpy as np


def numpy_json_serializer(obj: Any) -> Any:
    """
    JSON serializer helper for NumPy scalars, arrays, and non-serializable objects.
    """
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    return str(obj)


def _downsample_heatmap(arr: np.ndarray, max_dim: int = 16) -> List[Any]:
    """Downsample large 2D/3D heatmap arrays to keep JSON lightweight."""
    if arr.ndim > 2:
        arr = np.mean(arr, axis=0) if arr.shape[0] in (1, 3, 4) else np.mean(arr, axis=-1)
    h, w = arr.shape[:2]
    if h <= max_dim and w <= max_dim:
        return arr.tolist()

    step_h = max(1, h // max_dim)
    step_w = max(1, w // max_dim)
    downsampled = arr[::step_h, ::step_w]
    return downsampled.tolist()


@dataclass
class ExplanationResult:
    """
    Structured DTO holding standardized XAI explanation outputs across modalities,
    clean vs. adversarial comparison, prediction failure analysis, and domain metadata.
    """

    attack_name: str
    technique: str
    clean_prediction: Any
    adversarial_prediction: Any
    clean_confidence: Union[float, np.floating[Any]]
    adversarial_confidence: Union[float, np.floating[Any]]
    prediction_changed: bool
    true_label: Optional[Any] = None
    attack_caused_failure: Optional[bool] = None
    attribution: Dict[str, Any] = field(default_factory=dict)
    comparison: Dict[str, Any] = field(default_factory=dict)
    failure_analysis: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    modality: str = "image"
    domain: str = "general"
    feature_importance: Dict[str, Any] = field(default_factory=dict)
    feature_names: Optional[List[str]] = None
    perturbation_metrics: Dict[str, Any] = field(default_factory=dict)
    faithfulness: Dict[str, Any] = field(default_factory=dict)
    stability: Dict[str, Any] = field(default_factory=dict)
    validity_flags: Dict[str, Any] = field(default_factory=dict)
    attribution_status: str = "success"
    xai_decision: Dict[str, Any] = field(default_factory=dict)
    defense_hints: List[Dict[str, Any]] = field(default_factory=list)
    defense_pipeline: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ExplanationResult object to dictionary."""
        return asdict(self)

    def format_summary(self) -> str:
        """
        Produce a concise, human-readable summary of the security explanation.
        All numerical claims are directly traceable to computed result metrics.
        """
        lines = [
            f"=== AdverScan XAI Security Assessment [{self.attack_name}] ===",
            f"Technique: {self.technique.upper()} | Modality: {self.modality} | Domain: {self.domain}",
            f"Prediction Outcome: Clean={self.clean_prediction} (conf: {float(self.clean_confidence):.4f}) -> Adv={self.adversarial_prediction} (conf: {float(self.adversarial_confidence):.4f})",
            f"Prediction Flip: {self.prediction_changed} | Attack Caused Failure: {self.attack_caused_failure}",
        ]

        pred_an = self.prediction_analysis
        if pred_an and "attack_objective" in pred_an:
            lines.append(f"Attack Objective: {pred_an['attack_objective']} | Attack Success: {pred_an.get('attack_success')}")

        tf = self.attack_transformation
        if tf and tf.get("perturbation_count") is not None:
            p_cnt = tf.get("perturbation_count")
            p_ratio = tf.get("perturbation_ratio", 0.0)
            comp = tf.get("constraint_compliance")
            comp_str = f"Compliant: {comp.get('compliant')}" if isinstance(comp, dict) else "N/A"
            lines.append(f"Transformation Evidence: Perturbed elements={p_cnt} ({p_ratio*100:.2f}%) | Constraint={comp_str}")

        attr_val = self.attribution_validation
        if attr_val and "status" in attr_val:
            lines.append(f"Attribution Validation: Status={attr_val['status']} | Finite={attr_val.get('is_finite')} | Near-Zero={attr_val.get('is_near_zero')}")

        top_feats = self.feature_importance.get("adversarial_top_features") or self.feature_importance.get("clean_top_features") or []
        if top_feats:
            feat_strs = [f"{f.get('feature')} (imp: {float(f.get('importance', 0.0)):.4f})" for f in top_feats[:5]]
            lines.append("Top Salient Features: " + ", ".join(feat_strs))

        if self.failure_analysis and "attribution_interpretation" in self.failure_analysis:
            lines.append(f"Interpretation Note: {self.failure_analysis['attribution_interpretation']}")

        return "\n".join(lines)

    def save_json(
        self,
        output_path: Union[str, Path],
        store_npz: bool = True,
        max_heatmap_dim: int = 16,
    ) -> None:
        """
        Save explanation result metrics to a JSON file.

        If store_npz=True, extracts large full arrays into a companion .npz file,
        storing a downsampled heatmap and .npz file reference in JSON to prevent megabyte-sized JSON.

        Args:
            output_path: Destination JSON file path.
            store_npz: Whether to offload large arrays to .npz.
            max_heatmap_dim: Maximum dimension for JSON downsampled heatmaps.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        res_dict = self.to_dict()

        if store_npz:
            npz_arrays: Dict[str, Any] = {}
            # Check for large raw attributions
            for key in ("clean", "adversarial"):
                attr_obj = res_dict.get("attribution", {}).get(key)
                if isinstance(attr_obj, dict):
                    raw_attr = attr_obj.get("attribution")
                    if raw_attr is not None:
                        arr = np.asarray(raw_attr)
                        if arr.size > 256:
                            npz_arrays[f"{key}_attribution"] = arr
                            # In JSON, retain only downsampled heatmap
                            if "postprocessed" in attr_obj and isinstance(attr_obj["postprocessed"], dict):
                                hm = attr_obj["postprocessed"].get("heatmap")
                                if hm is not None:
                                    attr_obj["postprocessed"]["heatmap"] = _downsample_heatmap(np.asarray(hm), max_heatmap_dim)
                            attr_obj["attribution"] = f"[Stored in .npz: {key}_attribution]"

            if npz_arrays:
                npz_path = path.with_suffix(".npz")
                cast(Any, np.savez_compressed)(npz_path, **npz_arrays)
                res_dict["metadata"]["npz_arrays_path"] = str(npz_path)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(res_dict, f, indent=2, default=numpy_json_serializer)
