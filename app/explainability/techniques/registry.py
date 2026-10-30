"""
Technique registry for selecting and managing compatible XAI explainers in AdverScan.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import torch.nn as nn

from app.explainability.domain_profiles import get_domain_profile
from app.explainability.utils import unwrap_model
from app.explainability.modality import Modality
from app.explainability.techniques.base import BaseExplainer
from app.explainability.techniques.gradient_explainer import GradientExplainer
from app.explainability.techniques.layer_ig_explainer import LayerIGExplainer
from app.explainability.techniques.lime_explainer import LIMEExplainer
from app.explainability.techniques.shap_explainer import SHAPExplainer
from app.explainability.techniques.tree_shap_explainer import TreeSHAPExplainer


def _is_tree_model(model: Any) -> bool:
    """Check if model is a tree ensemble (scikit-learn, XGBoost, LightGBM, CatBoost)."""
    unwrapped = unwrap_model(model)
    return (
        hasattr(unwrapped, "estimators_")
        or hasattr(unwrapped, "get_booster")
        or hasattr(unwrapped, "tree_")
        or "Forest" in type(unwrapped).__name__
        or "Tree" in type(unwrapped).__name__
        or "Booster" in type(unwrapped).__name__
    )


# Rules table mapping (model_type, mode) -> technique
RULES_TABLE: Dict[str, Dict[str, str]] = {
    "tree (XGB/RF)": {
        "fast": "tree_shap",
        "standard": "tree_shap",
        "thorough": "tree_shap",
    },
    "torch MLP/LSTM": {
        "fast": "saliency",
        "standard": "integrated_gradients",
        "thorough": "gradient_shap",
    },
    "CNN": {
        "fast": "gradcam",
        "standard": "integrated_gradients",
        "thorough": "shap",
    },
    "transformer": {
        "fast": "saliency",
        "standard": "layer_integrated_gradients",
        "thorough": "shap",
    },
    "transformer/text": {
        "fast": "saliency",
        "standard": "layer_integrated_gradients",
        "thorough": "shap",
    },
}


import re


def normalize_model_type(model_type_str: str) -> str:
    """Normalize user or detected string to one of the canonical rules table model types."""
    norm = str(model_type_str).lower().strip()
    if any(k in norm for k in ("tree", "xgb", "forest", "boost", "random forest")) or re.search(r"\brf\b", norm):
        return "tree (XGB/RF)"
    if any(k in norm for k in ("transformer", "bert", "roberta", "vit", "patchtst", "distilbert", "text")):
        return "transformer"
    if any(k in norm for k in ("cnn", "conv", "resnet", "vgg", "1d-cnn")):
        return "CNN"
    if any(k in norm for k in ("mlp", "lstm", "torch", "linear", "rnn", "mixer")):
        return "torch MLP/LSTM"
    return model_type_str


def detect_model_type(model: Any, modality: Union[str, Modality]) -> str:
    """Infer model architecture type (tree (XGB/RF), torch MLP/LSTM, CNN, transformer)."""
    # 1. Preset model identifier lookup
    if isinstance(model, str):
        try:
            from app.ingestion.presets import MODEL_PRESETS
            preset = MODEL_PRESETS.get(model) or next(
                (p for p in MODEL_PRESETS.values() if p.get("model_id") == model),
                None,
            )
            if preset and preset.get("model_type"):
                return normalize_model_type(preset["model_type"])
        except Exception:
            pass
        return normalize_model_type(model)

    # 2. IngestionResult or Adapter with metadata
    if hasattr(model, "metadata"):
        meta = getattr(model, "metadata")
        if hasattr(meta, "extra_info") and isinstance(meta.extra_info, dict):
            meta_type = meta.extra_info.get("model_type")
            if meta_type:
                return normalize_model_type(meta_type)
        if getattr(meta, "model_architecture", None):
            return normalize_model_type(meta.model_architecture)

    norm_mod = Modality.from_str(modality) if modality else None
    unwrapped = unwrap_model(model) if model is not None else None

    # 3. Tree models
    if unwrapped is not None and _is_tree_model(unwrapped):
        return "tree (XGB/RF)"

    if isinstance(unwrapped, nn.Module):
        # 4. Transformer architectures (check attention/transformer first so ViT is recognized as transformer)
        has_transformer = any(
            isinstance(m, (nn.Transformer, nn.MultiheadAttention, nn.TransformerEncoder, nn.TransformerDecoder))
            or "transformer" in type(m).__name__.lower()
            or "bert" in type(m).__name__.lower()
            or "roberta" in type(m).__name__.lower()
            or "vit" in type(m).__name__.lower()
            or "attention" in type(m).__name__.lower()
            for m in unwrapped.modules()
        )
        if has_transformer or norm_mod == Modality.TEXT:
            return "transformer"

        # 5. Convolutional architectures
        has_conv = any(
            isinstance(m, (nn.modules.conv._ConvNd, nn.Conv1d, nn.Conv2d, nn.Conv3d))
            for m in unwrapped.modules()
        )
        if has_conv:
            return "CNN"

        return "torch MLP/LSTM"

    # Modality fallback when model is callable or generic
    if norm_mod == Modality.TEXT:
        return "transformer"
    if norm_mod == Modality.IMAGE:
        return "CNN"
    return "torch MLP/LSTM"


def check_override_compatibility(
    override: str,
    model_type: str,
    modality: Union[str, Modality],
    model: Optional[Any] = None,
) -> Tuple[bool, str]:
    """Verify that a manual technique override is compatible with the model type and modality."""
    norm_override = override.lower().replace("-", "_")
    norm_mod = Modality.from_str(modality) if modality else None
    canonical_model = normalize_model_type(model_type)

    # 1. Tree models cannot use gradient-based techniques
    if canonical_model == "tree (XGB/RF)":
        if norm_override in (
            "saliency",
            "integrated_gradients",
            "ig",
            "gradcam",
            "grad_cam",
            "layer_ig",
            "layer_integrated_gradients",
            "gradient_shap",
        ):
            return False, f"Unsupported XAI technique '{override}': requires neural network gradients and is incompatible with {canonical_model} models."
        if norm_override not in ("tree_shap", "treeshap", "shap", "lime"):
            return False, f"Unsupported XAI technique '{override}': incompatible with {canonical_model} models."

    # 2. GradCAM requires convolutional layers
    if norm_override in ("gradcam", "grad_cam"):
        if canonical_model != "CNN":
            return False, f"Unsupported XAI technique '{override}': requires convolutional layers and is incompatible with {canonical_model} models."
        if norm_mod is not None and norm_mod != Modality.IMAGE:
            return False, f"Unsupported XAI technique '{override}': only compatible with image/convolutional modality, got '{norm_mod.value}'."

    # 3. TreeSHAP requires tree ensemble
    if norm_override in ("tree_shap", "treeshap"):
        if canonical_model != "tree (XGB/RF)":
            return False, f"Unsupported XAI technique '{override}': requires tree-based models and is incompatible with {canonical_model} models."
        if norm_mod is not None and norm_mod != Modality.TABULAR:
            return False, f"Unsupported XAI technique '{override}': only compatible with tabular modality, got '{norm_mod.value}'."

    # 4. Neural-only gradient techniques require non-tree
    if norm_override in ("saliency", "integrated_gradients", "ig", "layer_ig", "layer_integrated_gradients", "gradient_shap"):
        if canonical_model == "tree (XGB/RF)":
            return False, f"Unsupported XAI technique '{override}': requires differentiable neural network and is incompatible with {canonical_model} models."

    return True, ""


class TechniqueRegistry:
    """
    Central registry for XAI explanation techniques in AdverScan.
    Validates compatibility against input data modality and model architecture.
    """

    def __init__(self):
        self._registry: Dict[str, Union[Type[BaseExplainer], Callable[[], BaseExplainer]]] = {}
        self._register_default_techniques()

    def _register_default_techniques(self) -> None:
        """Register built-in explainers."""
        self.register("shap", SHAPExplainer)
        self.register("treeshap", TreeSHAPExplainer)
        self.register("tree_shap", TreeSHAPExplainer)
        self.register("lime", LIMEExplainer)
        self.register("saliency", lambda **kw: GradientExplainer(method="saliency", **kw))
        self.register("integrated_gradients", lambda **kw: GradientExplainer(method="integrated_gradients", **kw))
        self.register("ig", lambda **kw: GradientExplainer(method="integrated_gradients", **kw))
        self.register("gradcam", lambda **kw: GradientExplainer(method="gradcam", **kw))
        self.register("grad_cam", lambda **kw: GradientExplainer(method="gradcam", **kw))
        self.register("layer_ig", LayerIGExplainer)
        self.register("layer_integrated_gradients", LayerIGExplainer)

    def register(
        self,
        name: str,
        explainer_factory: Union[Type[BaseExplainer], Callable[..., BaseExplainer]],
    ) -> None:
        """Register a new technique by name."""
        self._registry[name.lower().replace("-", "_")] = explainer_factory

    def list_techniques(self) -> List[str]:
        """List all registered technique identifiers."""
        # Deduplicate aliases
        unique_names = []
        for name in self._registry.keys():
            if name not in ("ig", "grad_cam", "tree_shap"):
                unique_names.append(name)
        return sorted(unique_names)

    def get(self, name: str, **kwargs: Any) -> BaseExplainer:
        """
        Instantiate an explainer by name.

        Args:
            name: Technique identifier (e.g., 'shap', 'treeshap', 'lime', 'saliency', 'integrated_gradients', 'gradcam').
            **kwargs: Extra parameters passed to the explainer constructor.

        Returns:
            BaseExplainer instance.
        """
        norm_name = name.lower().replace("-", "_")
        if norm_name not in self._registry:
            raise ValueError(
                f"Unsupported XAI technique '{name}'. Available techniques: {self.list_techniques()}"
            )
        factory = self._registry[norm_name]
        try:
            return factory(**kwargs)
        except TypeError:
            return factory()

    def get_compatible_techniques(
        self,
        modality: Union[str, Modality],
        model: Optional[Any] = None,
    ) -> List[str]:
        """
        Get list of techniques compatible with the given modality and model.
        """
        m = Modality.from_str(modality)
        profile = get_domain_profile(m)
        compatible = []
        unwrapped = unwrap_model(model) if model is not None else None

        # Add treeshap for tabular tree models
        available_names = list(profile.compatible_techniques)
        if m == Modality.TABULAR and unwrapped is not None and _is_tree_model(unwrapped):
            available_names.insert(0, "treeshap")

        for tech_name in available_names:
            try:
                explainer = self.get(tech_name)
                if not explainer.is_available:
                    continue
                if unwrapped is not None and hasattr(explainer, "is_compatible"):
                    if not explainer.is_compatible(m, unwrapped):
                        continue
                if tech_name not in compatible:
                    compatible.append(tech_name)
            except Exception:
                continue

        return compatible

    def select_technique(
        self,
        modality: Union[str, Modality],
        model: Optional[Any] = None,
        preferred: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseExplainer:
        """
        Select an explainer based on modality, model capabilities, and optional user preference.

        Args:
            modality: Target data modality.
            model: Target ML model.
            preferred: User's preferred technique name.
            **kwargs: Arguments to pass to the explainer.

        Returns:
            BaseExplainer instance.
        """
        m = Modality.from_str(modality)
        profile = get_domain_profile(m)
        unwrapped = unwrap_model(model) if model is not None else None

        if preferred is not None:
            norm_pref = preferred.lower().replace("-", "_")
            if norm_pref not in self._registry:
                raise ValueError(
                    f"Unsupported XAI technique '{preferred}'. Available options: {self.list_techniques()}"
                )

            # Auto-route generic 'shap' to 'treeshap' for tree models under tabular modality
            if norm_pref == "shap" and m == Modality.TABULAR and unwrapped is not None and _is_tree_model(unwrapped):
                return self.get("treeshap", **kwargs)

            explainer = self.get(norm_pref, **kwargs)

            # Check domain compatibility (treeshap is compatible with tabular)
            if norm_pref in ("treeshap", "tree_shap"):
                if m != Modality.TABULAR:
                    raise ValueError(
                        f"Unsupported XAI technique '{preferred}' for modality '{m.value}'."
                    )
            elif not profile.is_technique_compatible(norm_pref):
                raise ValueError(
                    f"Unsupported XAI technique '{preferred}' for modality '{m.value}'. "
                    f"Compatible techniques: {profile.compatible_techniques}"
                )

            # Check model compatibility
            if unwrapped is not None and hasattr(explainer, "is_compatible") and not explainer.is_compatible(m, unwrapped):
                raise ValueError(
                    f"Unsupported XAI technique '{preferred}' for model '{type(unwrapped).__name__}' "
                    f"under modality '{m.value}'."
                )

            return explainer

        # If no preferred technique specified:
        # 1. Tree models -> TreeSHAP
        if m == Modality.TABULAR and unwrapped is not None and _is_tree_model(unwrapped):
            return self.get("treeshap", **kwargs)

        # 2. Use domain profile default or first compatible
        default_tech = profile.default_technique
        compatible = self.get_compatible_techniques(m, model)

        if default_tech in compatible:
            return self.get(default_tech, **kwargs)
        elif compatible:
            return self.get(compatible[0], **kwargs)
        else:
            return self.get(default_tech, **kwargs)

    def select_by_mode(
        self,
        mode: str,
        model: Any,
        modality: Union[str, Modality],
        override: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[BaseExplainer, Dict[str, Any]]:
        """
        Select XAI explainer and decision metadata based on operational mode and model architecture.

        Args:
            mode: Operational mode preset ('fast' | 'standard' | 'thorough').
            model: Target ML model, architecture identifier, or adapter.
            modality: Data modality ('tabular', 'image', 'text', 'timeseries').
            override: Optional manual technique override.
            **kwargs: Extra parameters passed to the explainer constructor.

        Returns:
            Tuple of (explainer_instance, decision_dict).
        """
        norm_mode = str(mode).lower().strip() if mode else "standard"
        if norm_mode not in ("fast", "standard", "thorough"):
            norm_mode = "standard"

        model_type = detect_model_type(model, modality)
        norm_mod = Modality.from_str(modality) if modality else Modality.TABULAR
        modality_str = norm_mod.value if hasattr(norm_mod, "value") else str(modality)

        # 1. Manual override handling
        if override is not None and str(override).strip() and str(override).strip().lower() not in ("auto", "none", "default"):
            norm_override = str(override).strip().lower().replace("-", "_")
            compatible, err_msg = check_override_compatibility(norm_override, model_type, norm_mod, model)
            if not compatible:
                raise ValueError(err_msg)

            if norm_override not in self._registry:
                raise ValueError(
                    f"Unsupported XAI technique override '{override}'. Available techniques: {self.list_techniques()}"
                )

            explainer = self.get(norm_override, **kwargs)
            decision_dict = {
                "requested_mode": norm_mode,
                "model_type": model_type,
                "modality": modality_str,
                "technique_selected": norm_override,
                "selection_reason": f"Manual override to '{norm_override}' applied.",
            }
            return explainer, decision_dict

        # 2. Rules table lookup
        canonical_model_type = normalize_model_type(model_type)
        if canonical_model_type not in RULES_TABLE:
            canonical_model_type = "torch MLP/LSTM"

        technique_target = RULES_TABLE[canonical_model_type][norm_mode]

        # 3. Fallback check if technique does not exist in registry
        technique_selected = technique_target
        if "tree" in canonical_model_type.lower():
            selection_reason = f"Selected '{technique_target}' for {model_type} model under '{norm_mode}' mode."
        else:
            selection_reason = f"{model_type} + {norm_mode} mode (routing table)"

        if technique_target not in self._registry:
            fallback_map = {
                "gradient_shap": "integrated_gradients",
                "shap": "integrated_gradients",
                "integrated_gradients": "saliency",
                "layer_integrated_gradients": "saliency",
                "gradcam": "saliency",
                "tree_shap": "lime",
            }
            fallback = fallback_map.get(technique_target, "integrated_gradients")
            selection_reason = (
                f"Technique '{technique_target}' does not exist yet; "
                f"falling back to nearest available '{fallback}'."
            )
            technique_selected = fallback

        explainer = self.get(technique_selected, **kwargs)

        # 4. Fallback if explainer library is unavailable (e.g. shap not installed)
        if not getattr(explainer, "is_available", True):
            if "tree" in model_type.lower():
                fallback = "lime"
            else:
                fallback = "saliency"
            selection_reason = (
                f"Technique '{technique_selected}' is unavailable in current environment; "
                f"falling back to nearest available '{fallback}'."
            )
            technique_selected = fallback
            explainer = self.get(technique_selected, **kwargs)

        decision_dict = {
            "requested_mode": norm_mode,
            "model_type": model_type,
            "modality": modality_str,
            "technique_selected": technique_selected,
            "selection_reason": selection_reason,
        }
        return explainer, decision_dict


# Global default technique registry instance
technique_registry = TechniqueRegistry()


def select_by_mode(
    mode: str,
    model: Any,
    modality: Union[str, Modality],
    override: Optional[str] = None,
    registry: Optional[TechniqueRegistry] = None,
    **kwargs: Any,
) -> Tuple[BaseExplainer, Dict[str, Any]]:
    """
    Select XAI explainer and decision metadata based on operational mode and model architecture.

    Args:
        mode: Operational mode preset ('fast' | 'standard' | 'thorough').
        model: Target ML model, architecture identifier, or adapter.
        modality: Data modality ('tabular', 'image', 'text', 'timeseries').
        override: Optional manual technique override.
        registry: Optional TechniqueRegistry instance.
        **kwargs: Extra parameters passed to the explainer.

    Returns:
        Tuple of (explainer_instance, decision_dict).
    """
    reg = registry or technique_registry
    return reg.select_by_mode(mode=mode, model=model, modality=modality, override=override, **kwargs)
