"""
Technique registry for selecting and managing compatible XAI explainers in AdverScan.
"""

from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch.nn as nn

from app.explainability.domain_profiles import get_domain_profile
from app.explainability.utils import unwrap_model
from app.explainability.modality import Modality
from app.explainability.techniques.base import BaseExplainer
from app.explainability.techniques.gradient_explainer import GradientExplainer
from app.explainability.techniques.lime_explainer import LIMEExplainer
from app.explainability.techniques.shap_explainer import SHAPExplainer, TreeSHAPExplainer


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
        return factory(**kwargs)

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


# Global default technique registry instance
technique_registry = TechniqueRegistry()
