"""
Domain profiles and modality-specific configurations for AdverScan XAI.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Union
from app.explainability.modality import Modality


@dataclass
class DomainProfile:
    """
    Configuration profile defining supported techniques, defaults, and parameters for a modality.
    """

    modality: Modality
    default_technique: str
    compatible_techniques: List[str]
    baseline_strategy: str = "zeros"
    description: str = ""
    default_params: Dict[str, Any] = field(default_factory=dict)

    def is_technique_compatible(self, technique_name: str) -> bool:
        """Check if an XAI technique is compatible with this domain profile."""
        norm_name = technique_name.lower().replace("-", "_")
        return any(norm_name == t.lower().replace("-", "_") for t in self.compatible_techniques)


DOMAIN_PROFILES: Dict[Modality, DomainProfile] = {
    Modality.IMAGE: DomainProfile(
        modality=Modality.IMAGE,
        default_technique="shap",
        compatible_techniques=[
            "shap",
            "lime",
            "saliency",
            "integrated_gradients",
            "gradcam",
        ],
        baseline_strategy="zeros",
        description="Computer vision domain (2D/3D pixel grids and spatial feature maps).",
        default_params={"num_samples": 100, "steps": 50},
    ),
    Modality.TEXT: DomainProfile(
        modality=Modality.TEXT,
        default_technique="shap",
        compatible_techniques=[
            "shap",
            "lime",
            "saliency",
            "integrated_gradients",
        ],
        baseline_strategy="mask",
        description="Natural language domain (word tokens, sequences, and embeddings).",
        default_params={"num_samples": 100},
    ),
    Modality.TABULAR: DomainProfile(
        modality=Modality.TABULAR,
        default_technique="shap",
        compatible_techniques=[
            "shap",
            "lime",
            "saliency",
            "integrated_gradients",
        ],
        baseline_strategy="mean",
        description="Structured tabular domain (continuous and categorical feature columns).",
        default_params={"num_samples": 100, "steps": 50},
    ),
    Modality.TIMESERIES: DomainProfile(
        modality=Modality.TIMESERIES,
        default_technique="integrated_gradients",
        compatible_techniques=[
            "integrated_gradients",
            "saliency",
            "shap",
            "lime",
        ],
        baseline_strategy="zeros",
        description="Time-series and sequential sensor/financial domain (temporal steps and features).",
        default_params={"steps": 50},
    ),
}


def get_domain_profile(modality: Union[str, Modality]) -> DomainProfile:
    """
    Retrieve domain profile for the given modality.

    Args:
        modality: Modality enum or string representation.

    Returns:
        DomainProfile object.
    """
    m = Modality.from_str(modality)
    if m not in DOMAIN_PROFILES:
        raise ValueError(f"No domain profile found for modality: {m.value}")
    return DOMAIN_PROFILES[m]


def list_supported_domains() -> List[str]:
    """Return list of supported domain names."""
    return [m.value for m in DOMAIN_PROFILES.keys()]
