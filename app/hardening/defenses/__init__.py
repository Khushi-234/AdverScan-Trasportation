"""
Defenses package initialization and registry for Module 7 (Hardening).
"""

from typing import Dict, Type
from app.hardening.defenses.base import BaseDefense
from app.hardening.defenses.preprocessing import (
    BitDepthReductionDefense,
    JPEGCompressionDefense,
    PreprocessedModelWrapper,
    PreprocessingDefense,
    SpatialSmoothingDefense,
)
from app.hardening.defenses.smoothing import (
    RandomizedSmoothingDefense,
    RandomizedSmoothingModel,
)
from app.hardening.defenses.adversarial_training import AdversarialTrainingDefense
from app.hardening.exceptions import DefenseNotFoundError

# Defense Registry mapping string identifiers to defense classes
DEFENSE_REGISTRY: Dict[str, Type[BaseDefense]] = {
    "spatial_smoothing": SpatialSmoothingDefense,
    "bit_depth_reduction": BitDepthReductionDefense,
    "feature_squeezing": BitDepthReductionDefense,
    "jpeg_compression": JPEGCompressionDefense,
    "preprocessing": PreprocessingDefense,
    "randomized_smoothing": RandomizedSmoothingDefense,
    "smoothing": RandomizedSmoothingDefense,
    "adversarial_training": AdversarialTrainingDefense,
}


def get_defense_class(defense_name: str) -> Type[BaseDefense]:
    """
    Look up defense implementation class by registered key name.

    Args:
        defense_name: Name of defense (e.g. 'spatial_smoothing', 'randomized_smoothing', 'adversarial_training').

    Returns:
        Type[BaseDefense]: Defense implementation class.
    """
    key = defense_name.lower().strip()
    if key not in DEFENSE_REGISTRY:
        raise DefenseNotFoundError(
            f"Defense '{defense_name}' not found in registry. Available defenses: {list(DEFENSE_REGISTRY.keys())}"
        )
    return DEFENSE_REGISTRY[key]


__all__ = [
    "BaseDefense",
    "SpatialSmoothingDefense",
    "BitDepthReductionDefense",
    "JPEGCompressionDefense",
    "PreprocessingDefense",
    "PreprocessedModelWrapper",
    "RandomizedSmoothingDefense",
    "RandomizedSmoothingModel",
    "AdversarialTrainingDefense",
    "DEFENSE_REGISTRY",
    "get_defense_class",
]
