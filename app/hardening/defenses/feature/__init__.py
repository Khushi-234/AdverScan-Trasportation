"""
Feature-level defenses package for Module 7 (Hardening).
"""

from app.hardening.defenses.feature.feature_denoising import (
    FeatureDenoisingDefense,
    FeatureDenoisedModelWrapper,
    denoise_features,
)
from app.hardening.defenses.feature.feature_alignment import (
    FeatureAlignmentDefense,
    FeatureAlignmentModelWrapper,
)

__all__ = [
    "FeatureDenoisingDefense",
    "FeatureDenoisedModelWrapper",
    "denoise_features",
    "FeatureAlignmentDefense",
    "FeatureAlignmentModelWrapper",
]
