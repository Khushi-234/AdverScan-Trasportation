"""
Defenses package initialization and registry for Module 7 (Hardening).

Organizes post-training / deployment-time defenses by family:
- Preprocessing (Normalization, Gaussian, Median, Denoising, Squeezing)
- Smoothing (Randomized Smoothing)
- Detection (Confidence Rejection, Adversarial Detection)
- Transformation (Random Resize, Random Crop, Padding, Rotation, Translation)
- Feature-level (Feature Denoising, Feature Alignment)
- Ensemble (Model Ensemble, Prediction Ensemble)
"""

from typing import Dict, Type
from app.hardening.defenses.base import BaseDefense

from app.hardening.defenses.wrapper import (
    HardenedModelWrapper,
    StochasticInferenceWrapper,
)

# Preprocessing family
from app.hardening.defenses.preprocessing import (
    NormalizationDefense,
    GaussianFilterDefense,
    SpatialSmoothingDefense,
    MedianFilterDefense,
    ImageDenoisingDefense,
    FeatureSqueezingDefense,
    JPEGCompressionDefense,
    PreprocessingDefense,
)

# Smoothing family
from app.hardening.defenses.smoothing import (
    RandomizedSmoothingDefense,
    RandomizedSmoothingModel,
)

# Detection family
from app.hardening.defenses.detection import (
    ConfidenceRejectionDefense,
    ConfidenceRejectionModelWrapper,
    AdversarialDetectionDefense,
    AdversarialDetectorModelWrapper,
)

# Transformation family
from app.hardening.defenses.transformation import (
    RandomResizeDefense,
    RandomCropDefense,
    PaddingDefense,
    RotationDefense,
    TranslationDefense,
)

# Feature family
from app.hardening.defenses.feature import (
    FeatureDenoisingDefense,
    FeatureDenoisedModelWrapper,
    FeatureAlignmentDefense,
    FeatureAlignmentModelWrapper,
)

# Ensemble family
from app.hardening.defenses.ensemble import (
    ModelEnsembleDefense,
    EnsembleModelWrapper,
    PredictionEnsembleDefense,
    PredictionEnsembleModelWrapper,
)

# Training imports
from app.hardening.defenses.adversarial_training import AdversarialTrainingDefense
from app.hardening.exceptions import DefenseNotFoundError

# Defense Registry mapping string identifiers to defense classes
DEFENSE_REGISTRY: Dict[str, Type[BaseDefense]] = {
    # Preprocessing
    "normalization": NormalizationDefense,
    "input_normalization": NormalizationDefense,
    "gaussian_filter": GaussianFilterDefense,
    "spatial_smoothing": SpatialSmoothingDefense,
    "median_filter": MedianFilterDefense,
    "image_denoising": ImageDenoisingDefense,
    "feature_squeezing": FeatureSqueezingDefense,
    "bit_depth_reduction": FeatureSqueezingDefense,
    "bit_depth": FeatureSqueezingDefense,
    "jpeg_compression": JPEGCompressionDefense,
    "preprocessing": PreprocessingDefense,
    # Smoothing
    "randomized_smoothing": RandomizedSmoothingDefense,
    "smoothing": RandomizedSmoothingDefense,
    # Detection
    "confidence_rejection": ConfidenceRejectionDefense,
    "confidence_rejection_defense": ConfidenceRejectionDefense,
    "rejection": ConfidenceRejectionDefense,
    "adversarial_detection": AdversarialDetectionDefense,
    "adversarial_example_detection": AdversarialDetectionDefense,
    "detection": AdversarialDetectionDefense,
    # Transformation
    "random_resize": RandomResizeDefense,
    "resize": RandomResizeDefense,
    "random_crop": RandomCropDefense,
    "crop": RandomCropDefense,
    "padding": PaddingDefense,
    "rotation": RotationDefense,
    "random_rotation": RotationDefense,
    "translation": TranslationDefense,
    "random_translation": TranslationDefense,
    # Feature
    "feature_denoising": FeatureDenoisingDefense,
    "feature_alignment": FeatureAlignmentDefense,
    "feature_consistency": FeatureAlignmentDefense,
    # Ensemble
    "model_ensemble": ModelEnsembleDefense,
    "prediction_ensemble": PredictionEnsembleDefense,
    "ensemble": ModelEnsembleDefense,
    # Legacy training defense
    "adversarial_training": AdversarialTrainingDefense,
}


def get_defense_class(defense_name: str) -> Type[BaseDefense]:
    """
    Look up defense implementation class by registered key name.

    Args:
        defense_name: Name of defense (e.g. 'normalization', 'gaussian_filter', 'randomized_smoothing').

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
    "HardenedModelWrapper",
    "StochasticInferenceWrapper",
    # Preprocessing
    "NormalizationDefense",
    "GaussianFilterDefense",
    "SpatialSmoothingDefense",
    "MedianFilterDefense",
    "ImageDenoisingDefense",
    "FeatureSqueezingDefense",
    "JPEGCompressionDefense",
    "PreprocessingDefense",
    # Smoothing
    "RandomizedSmoothingDefense",
    "RandomizedSmoothingModel",
    # Detection
    "ConfidenceRejectionDefense",
    "ConfidenceRejectionModelWrapper",
    "AdversarialDetectionDefense",
    "AdversarialDetectorModelWrapper",
    # Transformation
    "RandomResizeDefense",
    "RandomCropDefense",
    "PaddingDefense",
    "RotationDefense",
    "TranslationDefense",
    # Feature
    "FeatureDenoisingDefense",
    "FeatureDenoisedModelWrapper",
    "FeatureAlignmentDefense",
    "FeatureAlignmentModelWrapper",
    # Ensemble
    "ModelEnsembleDefense",
    "EnsembleModelWrapper",
    "PredictionEnsembleDefense",
    "PredictionEnsembleModelWrapper",
    # Legacy
    "AdversarialTrainingDefense",
    # Registry
    "DEFENSE_REGISTRY",
    "get_defense_class",
]
