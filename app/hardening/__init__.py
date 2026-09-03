"""
Module 7 — Hardening package for AdverScan.

Provides adversarial hardening coordinator, defense selector, defenses (preprocessing, smoothing, adversarial training),
result DTOs, and exception handling.
"""

from app.hardening.hardening_engine import HardeningEngine
from app.hardening.defense_selector import DefenseSelector
from app.hardening.hardening_context import HardeningContext
from app.hardening.defense_capabilities import DEFENSE_CAPABILITIES, SCORING_WEIGHTS
from app.hardening.hardening_result import HardeningResult, HardeningMetadata
from app.hardening.exceptions import (
    HardeningError,
    DefenseNotFoundError,
    HardeningConfigurationError,
    DefenseExecutionError,
)
from app.hardening.defenses import (
    BaseDefense,
    SpatialSmoothingDefense,
    BitDepthReductionDefense,
    JPEGCompressionDefense,
    PreprocessingDefense,
    RandomizedSmoothingDefense,
    AdversarialTrainingDefense,
    ConfidenceRejectionDefense,
    AdversarialDetectionDefense,
    DataAugmentationDefense,
)

__all__ = [
    "HardeningEngine",
    "DefenseSelector",
    "HardeningContext",
    "DEFENSE_CAPABILITIES",
    "SCORING_WEIGHTS",
    "HardeningResult",
    "HardeningMetadata",
    "HardeningError",
    "DefenseNotFoundError",
    "HardeningConfigurationError",
    "DefenseExecutionError",
    "BaseDefense",
    "SpatialSmoothingDefense",
    "BitDepthReductionDefense",
    "JPEGCompressionDefense",
    "PreprocessingDefense",
    "RandomizedSmoothingDefense",
    "AdversarialTrainingDefense",
    "ConfidenceRejectionDefense",
    "AdversarialDetectionDefense",
    "DataAugmentationDefense",
]




