"""
Module 7 — Hardening package for AdverScan.

Provides adversarial hardening coordinator, defense selector, defenses (preprocessing, smoothing, adversarial training),
result DTOs, and exception handling.
"""

from app.hardening.hardening_engine import HardeningEngine
from app.hardening.defense_selector import DefenseSelector
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
)

__all__ = [
    "HardeningEngine",
    "DefenseSelector",
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
]
