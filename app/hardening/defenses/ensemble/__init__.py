"""
Ensemble defenses package for Module 7 (Hardening).
"""

from app.hardening.defenses.ensemble.model_ensemble import (
    ModelEnsembleDefense,
    EnsembleModelWrapper,
)
from app.hardening.defenses.ensemble.prediction_ensemble import (
    PredictionEnsembleDefense,
    PredictionEnsembleModelWrapper,
)

__all__ = [
    "ModelEnsembleDefense",
    "EnsembleModelWrapper",
    "PredictionEnsembleDefense",
    "PredictionEnsembleModelWrapper",
]
