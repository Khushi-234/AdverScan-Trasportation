"""
Detection defenses package for Module 7 (Hardening).
"""

from app.hardening.defenses.detection.confidence_rejection import (
    ConfidenceRejectionDefense,
    ConfidenceRejectionModelWrapper,
)
from app.hardening.defenses.detection.adversarial_detection import (
    AdversarialDetectionDefense,
    AdversarialDetectorModelWrapper,
)

__all__ = [
    "ConfidenceRejectionDefense",
    "ConfidenceRejectionModelWrapper",
    "AdversarialDetectionDefense",
    "AdversarialDetectorModelWrapper",
]
