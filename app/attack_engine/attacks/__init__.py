"""
Adversarial attacks package for the AdverScan framework.
"""

from app.attack_engine.attacks.base_attack import BaseAttack
from app.attack_engine.attacks.image import FGSM, PGD, DeepFool, CW, FAB
from app.attack_engine.attacks.time_series import (
    TimeSeriesPGD,
    SmoothPGD,
    WaveletAttack,
    SoftDTWAttack,
    JSMA,
)

__all__ = [
    "BaseAttack",
    "FGSM",
    "PGD",
    "DeepFool",
    "CW",
    "FAB",
    "TimeSeriesPGD",
    "SmoothPGD",
    "WaveletAttack",
    "SoftDTWAttack",
    "JSMA",
]
