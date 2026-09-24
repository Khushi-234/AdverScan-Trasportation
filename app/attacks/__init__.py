"""
Adversarial attack generation module for AdverScan.
"""

from app.attacks.base_attack import BaseAttack
from app.attacks.fgsm import FGSMAttack

__all__ = ["BaseAttack", "FGSMAttack"]
