"""
Image domain adversarial attacks suite.
"""

from app.attack_engine.attacks.image.fgsm import FGSM
from app.attack_engine.attacks.image.pgd import PGD
from app.attack_engine.attacks.image.deepfool import DeepFool
from app.attack_engine.attacks.image.cw import CW
from app.attack_engine.attacks.image.fab import FAB

__all__ = ["FGSM", "PGD", "DeepFool", "CW", "FAB"]
