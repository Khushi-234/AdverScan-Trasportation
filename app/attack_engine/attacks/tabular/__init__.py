"""
White-box adversarial attacks for tabular models.
"""

from app.attack_engine.attacks.tabular.cap_pgd import CapPGD
from app.attack_engine.attacks.tabular.lowprofool import LowProFool
from app.attack_engine.attacks.tabular.carlini_wagner import TabularCW
from app.attack_engine.attacks.tabular.jsma import TabularJSMA
from app.attack_engine.attacks.tabular.vae_attack import VAEAttack

__all__ = [
    "CapPGD",
    "LowProFool",
    "TabularCW",
    "TabularJSMA",
    "VAEAttack",
]
