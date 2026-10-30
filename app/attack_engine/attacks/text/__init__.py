"""
White-box adversarial attacks for text classification models.
"""

from app.attack_engine.attacks.text.embedding_pgd import EmbeddingPGD
from app.attack_engine.attacks.text.gradient_substitution import GradientSubstitution
from app.attack_engine.attacks.text.hotflip import HotFlip
from app.attack_engine.attacks.text.textbugger_wb import TextBuggerWB
from app.attack_engine.attacks.text.universal_trigger import UniversalTrigger

__all__ = [
    "HotFlip",
    "EmbeddingPGD",
    "TextBuggerWB",
    "UniversalTrigger",
    "GradientSubstitution",
]
