"""
Unit tests for attack selector.
"""

import pytest
import torch.nn as nn

from app.attack_engine.selector.attack_selector import (
    select_attacks,
    select_compatible_attacks,
)
from app.attack_engine.attacks.fgsm import FGSM
from app.attack_engine.exceptions import AttackConfigurationError


class DummyModel(nn.Module):
    pass


def test_select_attacks_single():
    selected = select_attacks("fgsm")
    assert len(selected) == 1
    assert selected[0] is FGSM


def test_select_attacks_list():
    selected = select_attacks(["fgsm"])
    assert len(selected) == 1
    assert selected[0] is FGSM


def test_select_attacks_unknown():
    with pytest.raises(AttackConfigurationError):
        select_attacks(["fgsm", "nonexistent_attack"])


def test_select_compatible_attacks():
    model = DummyModel()
    selected = select_compatible_attacks(model, "fgsm")
    assert len(selected) == 1
    assert selected[0] is FGSM


def test_select_compatible_attacks_no_model():
    with pytest.raises(AttackConfigurationError):
        select_compatible_attacks(None, "fgsm")
