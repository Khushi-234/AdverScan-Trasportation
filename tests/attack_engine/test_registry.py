"""
Unit tests for attack registry.
"""

import pytest
from app.attack_engine.registry.attack_registry import (
    register_attack,
    get_attack,
    list_attacks,
)
from app.attack_engine.attacks.fgsm import FGSM
from app.attack_engine.base.base_attack import BaseAttack
from app.attack_engine.exceptions import AttackConfigurationError


class MockAttack(BaseAttack):
    def generate(self, inputs, labels, config=None):
        return inputs


def test_list_attacks():
    attacks = list_attacks()
    assert "fgsm" in attacks


def test_get_attack_fgsm():
    fgsm_cls = get_attack("fgsm")
    assert fgsm_cls is FGSM


def test_get_attack_case_insensitive():
    fgsm_cls = get_attack("FGSM")
    assert fgsm_cls is FGSM


def test_get_attack_unknown():
    with pytest.raises(AttackConfigurationError):
        get_attack("unknown_attack_name")


def test_register_attack():
    register_attack("mock_attack", MockAttack)
    assert "mock_attack" in list_attacks()
    assert get_attack("mock_attack") is MockAttack


def test_register_invalid_class():
    class NotAnAttack:
        pass

    with pytest.raises(AttackConfigurationError):
        register_attack("invalid", NotAnAttack)
