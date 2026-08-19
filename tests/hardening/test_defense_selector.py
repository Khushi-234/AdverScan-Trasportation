"""
Unit tests for DefenseSelector in Module 7.
"""

import pytest
from app.hardening.defense_selector import DefenseSelector
from app.hardening.defenses import AdversarialTrainingDefense, SpatialSmoothingDefense, RandomizedSmoothingDefense


def test_selector_latency_sensitive():
    selector = DefenseSelector()
    rec = selector.recommend(attack_name="pgd", risk_level="CRITICAL", latency_sensitive=True)
    assert rec["primary_defense"] == "spatial_smoothing"

    defense_inst = selector.select(attack_name="pgd", risk_level="CRITICAL", latency_sensitive=True)
    assert isinstance(defense_inst, SpatialSmoothingDefense)


def test_selector_high_risk_pgd():
    selector = DefenseSelector()
    rec = selector.recommend(attack_name="pgd", risk_level="HIGH", vulnerability_score=85.0)
    assert rec["primary_defense"] == "adversarial_training"

    defense_inst = selector.select(attack_name="pgd", risk_level="HIGH", vulnerability_score=85.0)
    assert isinstance(defense_inst, AdversarialTrainingDefense)


def test_selector_deepfool():
    selector = DefenseSelector()
    rec = selector.recommend(attack_name="deepfool", epsilon=0.03)
    assert rec["primary_defense"] == "randomized_smoothing"

    defense_inst = selector.select(attack_name="deepfool", epsilon=0.03)
    assert isinstance(defense_inst, RandomizedSmoothingDefense)


def test_selector_fgsm_low_risk():
    selector = DefenseSelector()
    rec = selector.recommend(attack_name="fgsm", risk_level="LOW", vulnerability_score=20.0)
    assert rec["primary_defense"] == "spatial_smoothing"
