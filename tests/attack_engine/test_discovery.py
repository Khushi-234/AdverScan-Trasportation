"""
Unit tests for attack discovery module.
"""

from app.attack_engine.attack_discovery import discover_attacks
from app.attack_engine.attack_registry import list_attacks, clear_registry


def test_discover_attacks():
    # Clear registry to ensure discovery auto-registers modules
    clear_registry()
    assert "fgsm" not in list_attacks()

    discovered = discover_attacks(force_reload=True)
    assert len(discovered) > 0

    registered = list_attacks()
    assert "fgsm" in registered
