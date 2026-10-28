"""
Attack discovery module for scanning and dynamically importing attack implementations across domains.
"""

import importlib
import pkgutil
import sys
from typing import List, Set
import app.attack_engine.attacks as attacks_package
from app.attack_engine.exceptions import AttackError

# Set of full module names that have been discovered and imported
_DISCOVERED_MODULES: Set[str] = set()


def reset_discovery_state() -> None:
    """
    Reset the internal discovered modules tracker.
    """
    _DISCOVERED_MODULES.clear()


def discover_attacks(force_reload: bool = False) -> List[str]:
    """
    Automatically discover and import all attack modules under `app.attack_engine.attacks` recursively.

    Using pkgutil.walk_packages, this traverses all domain packages (e.g. image, text, tabular)
    and submodules, importing each module and triggering its self-registration via `register_attack()`.

    Args:
        force_reload: If True, re-imports/reloads all discovered modules.

    Returns:
        List of newly discovered/imported full module names.
    """
    discovered_now: List[str] = []
    package_path = attacks_package.__path__
    package_prefix = attacks_package.__name__ + "."

    for module_info in pkgutil.walk_packages(package_path, prefix=package_prefix):
        full_module_name = module_info.name
        is_pkg = module_info.ispkg

        # Skip package containers themselves; only import leaf modules
        if is_pkg:
            continue

        # Skip private/internal modules starting with '_' or base classes
        short_name = full_module_name.split(".")[-1]
        if short_name.startswith("_") or short_name == "base_attack":
            continue

        is_imported = full_module_name in _DISCOVERED_MODULES

        if not is_imported or force_reload:
            try:
                if full_module_name in sys.modules:
                    mod = sys.modules[full_module_name]
                    importlib.reload(mod)
                else:
                    mod = importlib.import_module(full_module_name)

                _DISCOVERED_MODULES.add(full_module_name)
                discovered_now.append(full_module_name)
            except Exception as e:
                raise AttackError(
                    f"Failed to discover/import attack module '{full_module_name}': {str(e)}"
                ) from e

    return discovered_now


__all__ = ["discover_attacks", "reset_discovery_state"]
