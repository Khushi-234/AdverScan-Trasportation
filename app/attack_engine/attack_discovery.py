"""
Attack discovery module for scanning and dynamically importing attack implementations.
"""

import importlib
import pkgutil
import sys
from typing import List, Set
import app.attack_engine.attacks as attacks_package
from app.attack_engine.exceptions import AttackError

# Set of full module names that have been discovered and imported by discovery
_DISCOVERED_MODULES: Set[str] = set()


def reset_discovery_state() -> None:
    """
    Reset the internal discovered modules tracker.

    Called when the registry is cleared to allow subsequent discovery calls
    to re-import/reload attack modules and re-trigger self-registration.
    """
    _DISCOVERED_MODULES.clear()


def discover_attacks(force_reload: bool = False) -> List[str]:
    """
    Automatically discover and import all attack modules in the `attacks` package.

    Importing each attack module triggers its self-registration via `register_attack()`.
    Discovery tracks imported modules independently of filenames or registered attack names.

    Args:
        force_reload: If True, re-imports/reloads all discovered modules.

    Returns:
        List of newly discovered/imported full module names.
    """
    discovered_now: List[str] = []
    package_path = attacks_package.__path__
    package_name = attacks_package.__name__

    for _, module_name, is_pkg in pkgutil.iter_modules(package_path):
        # Skip package directories or private/internal modules starting with '_'
        if is_pkg or module_name.startswith("_"):
            continue

        full_module_name = f"{package_name}.{module_name}"

        # Determine if module needs to be imported or reloaded
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
