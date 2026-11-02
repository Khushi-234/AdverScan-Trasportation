"""
Validation Utilities for AdverScan Frontend.

Validates user input parameters, paths, and configurations prior to execution.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

SUPPORTED_FRAMEWORKS = ["pytorch", "huggingface", "onnx", "sklearn", "tensorflow"]
SUPPORTED_EXTENSIONS = [
    ".pt",
    ".pth",
    ".onnx",
    ".pkl",
    ".joblib",
    ".bin",
    ".safetensors",
]


def validate_model_path(path_str: str) -> Tuple[bool, str]:
    """Validate that the model path exists or is a valid HuggingFace identifier."""
    if not path_str or not path_str.strip():
        return False, "Model path or identifier cannot be empty."

    path_str = path_str.strip()

    # Hugging Face hub identifier (e.g., 'bert-base-uncased', 'org/model')
    if "/" in path_str and not os.path.isabs(path_str) and not Path(path_str).exists():
        # Looks like a repo ID
        return True, "Valid Hugging Face repository identifier format."

    path = Path(path_str)
    if not path.exists():
        return False, f"File or directory not found at path: {path_str}"

    if path.is_file():
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            return (
                False,
                f"Unsupported file format '{suffix}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
            )

    return True, "Valid model file or directory path."


def validate_epsilon(epsilon: float) -> Tuple[bool, str]:
    """Validate perturbation budget epsilon."""
    if epsilon <= 0.0:
        return False, "Perturbation budget (epsilon) must be strictly greater than 0.0."
    if epsilon > 1.0:
        return (
            False,
            "Epsilon is typically normalized between 0.0 and 1.0 for image/tensor inputs.",
        )
    return True, "Valid epsilon."


def validate_attack_selection(attacks: List[str]) -> Tuple[bool, str]:
    """Validate that at least one attack has been selected."""
    if not attacks:
        return False, "At least one adversarial attack must be selected."
    return True, f"Selected {len(attacks)} attack(s)."
