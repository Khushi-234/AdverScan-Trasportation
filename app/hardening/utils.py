"""
Utility module for Module 7 Hardening.

Defense-specific functions have been relocated directly into their respective modules
to prevent unnecessary jumping between files:
- apply_spatial_smoothing: app.hardening.defenses.preprocessing.gaussian_filter
- reduce_bit_depth: app.hardening.defenses.preprocessing.feature_squeezing
- simulate_jpeg_compression: app.hardening.defenses.preprocessing
- add_gaussian_noise: app.hardening.defenses.smoothing.randomized_smoothing
- clone_model: app.hardening.defenses.adversarial_training

This module re-exports them for shared access and backward compatibility.
Only functions shared across multiple distinct defense folders should be maintained here.
"""

from app.hardening.defenses.adversarial_training import clone_model
from app.hardening.defenses.preprocessing.gaussian_filter import apply_spatial_smoothing
from app.hardening.defenses.preprocessing.feature_squeezing import reduce_bit_depth
from app.hardening.defenses.preprocessing import simulate_jpeg_compression
from app.hardening.defenses.smoothing.randomized_smoothing import add_gaussian_noise

__all__ = [
    "clone_model",
    "add_gaussian_noise",
    "apply_spatial_smoothing",
    "reduce_bit_depth",
    "simulate_jpeg_compression",
]
