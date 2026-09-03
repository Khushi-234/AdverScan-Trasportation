"""
Defense Capability Metadata and Scoring Weight configurations for Module 7 (Hardening).

Defines supported domains (using '*' for domain-agnostic defenses), operational requirements,
latency/training costs, and explicit formula scoring weights.
"""

from typing import Any, Dict, List

# Explicit formula weights for rule-based defense scoring
SCORING_WEIGHTS: Dict[str, float] = {
    "attack_compatibility": 1.0,
    "risk_suitability": 0.2,
    "domain_compatibility": 10.0,
    "resource_suitability": 10.0,
    "expected_robustness": 0.3,
    "latency_cost": 0.5,
    "training_cost": 0.8,
}

# Static capability registry for registered defenses
DEFENSE_CAPABILITIES: Dict[str, Dict[str, Any]] = {
    "spatial_smoothing": {
        "defense_type": "preprocessing",
        "supported_domains": ["image"],
        "requires_retraining": False,
        "latency_cost": 5.0,
        "training_cost": 0.0,
        "robustness_against_iterative": 40.0,
        "robustness_against_single_step": 75.0,
    },
    "bit_depth_reduction": {
        "defense_type": "preprocessing",
        "supported_domains": ["image"],
        "requires_retraining": False,
        "latency_cost": 5.0,
        "training_cost": 0.0,
        "robustness_against_iterative": 35.0,
        "robustness_against_single_step": 70.0,
    },
    "jpeg_compression": {
        "defense_type": "preprocessing",
        "supported_domains": ["image"],
        "requires_retraining": False,
        "latency_cost": 10.0,
        "training_cost": 0.0,
        "robustness_against_iterative": 40.0,
        "robustness_against_single_step": 70.0,
    },
    "data_augmentation": {
        "defense_type": "preprocessing",
        "supported_domains": ["image", "*"],
        "requires_retraining": False,
        "latency_cost": 10.0,
        "training_cost": 0.0,
        "robustness_against_iterative": 45.0,
        "robustness_against_single_step": 75.0,
    },
    "randomized_smoothing": {
        "defense_type": "smoothing",
        "supported_domains": ["*"],
        "requires_retraining": False,
        "latency_cost": 40.0,
        "training_cost": 0.0,
        "robustness_against_iterative": 80.0,
        "robustness_against_single_step": 85.0,
    },
    "adversarial_training": {
        "defense_type": "training",
        "supported_domains": ["*"],
        "requires_retraining": True,
        "requires_labels": True,
        "requires_data": True,
        "latency_cost": 10.0,
        "training_cost": 80.0,
        "robustness_against_iterative": 90.0,
        "robustness_against_single_step": 95.0,
    },
    "confidence_rejection": {
        "defense_type": "rejection",
        "supported_domains": ["*"],
        "requires_retraining": False,
        "latency_cost": 2.0,
        "training_cost": 0.0,
        "robustness_against_iterative": 60.0,
        "robustness_against_single_step": 65.0,
    },
    "adversarial_detection": {
        "defense_type": "detection",
        "supported_domains": ["*"],
        "requires_retraining": False,
        "latency_cost": 15.0,
        "training_cost": 0.0,
        "robustness_against_iterative": 75.0,
        "robustness_against_single_step": 80.0,
    },
}
