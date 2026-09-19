"""
Defense Capability Metadata and Scoring Weight configurations for Module 7 (Hardening).

Defines realistic supported domains, operational requirements (all post-training defenses have
requires_retraining=False), latency costs, and formula scoring weights.
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
    # 1. Preprocessing Family
    "normalization": {
        "defense_type": "preprocessing",
        "defense_family": "preprocessing",
        "supported_domains": ["image", "tabular", "audio"],
        "requires_retraining": False,
        "latency_cost": 1.0,
        "training_cost": 0.0,
        "robustness_against_iterative": 25.0,
        "robustness_against_single_step": 40.0,
    },
    "gaussian_filter": {
        "defense_type": "preprocessing",
        "defense_family": "preprocessing",
        "supported_domains": ["image"],
        "requires_retraining": False,
        "latency_cost": 5.0,
        "training_cost": 0.0,
        "robustness_against_iterative": 40.0,
        "robustness_against_single_step": 75.0,
    },
    "spatial_smoothing": {
        "defense_type": "preprocessing",
        "defense_family": "preprocessing",
        "supported_domains": ["image"],
        "requires_retraining": False,
        "latency_cost": 5.0,
        "training_cost": 0.0,
        "robustness_against_iterative": 40.0,
        "robustness_against_single_step": 75.0,
    },
    "median_filter": {
        "defense_type": "preprocessing",
        "defense_family": "preprocessing",
        "supported_domains": ["image"],
        "requires_retraining": False,
        "latency_cost": 8.0,
        "training_cost": 0.0,
        "robustness_against_iterative": 50.0,
        "robustness_against_single_step": 75.0,
    },
    "image_denoising": {
        "defense_type": "preprocessing",
        "defense_family": "preprocessing",
        "supported_domains": ["image"],
        "requires_retraining": False,
        "latency_cost": 12.0,
        "training_cost": 0.0,
        "robustness_against_iterative": 55.0,
        "robustness_against_single_step": 75.0,
    },
    "feature_squeezing": {
        "defense_type": "preprocessing",
        "defense_family": "preprocessing",
        "supported_domains": ["image", "tabular"],
        "requires_retraining": False,
        "latency_cost": 4.0,
        "training_cost": 0.0,
        "robustness_against_iterative": 45.0,
        "robustness_against_single_step": 70.0,
    },
    "jpeg_compression": {
        "defense_type": "preprocessing",
        "defense_family": "preprocessing",
        "supported_domains": ["image"],
        "requires_retraining": False,
        "latency_cost": 10.0,
        "training_cost": 0.0,
        "robustness_against_iterative": 40.0,
        "robustness_against_single_step": 70.0,
    },

    # 2. Smoothing Family
    "randomized_smoothing": {
        "defense_type": "smoothing",
        "defense_family": "smoothing",
        "supported_domains": ["image", "tabular", "audio"],
        "requires_retraining": False,
        "latency_cost": 40.0,
        "training_cost": 0.0,
        "robustness_against_iterative": 80.0,
        "robustness_against_single_step": 85.0,
    },

    # 3. Detection Family
    "confidence_rejection": {
        "defense_type": "rejection",
        "defense_family": "detection",
        "supported_domains": ["image", "nlp", "tabular", "audio"],
        "requires_retraining": False,
        "latency_cost": 2.0,
        "training_cost": 0.0,
        "robustness_against_iterative": 60.0,
        "robustness_against_single_step": 65.0,
    },
    "adversarial_detection": {
        "defense_type": "detection",
        "defense_family": "detection",
        "supported_domains": ["image", "nlp", "tabular", "audio"],
        "requires_retraining": False,
        "latency_cost": 15.0,
        "training_cost": 0.0,
        "robustness_against_iterative": 75.0,
        "robustness_against_single_step": 80.0,
    },

    # 4. Input Transformation Family
    "random_resize": {
        "defense_type": "transformation",
        "defense_family": "transformation",
        "supported_domains": ["image"],
        "requires_retraining": False,
        "latency_cost": 6.0,
        "training_cost": 0.0,
        "robustness_against_iterative": 55.0,
        "robustness_against_single_step": 70.0,
    },
    "random_crop": {
        "defense_type": "transformation",
        "defense_family": "transformation",
        "supported_domains": ["image"],
        "requires_retraining": False,
        "latency_cost": 4.0,
        "training_cost": 0.0,
        "robustness_against_iterative": 50.0,
        "robustness_against_single_step": 65.0,
    },
    "padding": {
        "defense_type": "transformation",
        "defense_family": "transformation",
        "supported_domains": ["image"],
        "requires_retraining": False,
        "latency_cost": 4.0,
        "training_cost": 0.0,
        "robustness_against_iterative": 45.0,
        "robustness_against_single_step": 60.0,
    },
    "rotation": {
        "defense_type": "transformation",
        "defense_family": "transformation",
        "supported_domains": ["image"],
        "requires_retraining": False,
        "latency_cost": 7.0,
        "training_cost": 0.0,
        "robustness_against_iterative": 60.0,
        "robustness_against_single_step": 70.0,
    },
    "translation": {
        "defense_type": "transformation",
        "defense_family": "transformation",
        "supported_domains": ["image"],
        "requires_retraining": False,
        "latency_cost": 5.0,
        "training_cost": 0.0,
        "robustness_against_iterative": 55.0,
        "robustness_against_single_step": 65.0,
    },

    # 5. Feature-Level Family
    "feature_denoising": {
        "defense_type": "feature",
        "defense_family": "feature",
        "supported_domains": ["image", "tabular"],
        "requires_retraining": False,
        "latency_cost": 8.0,
        "training_cost": 0.0,
        "robustness_against_iterative": 65.0,
        "robustness_against_single_step": 75.0,
    },
    "feature_alignment": {
        "defense_type": "feature",
        "defense_family": "feature",
        "supported_domains": ["image", "tabular"],
        "requires_retraining": False,
        "latency_cost": 10.0,
        "training_cost": 0.0,
        "robustness_against_iterative": 70.0,
        "robustness_against_single_step": 75.0,
    },

    # 6. Ensemble Family
    "model_ensemble": {
        "defense_type": "ensemble",
        "defense_family": "ensemble",
        "supported_domains": ["image"],
        "requires_retraining": False,
        "latency_cost": 15.0,
        "training_cost": 0.0,
        "robustness_against_iterative": 75.0,
        "robustness_against_single_step": 85.0,
    },
    "prediction_ensemble": {
        "defense_type": "ensemble",
        "defense_family": "ensemble",
        "supported_domains": ["image"],
        "requires_retraining": False,
        "latency_cost": 18.0,
        "training_cost": 0.0,
        "robustness_against_iterative": 70.0,
        "robustness_against_single_step": 80.0,
    },

    # Legacy training defense
    "adversarial_training": {
        "defense_type": "training",
        "supported_domains": ["image", "tabular", "nlp", "audio"],
        "requires_retraining": True,
        "requires_labels": True,
        "requires_data": True,
        "latency_cost": 10.0,
        "training_cost": 80.0,
        "robustness_against_iterative": 90.0,
        "robustness_against_single_step": 95.0,
    },
}
