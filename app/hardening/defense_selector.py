"""
Defense Selector module for Module 7 (Hardening).

Analyzes attack parameters, perturbation scale, risk levels, and model details to select
and recommend appropriate defensive strategies and hyperparameters.
"""

from typing import Any, Dict, List, Optional, Union
from app.hardening.defenses import DEFENSE_REGISTRY, get_defense_class
from app.hardening.defenses.base import BaseDefense


class DefenseSelector:
    """
    Selects and recommends optimal defense implementations based on vulnerability analysis output.
    """

    def __init__(self, default_defense: str = "preprocessing") -> None:
        """
        Initialize DefenseSelector.

        Args:
            default_defense: Fallback defense identifier if recommendations yield ambiguous options.
        """
        self.default_defense = default_defense

    def select(
        self,
        attack_name: Optional[str] = None,
        risk_level: Optional[str] = None,
        epsilon: Optional[float] = None,
        vulnerability_score: Optional[float] = None,
        latency_sensitive: bool = False,
    ) -> BaseDefense:
        """
        Select and instantiate an appropriate defense instance based on provided attributes.

        Args:
            attack_name: Identifier of attack (e.g., 'fgsm', 'pgd', 'deepfool').
            risk_level: Risk classification ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW').
            epsilon: Perturbation magnitude bound.
            vulnerability_score: Score between 0.0 and 100.0.
            latency_sensitive: If True, prefers fast input preprocessing over adversarial fine-tuning.

        Returns:
            BaseDefense: Instantiated defense instance ready for execution.
        """
        recommendation = self.recommend(
            attack_name=attack_name,
            risk_level=risk_level,
            epsilon=epsilon,
            vulnerability_score=vulnerability_score,
            latency_sensitive=latency_sensitive,
        )

        defense_name = recommendation["primary_defense"]
        params = recommendation.get("suggested_params", {})
        defense_cls = get_defense_class(defense_name)
        return defense_cls(**params)

    def recommend(
        self,
        attack_name: Optional[str] = None,
        risk_level: Optional[str] = None,
        epsilon: Optional[float] = None,
        vulnerability_score: Optional[float] = None,
        latency_sensitive: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate detailed defensive recommendations and parameter suggestions.

        Returns:
            Dict containing:
                - primary_defense: Recommended defense identifier
                - secondary_defenses: Alternative options
                - suggested_params: Dictionary of parameters for primary defense
                - rationale: Explanation of defense selection reasoning
        """
        attack = (attack_name or "").lower().strip()
        risk = (risk_level or "").upper().strip()
        eps = epsilon if epsilon is not None else 0.0
        score = vulnerability_score if vulnerability_score is not None else 0.0

        # Decision Tree Logic
        if latency_sensitive:
            primary = "spatial_smoothing"
            secondary = ["bit_depth_reduction", "jpeg_compression"]
            params = {"kernel_size": 3, "sigma": 1.0}
            rationale = "Latency sensitive constraint specified. Selecting low-overhead Spatial Smoothing preprocessing defense."

        elif attack in ("pgd", "bim") or risk in ("CRITICAL", "HIGH") or score >= 70.0:
            primary = "adversarial_training"
            secondary = ["randomized_smoothing", "preprocessing"]
            params = {
                "epochs": 2,
                "lr": 1e-4,
                "epsilon": max(eps, 0.03),
                "attack_type": "pgd" if attack in ("pgd", "") else "fgsm",
            }
            rationale = f"High severity risk level ({risk or 'HIGH'}) or iterative gradient attack ('{attack}'). Recommending robust Adversarial Training fine-tuning."

        elif attack == "deepfool" or (0.01 < eps <= 0.05):
            primary = "randomized_smoothing"
            secondary = ["spatial_smoothing", "adversarial_training"]
            params = {"sigma": max(eps * 1.5, 0.1), "num_samples": 10}
            rationale = f"Small decision boundary perturbation attack ('{attack}', eps={eps:.4f}). Recommending Randomized Smoothing for provable noise robustness."

        elif attack == "fgsm" or risk in ("MEDIUM", "LOW") or score < 40.0:
            primary = "spatial_smoothing"
            secondary = ["bit_depth_reduction", "jpeg_compression"]
            params = {"kernel_size": 3, "sigma": 1.0}
            rationale = f"Single-step or moderate risk attack ('{attack}'). Recommending Spatial Smoothing input preprocessing."

        else:
            primary = self.default_defense
            secondary = ["spatial_smoothing", "randomized_smoothing"]
            params = {}
            rationale = f"Fallback selection to default defense '{primary}'."

        return {
            "primary_defense": primary,
            "secondary_defenses": secondary,
            "suggested_params": params,
            "rationale": rationale,
        }
