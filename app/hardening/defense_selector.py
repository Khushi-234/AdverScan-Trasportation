"""
Defense Selector module for Module 7 (Hardening).

Analyzes attack parameters, perturbation scale, risk levels, and model details to select
and recommend appropriate defensive strategies, rank candidate defenses, and provide hyperparameters.
"""

from typing import Any, Dict, List, Optional, Union
from app.hardening.defenses import DEFENSE_REGISTRY, get_defense_class
from app.hardening.defenses.base import BaseDefense


class DefenseSelector:
    """
    Selects and recommends optimal defense implementations based on vulnerability analysis output.
    Supports candidate ranking for iterative defense evaluation.
    """

    def __init__(self, default_defense: str = "preprocessing") -> None:
        """
        Initialize DefenseSelector.

        Args:
            default_defense: Fallback defense identifier if recommendations yield ambiguous options.
        """
        self.default_defense = default_defense

    def _suggest_params(
        self,
        defense_name: str,
        attack: str,
        eps: float,
        risk: str,
        score: float,
        latency_sensitive: bool = False,
    ) -> Dict[str, Any]:
        """Generate default hyperparameters for a given candidate defense."""
        def_key = defense_name.lower().strip()
        if def_key in ("spatial_smoothing",):
            return {"kernel_size": 3, "sigma": 1.0}
        elif def_key in ("bit_depth_reduction", "feature_squeezing"):
            return {"bit_depth": 4 if eps <= 0.03 else 3}
        elif def_key in ("jpeg_compression",):
            return {"quality": 75 if eps <= 0.03 else 50}
        elif def_key in ("randomized_smoothing", "smoothing"):
            return {"sigma": max(eps * 1.5, 0.1), "num_samples": 5 if latency_sensitive else 10}
        elif def_key in ("adversarial_training",):
            return {
                "epochs": 2,
                "lr": 1e-4,
                "epsilon": max(eps, 0.03),
                "attack_type": "pgd" if attack in ("pgd", "bim", "") else "fgsm",
            }
        elif def_key in ("preprocessing",):
            return {"strategy": "spatial_smoothing", "kernel_size": 3}
        return {}

    def get_candidate_defenses(
        self,
        attack_name: Optional[str] = None,
        risk_level: Optional[str] = None,
        epsilon: Optional[float] = None,
        vulnerability_score: Optional[float] = None,
        latency_sensitive: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        """
        Return ordered list of ranked candidate defenses for iterative evaluation.
        """
        rec = self.recommend(
            attack_name=attack_name,
            risk_level=risk_level,
            epsilon=epsilon,
            vulnerability_score=vulnerability_score,
            latency_sensitive=latency_sensitive,
            **kwargs,
        )
        return rec.get("candidate_defenses", [])

    def select(
        self,
        attack_name: Optional[str] = None,
        risk_level: Optional[str] = None,
        epsilon: Optional[float] = None,
        vulnerability_score: Optional[float] = None,
        latency_sensitive: bool = False,
        **kwargs: Any,
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
            **kwargs,
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
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate detailed defensive recommendations, candidate rankings, and parameter suggestions.

        Returns:
            Dict containing:
                - primary_defense: Recommended defense identifier
                - secondary_defenses: Alternative options
                - candidate_defenses: Complete ordered candidate pool for iterative selection
                - suggested_params: Dictionary of parameters for primary defense
                - all_candidate_params: Dictionary of parameters for all candidate defenses
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
            secondary = ["randomized_smoothing", "preprocessing", "spatial_smoothing"]
            params = {
                "epochs": 2,
                "lr": 1e-4,
                "epsilon": max(eps, 0.03),
                "attack_type": "pgd" if attack in ("pgd", "") else "fgsm",
            }
            rationale = f"High severity risk level ({risk or 'HIGH'}) or iterative gradient attack ('{attack}'). Recommending robust Adversarial Training fine-tuning."

        elif attack == "deepfool" or (0.01 < eps <= 0.05):
            primary = "randomized_smoothing"
            secondary = ["spatial_smoothing", "bit_depth_reduction", "adversarial_training"]
            params = {"sigma": max(eps * 1.5, 0.1), "num_samples": 10}
            rationale = f"Small decision boundary perturbation attack ('{attack}', eps={eps:.4f}). Recommending Randomized Smoothing for provable noise robustness."

        elif attack == "fgsm" or risk in ("MEDIUM", "LOW") or score < 40.0:
            primary = "spatial_smoothing"
            secondary = ["bit_depth_reduction", "jpeg_compression", "randomized_smoothing"]
            params = {"kernel_size": 3, "sigma": 1.0}
            rationale = f"Single-step or moderate risk attack ('{attack}'). Recommending Spatial Smoothing input preprocessing."

        else:
            primary = self.default_defense
            secondary = ["spatial_smoothing", "randomized_smoothing", "bit_depth_reduction"]
            params = self._suggest_params(primary, attack, eps, risk, score, latency_sensitive)
            rationale = f"Fallback selection to default defense '{primary}'."

        # Complete ranked candidate pool
        candidates: List[str] = [primary]
        for s in secondary:
            if s not in candidates:
                candidates.append(s)

        # Build parameters dictionary for all candidates
        all_params: Dict[str, Dict[str, Any]] = {}
        for c in candidates:
            if c == primary and params:
                all_params[c] = params
            else:
                all_params[c] = self._suggest_params(c, attack, eps, risk, score, latency_sensitive)

        return {
            "primary_defense": primary,
            "secondary_defenses": [s for s in candidates if s != primary],
            "candidate_defenses": candidates,
            "suggested_params": params,
            "all_candidate_params": all_params,
            "rationale": rationale,
        }
