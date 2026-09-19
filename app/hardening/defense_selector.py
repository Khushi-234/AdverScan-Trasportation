"""
Context-Aware Dynamic Defense Selector for Module 7 (Hardening) in AdverScan.

Analyzes attack parameters, perturbation scale, risk levels, and model details to select
and recommend appropriate defensive strategies, rank candidate defenses, and provide hyperparameters.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from app.hardening.defense_capabilities import DEFENSE_CAPABILITIES, SCORING_WEIGHTS
from app.hardening.defenses import get_defense_class
from app.hardening.defenses.base import BaseDefense
from app.hardening.hardening_context import HardeningContext
from app.hardening.exceptions import DefenseNotFoundError


class DefenseSelector:
    """
    Selects and recommends optimal defense implementations based on vulnerability analysis output.
    Supports candidate ranking for iterative defense evaluation.
    """

    def __init__(
        self,
        default_defense: Optional[str] = None,
        capabilities: Optional[Dict[str, Dict[str, Any]]] = None,
        weights: Optional[Dict[str, float]] = None,
        eval_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize DefenseSelector.

        Args:
            default_defense: Optional fallback defense key if no candidates match.
            capabilities: Optional override capability metadata dictionary.
            weights: Optional override scoring weights dictionary for candidate ranking.
            eval_weights: Optional override weights for empirical evaluation (robustness, performance, latency).
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
            context: HardeningContext describing attack, model, and operational constraints.
            eligible_defenses: Pre-filtered list of compatible defense names.
            min_candidates: Minimum candidate count (default 3).
            max_candidates: Maximum candidate count (default 5).

        Returns:
            List[str]: Ranked list of 3 to 5 candidate defense keys.
        """
        if eligible_defenses is None:
            eligible_defenses, _ = self.filter_incompatible(context)

        # Retain only eligible defenses, strictly excluding adversarial training from auto-selection
        eligible_pool = [d for d in eligible_defenses if d != "adversarial_training"]
        if not eligible_pool:
            return []

        attack = (context.attack_name or "").lower().strip()

        # Problem 1 Fix: Attack-specific prioritization takes precedence over generic epsilon intervals
        if attack in ("fgsm", "single_step", "fast_gradient"):
            # Single-step gradient attacks: preprocessing and spatial filters excel
            primary_pool = [
                "spatial_smoothing",
                "feature_squeezing",
                "gaussian_filter",
                "median_filter",
                "image_denoising",
                "jpeg_compression",
            ]
        elif attack in ("pgd", "bim", "iterative"):
            # Iterative gradient attacks: robust smoothing, feature denoising, and detection
            primary_pool = [
                "randomized_smoothing",
                "feature_denoising",
                "feature_alignment",
                "confidence_rejection",
                "feature_squeezing",
                "spatial_smoothing",
            ]
        elif attack in ("deepfool",):
            # Decision-boundary attacks: noise smoothing, feature denoising, confidence rejection
            primary_pool = [
                "randomized_smoothing",
                "feature_denoising",
                "confidence_rejection",
                "feature_squeezing",
                "spatial_smoothing",
            ]
        elif attack in ("cw", "carlini_wagner"):
            # Optimization L2/Linf attacks: feature denoising, randomized smoothing, confidence rejection
            primary_pool = [
                "feature_denoising",
                "randomized_smoothing",
                "confidence_rejection",
                "feature_squeezing",
                "adversarial_detection",
            ]
        else:
            # General / unknown attack fallback guided by iterative vs single-step nature
            if context.is_iterative:
                primary_pool = [
                    "randomized_smoothing",
                    "feature_denoising",
                    "confidence_rejection",
                    "feature_squeezing",
                    "spatial_smoothing",
                ]
            else:
                primary_pool = [
                    "spatial_smoothing",
                    "feature_squeezing",
                    "gaussian_filter",
                    "median_filter",
                    "confidence_rejection",
                ]

        # Filter primary pool to only eligible defenses
        candidates = [d for d in primary_pool if d in eligible_pool]

        # Problem 3 Fix: Latency sensitivity sorts lightweight defenses forward without blindly overriding attack severity
        if context.latency_sensitive:
            candidates.sort(key=lambda d: self.capabilities.get(d, {}).get("latency_cost", 10.0))
        else:
            # For high-risk or critical cases, ensure robust defenses lead the candidate list
            if context.risk_level in ("CRITICAL", "HIGH") or context.vulnerability_score >= 70.0:
                robust_defenses = [d for d in candidates if self.capabilities.get(d, {}).get("robustness_against_iterative", 0.0) >= 60.0]
                other_defenses = [d for d in candidates if d not in robust_defenses]
                candidates = robust_defenses + other_defenses

        # Ensure we have at least min_candidates by pulling remaining eligible defenses sorted by score
        if len(candidates) < min_candidates:
            remaining = [d for d in eligible_pool if d not in candidates]
            remaining.sort(key=lambda d: self.score_defense(d, context), reverse=True)
            candidates.extend(remaining[: min_candidates - len(candidates)])

        # Clamp to max_candidates (keeping 3-5 candidates for practical re-testing)
        return candidates[:max_candidates]

    def _suggest_parameters(self, defense_key: str, context: HardeningContext) -> Dict[str, Any]:
        """Generate suggested parameters for recommended defense."""
        eps = context.epsilon or 0.03
        if defense_key in ("spatial_smoothing", "gaussian_filter"):
            return {"kernel_size": 3, "sigma": 1.0 if eps <= 0.03 else 1.5}
        elif defense_key == "feature_squeezing":
            return {"bit_depth": 4 if eps <= 0.03 else 3}
        elif defense_key == "normalization":
            return {"norm_type": "mean_std"}
        elif defense_key == "median_filter":
            return {"kernel_size": 3}
        elif defense_key == "image_denoising":
            return {"method": "tv", "strength": 0.05}
        elif defense_key == "jpeg_compression":
            return {"quality": 75 if eps <= 0.03 else 50}
        elif defense_key == "random_resize":
            return {"min_scale": 0.9, "max_scale": 1.1}
        elif defense_key == "random_crop":
            return {"pad_size": 4, "padding_mode": "reflect"}
        elif defense_key == "padding":
            return {"pad_size": 4, "padding_mode": "reflect"}
        elif defense_key == "rotation":
            return {"max_angle": 10.0}
        elif defense_key == "translation":
            return {"max_dx": 0.08, "max_dy": 0.08}
        elif defense_key == "feature_denoising":
            return {"method": "mean", "strength": 0.2}
        elif defense_key == "feature_alignment":
            return {"alignment_weight": 0.5, "noise_std": max(eps * 0.5, 0.02)}
        elif defense_key == "model_ensemble":
            return {"aggregation": "mean"}
        elif defense_key == "prediction_ensemble":
            return {"voting": "soft", "num_views": 3}
        elif defense_key == "randomized_smoothing":
            return {"sigma": max(eps * 1.5, 0.1), "num_samples": 10 if context.latency_sensitive else 20}
        elif defense_key == "adversarial_training":
            return {"epochs": 2 if context.max_hardening_time < 120.0 else 3, "lr": 1e-4, "epsilon": max(eps, 0.03), "attack_type": "pgd" if context.is_iterative else "fgsm"}
        elif defense_key == "confidence_rejection":
            return {"threshold": 0.6 if context.risk_level in ("CRITICAL", "HIGH") else 0.5}
        elif defense_key == "adversarial_detection":
            return {"threshold": 0.5, "method": "sensitivity" if context.is_iterative else "margin"}
        return {}

    def recommend(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Generate context-aware dynamic defense recommendations (Stage 1).

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

    def _extract_metric(self, source: Dict[str, Any], *keys: str) -> Optional[float]:
        """Safely extract float metric without boolean or-skipping of 0.0."""
        for key in keys:
            if key in source and source[key] is not None:
                try:
                    return float(source[key])
                except (ValueError, TypeError):
                    continue
        return None

    def _extract_latency_ms(self, source: Dict[str, Any]) -> Optional[float]:
        """
        Extract latency and guarantee conversion to milliseconds (ms).

        Supports:
            - latency_ms: milliseconds
            - latency: treated as milliseconds
            - execution_time_seconds / runtime_seconds: converted from seconds to ms (* 1000.0)
        """
        val_ms = self._extract_metric(source, "latency_ms", "latency")
        if val_ms is not None:
            return val_ms

        val_sec = self._extract_metric(source, "execution_time_seconds", "runtime_seconds", "latency_sec")
        if val_sec is not None:
            return round(val_sec * 1000.0, 4)

        return None

    def evaluate_candidate_results(
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


