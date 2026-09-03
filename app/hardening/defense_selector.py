"""
Context-Aware Dynamic Defense Selector for Module 7 (Hardening) in AdverScan.

Analyzes attack parameters, model metadata, operational constraints, and vulnerability scores
to filter, score, rank, and recommend optimal defensive strategies and hyperparameters.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from app.hardening.defense_capabilities import DEFENSE_CAPABILITIES, SCORING_WEIGHTS
from app.hardening.defenses import get_defense_class
from app.hardening.defenses.base import BaseDefense
from app.hardening.hardening_context import HardeningContext
from app.hardening.exceptions import DefenseNotFoundError


class DefenseSelector:
    """
    Context-aware dynamic defense recommendation engine.

    Filters incompatible defenses, evaluates candidates using explicit rule-based weights,
    ranks remaining options, and returns structured recommendations.
    """

    def __init__(
        self,
        default_defense: Optional[str] = None,
        capabilities: Optional[Dict[str, Dict[str, Any]]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize DefenseSelector.

        Args:
            default_defense: Optional fallback defense key if no candidates match.
            capabilities: Optional override capability metadata dictionary.
            weights: Optional override scoring weights dictionary.
        """
        self.default_defense = default_defense
        self.capabilities = capabilities or DEFENSE_CAPABILITIES
        self.weights = weights or SCORING_WEIGHTS

    def _build_context(self, **kwargs: Any) -> HardeningContext:
        """Construct HardeningContext instance from kwargs or context object."""
        if "context" in kwargs and isinstance(kwargs["context"], HardeningContext):
            return kwargs["context"]

        ctx_kwargs: Dict[str, Any] = {}
        for field_name in HardeningContext.__dataclass_fields__:
            if field_name in kwargs and kwargs[field_name] is not None:
                ctx_kwargs[field_name] = kwargs[field_name]

        return HardeningContext(**ctx_kwargs)

    def filter_incompatible(self, context: HardeningContext) -> Tuple[List[str], Dict[str, str]]:
        """
        Eliminate defenses that cannot be executed under given context.

        Returns:
            Tuple of (eligible_defenses, rejected_defenses_dict)
        """
        eligible: List[str] = []
        rejected: Dict[str, str] = {}

        for def_key, meta in self.capabilities.items():
            # Rule 1: Input Domain Check (using metadata-driven supported_domains)
            supported_domains = meta.get("supported_domains", ["image"])
            if "*" not in supported_domains and context.input_domain not in supported_domains:
                rejected[def_key] = f"Input domain '{context.input_domain}' not supported (requires {supported_domains})"
                continue

            # Rule 2: Retraining availability check
            if meta.get("requires_retraining", False):
                if not context.allow_retraining:
                    rejected[def_key] = "Retraining is disabled by operational constraints"
                    continue
                if not context.has_training_data:
                    rejected[def_key] = "Training dataset is unavailable for model retraining"
                    continue
                if not context.has_labels:
                    rejected[def_key] = "Ground truth labels are unavailable for model retraining"
                    continue
                if context.max_hardening_time < 30.0:
                    rejected[def_key] = f"Insufficient hardening time window ({context.max_hardening_time:.1f}s < 30s) for retraining"
                    continue

            # Rule 3: Latency sensitivity check
            if context.latency_sensitive and meta.get("latency_cost", 0.0) > 30.0:
                rejected[def_key] = f"High inference latency overhead ({meta.get('latency_cost')} ms/sample) incompatible with strict latency constraint"
                continue

            eligible.append(def_key)

        return eligible, rejected

    def score_defense(self, defense_key: str, context: HardeningContext) -> float:
        """
        Calculate weighted suitability score using explicit SCORING_WEIGHTS.

        Score = w_attack * Attack_Compat + w_risk * Risk_Suitability + w_domain * Domain_Compat
                + w_resource * Resource_Suitability + w_robustness * Exp_Robustness
                - w_latency * Latency_Cost - w_training * Training_Cost
        """
        meta = self.capabilities.get(defense_key, {})
        w = self.weights

        # 1. Attack Compatibility
        attack_compat = meta["robustness_against_iterative"] if context.is_iterative else meta["robustness_against_single_step"]

        # 2. Risk Suitability
        risk_map = {"CRITICAL": 30.0, "HIGH": 20.0, "MEDIUM": 10.0, "LOW": 5.0}
        risk_weight = risk_map.get(context.risk_level, 10.0)
        risk_suitability = (attack_compat / 100.0) * risk_weight

        # 3. Domain Compatibility
        supported_domains = meta.get("supported_domains", [])
        domain_compat = 15.0 if context.input_domain in supported_domains else (10.0 if "*" in supported_domains else 0.0)

        # 4. Resource Suitability
        resource_suitability = 15.0
        if meta.get("requires_retraining", False) and context.parameter_count is not None and context.parameter_count > 100_000_000:
            resource_suitability -= 10.0

        # 5. Expected Robustness
        exp_robustness = (context.vulnerability_score / 100.0) * attack_compat

        # Costs
        latency_cost = meta.get("latency_cost", 0.0) * (2.0 if context.latency_sensitive else 0.5)
        training_cost = meta.get("training_cost", 0.0) * (0.8 if context.allow_retraining else 2.0)

        total_score = (
            w.get("attack_compatibility", 1.0) * attack_compat
            + w.get("risk_suitability", 0.2) * risk_suitability
            + w.get("domain_compatibility", 10.0) * domain_compat
            + w.get("resource_suitability", 10.0) * resource_suitability
            + w.get("expected_robustness", 0.3) * exp_robustness
            - w.get("latency_cost", 0.5) * latency_cost
            - w.get("training_cost", 0.8) * training_cost
        )
        return round(float(total_score), 2)

    def _suggest_parameters(self, defense_key: str, context: HardeningContext) -> Dict[str, Any]:
        """Generate suggested parameters for recommended primary defense."""
        eps = context.epsilon or 0.03
        if defense_key == "spatial_smoothing":
            return {"kernel_size": 3, "sigma": 1.0 if eps <= 0.03 else 1.5}
        elif defense_key == "bit_depth_reduction":
            return {"bit_depth": 4 if eps <= 0.03 else 3}
        elif defense_key == "jpeg_compression":
            return {"quality": 75 if eps <= 0.03 else 50}
        elif defense_key == "data_augmentation":
            return {"noise_std": max(eps * 0.5, 0.02), "flip_prob": 0.5, "brightness_jitter": 0.1}
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
        Generate context-aware dynamic defense recommendations.

        Returns:
            Dict containing:
                - primary_defense: Best ranked defense candidate key (or None if no compatible defense)
                - secondary_defenses: List of alternative ranked candidate keys
                - candidate_scores: Dict of candidate keys to total calculated scores
                - rejected_defenses: Dict of rejected candidate keys to rejection reasons
                - suggested_params: Dictionary of parameters for primary defense
                - rationale: Explanation of recommendation reasoning
        """
        context = self._build_context(**kwargs)

        # 1. Filter incompatible defenses
        eligible_defenses, rejected_defenses = self.filter_incompatible(context)

        if not eligible_defenses:
            if self.default_defense and self.default_defense not in rejected_defenses:
                primary = self.default_defense
                secondary: List[str] = []
                candidate_scores: Dict[str, float] = {primary: 0.0}
                rationale = "No candidate defenses matched context; returning configured default fallback defense."
                suggested_params = self._suggest_parameters(primary, context)
            else:
                # Clear empty recommendation for incompatible context when no fallback available
                primary = None
                secondary = []
                candidate_scores = {}
                rationale = "No compatible defense found for the given operational constraints, domain, and model context."
                suggested_params = {}
        else:
            # 2. Score remaining candidates
            candidate_scores = {def_key: self.score_defense(def_key, context) for def_key in eligible_defenses}

            # 3. Rank candidates
            ranked = sorted(candidate_scores.keys(), key=lambda k: candidate_scores[k], reverse=True)
            primary = ranked[0]
            secondary = ranked[1:]

            suggested_params = self._suggest_parameters(primary, context)
            top_score = candidate_scores[primary]

            rationale = (
                f"Selected '{primary}' (score: {top_score:.1f}) based on context: risk={context.risk_level}, "
                f"attack={context.attack_name or 'generic'} (iterative={context.is_iterative}), "
                f"domain={context.input_domain}, latency_sensitive={context.latency_sensitive}, "
                f"retraining_allowed={context.allow_retraining}."
            )

        return {
            "primary_defense": primary,
            "secondary_defenses": secondary,
            "candidate_scores": candidate_scores,
            "rejected_defenses": rejected_defenses,
            "suggested_params": suggested_params,
            "rationale": rationale,
        }

    def select(self, **kwargs: Any) -> BaseDefense:
        """
        Select and instantiate top primary defense instance based on context.

        Returns:
            BaseDefense: Instantiated PyTorch defense instance.

        Raises:
            DefenseNotFoundError: If no compatible defense candidate was found.
        """
        recommendation = self.recommend(**kwargs)
        defense_name = recommendation.get("primary_defense")
        if not defense_name:
            raise DefenseNotFoundError("No compatible defense candidate found for the provided evaluation context.")

        params = recommendation.get("suggested_params", {})
        defense_cls = get_defense_class(defense_name)
        return defense_cls(**params)
