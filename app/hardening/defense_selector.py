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
        self.capabilities = capabilities or DEFENSE_CAPABILITIES
        self.weights = weights or SCORING_WEIGHTS
        self.eval_weights = eval_weights or {
            "robustness": 0.60,
            "performance": 0.25,
            "latency": 0.15,
        }

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

        # Costs: latency cost penalizes high latency in sensitive contexts without blindly overriding attack severity
        latency_cost = meta.get("latency_cost", 0.0) * (2.0 if context.latency_sensitive else 0.5)

        # Adversarial training is not automatically applied: apply strong training penalty so post-training defenses rank higher
        training_multiplier = 2.0 if meta.get("requires_retraining", False) else 0.8
        training_cost = meta.get("training_cost", 0.0) * training_multiplier

        # Give post-training defenses a priority bonus over retraining-heavy defenses
        post_training_bonus = 20.0 if not meta.get("requires_retraining", False) else 0.0

        total_score = (
            w.get("attack_compatibility", 1.0) * attack_compat
            + w.get("risk_suitability", 0.2) * risk_suitability
            + w.get("domain_compatibility", 10.0) * domain_compat
            + w.get("resource_suitability", 10.0) * resource_suitability
            + w.get("expected_robustness", 0.3) * exp_robustness
            + post_training_bonus
            - w.get("latency_cost", 0.5) * latency_cost
            - w.get("training_cost", 0.8) * training_cost
        )
        return round(float(total_score), 2)

    def get_candidate_defenses(
        self,
        context: HardeningContext,
        eligible_defenses: Optional[List[str]] = None,
        min_candidates: int = 3,
        max_candidates: int = 5,
    ) -> List[str]:
        """
        Stage 1: Generate a concise, ranked pool of 3 to 5 suitable defense candidates.

        Prioritizes attack-specific compatibility over generic epsilon thresholds,
        treats latency as an operational constraint/penalty rather than an absolute blind override,
        and strictly excludes adversarial training from being selected as an applied defense.

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
            Dict containing:
                - primary_defense: Best ranked candidate key (or None if no compatible defense)
                - secondary_defenses: List of alternative candidate keys
                - candidate_defenses: Complete curated candidate pool (3-5 defenses)
                - candidate_scores: Dict of candidate keys to calculated scores
                - rejected_defenses: Dict of rejected candidate keys to rejection reasons
                - suggested_params: Dictionary of parameters for primary defense
                - all_candidate_params: Dictionary of parameters for all candidates
                - selection_basis: Contextual metrics used during selection
                - rationale: Clear, explainable rationale for logging and presentation
        """
        context = self._build_context(**kwargs)

        # 1. Filter incompatible defenses
        eligible_defenses, rejected_defenses = self.filter_incompatible(context)

        if not eligible_defenses:
            if self.default_defense and self.default_defense not in rejected_defenses:
                primary = self.default_defense
                secondary: List[str] = []
                candidates: List[str] = [primary]
                candidate_scores: Dict[str, float] = {primary: 0.0}
                rationale = "No candidate defenses matched context; returning configured default fallback defense."
                suggested_params = self._suggest_parameters(primary, context)
                all_params = {primary: suggested_params}
            else:
                primary = None
                secondary = []
                candidates = []
                candidate_scores = {}
                rationale = "No compatible defense found for the given operational constraints, domain, and model context."
                suggested_params = {}
                all_params = {}
        else:
            # 2. Score all eligible defenses for backward compatibility
            candidate_scores = {def_key: self.score_defense(def_key, context) for def_key in eligible_defenses}

            # 3. Generate curated 3 to 5 defense candidates
            candidates = self.get_candidate_defenses(context, eligible_defenses=eligible_defenses)

            # Sort candidate pool by suitability score
            candidates.sort(key=lambda k: candidate_scores.get(k, 0.0), reverse=True)

            primary = candidates[0] if candidates else None

            # Fallback if candidates pool was empty but eligible defenses exist
            if not primary and eligible_defenses:
                eligible_sorted = sorted(eligible_defenses, key=lambda k: candidate_scores[k], reverse=True)
                primary = eligible_sorted[0]
                candidates = [primary]

            secondary = [d for d in candidates if d != primary]

            suggested_params = self._suggest_parameters(primary, context) if primary else {}
            all_params = {d: self._suggest_parameters(d, context) for d in candidates}
            top_score = candidate_scores.get(primary, 0.0) if primary else 0.0

            rationale = (
                f"Attack: {context.attack_name.upper() or 'GENERIC'} | Risk: {context.risk_level} | "
                f"Vulnerability Score: {context.vulnerability_score:.1f} | Latency Sensitive: {context.latency_sensitive}. "
                f"Generated {len(candidates)} defense candidates: {candidates}. "
                f"Primary recommendation is '{primary}' (suitability score: {top_score:.1f}). "
                "Final defense selection should be based on post-defense empirical re-test results "
                "to confirm robustness improvement and clean performance preservation."
            )

        selection_basis = {
            "attack": context.attack_name,
            "is_iterative": context.is_iterative,
            "risk_level": context.risk_level,
            "vulnerability_score": context.vulnerability_score,
            "epsilon": context.epsilon,
            "latency_sensitive": context.latency_sensitive,
            "input_domain": context.input_domain,
        }

        return {
            "primary_defense": primary,
            "secondary_defenses": secondary,
            "candidate_defenses": candidates,
            "candidate_scores": candidate_scores,
            "rejected_defenses": rejected_defenses,
            "suggested_params": suggested_params,
            "all_candidate_params": all_params,
            "selection_basis": selection_basis,
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
        candidate_metrics: Dict[str, Any],
        baseline_metrics: Optional[Dict[str, Any]] = None,
        weights: Optional[Dict[str, float]] = None,
        max_acceptable_clean_drop: float = 0.05,
        severe_clean_drop_threshold: float = 0.10,
        max_acceptable_latency_overhead: float = 50.0,
    ) -> Dict[str, Any]:
        """
        Stage 2: Empirically evaluate re-tested candidate defense metrics.

        Calculates:
            - Robustness improvement (baseline_asr - defended_asr) with missing baseline protection.
            - Clean accuracy drop in decimal fraction and percentage points with zero-baseline protection.
            - Latency overhead standardized to milliseconds (defended_latency_ms - baseline_latency_ms).
            - Composite score balancing robustness improvement, clean accuracy retention, and latency efficiency.
            - Explicit evaluation status (robustness_verified, ineffective_defense, partial_evaluation_no_asr).

        Args:
            candidate_metrics: Dict mapping candidate defense names to metric dicts or ComparisonResult objects.
            baseline_metrics: Baseline metrics before hardening (asr, clean_accuracy, latency).
            weights: Optional custom composite scoring weights (robustness, performance, latency).
            max_acceptable_clean_drop: Tolerable drop threshold (default 0.05 = 5 percentage points).
            severe_clean_drop_threshold: Severe degradation threshold (default 0.10 = 10 percentage points).
            max_acceptable_latency_overhead: Reference latency overhead threshold in ms for scaling.

        Returns:
            Dict[str, Any]: Detailed evaluation metrics and composite scores per candidate.
        """
        w = weights or self.eval_weights
        w_rob = w.get("robustness", 0.60)
        w_perf = w.get("performance", 0.25)
        w_lat = w.get("latency", 0.15)

        base = baseline_metrics or {}
        b_asr = self._extract_metric(base, "attack_success_rate", "asr")
        b_clean = self._extract_metric(base, "clean_accuracy", "accuracy")
        b_lat_ms = self._extract_latency_ms(base)

        evaluations: Dict[str, Any] = {}

        for def_name, data in candidate_metrics.items():
            # Handle dictionary, DTO, or ComparisonResult object
            if hasattr(data, "to_dict") and callable(data.to_dict):
                d = data.to_dict()
            elif isinstance(data, dict):
                d = data
            else:
                d = vars(data) if hasattr(data, "__dict__") else {}

            d_after = d.get("after_assessment") if isinstance(d.get("after_assessment"), dict) else {}
            d_before = d.get("before_assessment") if isinstance(d.get("before_assessment"), dict) else {}

            # Defended metrics extraction without boolean 'or' falsy 0.0 traps
            d_asr = self._extract_metric(d, "attack_success_rate", "asr")
            if d_asr is None and d_after:
                d_asr = self._extract_metric(d_after, "attack_success_rate", "asr")

            d_clean = self._extract_metric(d, "clean_accuracy", "accuracy")
            if d_clean is None and d_after:
                d_clean = self._extract_metric(d_after, "clean_accuracy", "accuracy")

            d_lat_ms = self._extract_latency_ms(d)
            if d_lat_ms is None and d_after:
                d_lat_ms = self._extract_latency_ms(d_after)

            # Fallback baseline extraction from ComparisonResult if baseline_metrics was omitted
            cand_b_asr = b_asr
            if cand_b_asr is None and d_before:
                cand_b_asr = self._extract_metric(d_before, "attack_success_rate", "asr")

            cand_b_clean = b_clean
            if cand_b_clean is None and d_before:
                cand_b_clean = self._extract_metric(d_before, "clean_accuracy", "accuracy")

            cand_b_lat_ms = b_lat_ms
            if cand_b_lat_ms is None and d_before:
                cand_b_lat_ms = self._extract_latency_ms(d_before)

            if cand_b_lat_ms is None:
                cand_b_lat_ms = 0.0

            # 1. Robustness Evaluation (normalized bounded metric)
            if cand_b_asr is not None and d_asr is not None:
                robustness_available = True
                robustness_improvement = round(float(cand_b_asr) - float(d_asr), 4)
                is_effective = robustness_improvement > 0.0

                # ASR is in [0.0, 1.0]; reduction is bounded in [-1.0, 1.0] without fragile zero-division
                norm_robustness = max(-1.0, min(1.0, robustness_improvement))
                relative_robustness = round(robustness_improvement / float(cand_b_asr), 4) if float(cand_b_asr) > 0.0 else None
                eval_status = "robustness_verified" if is_effective else "ineffective_defense"
            else:
                # Explicit missing ASR handling
                robustness_available = False
                robustness_improvement = None
                relative_robustness = None
                is_effective = None
                norm_robustness = None
                eval_status = "partial_evaluation_no_asr"

            # 2. Clean Accuracy Evaluation (explicit fraction and percentage points)
            if cand_b_clean is not None and d_clean is not None:
                clean_available = True
                clean_accuracy_drop = round(float(cand_b_clean) - float(d_clean), 4)
                clean_accuracy_drop_pct_points = round(clean_accuracy_drop * 100.0, 2)

                # Unified, consistent threshold penalty schedule
                if clean_accuracy_drop <= 0.0:
                    clean_preservation_score = 1.0
                elif clean_accuracy_drop <= max_acceptable_clean_drop:
                    # Minor drop within tolerance (e.g. <= 5 pts): gentle slope from 1.0 to 0.85
                    clean_preservation_score = round(1.0 - (clean_accuracy_drop / max_acceptable_clean_drop) * 0.15, 4)
                elif clean_accuracy_drop <= severe_clean_drop_threshold:
                    # Moderate drop between 5% and 10%: slope from 0.85 down to 0.40
                    fraction = (clean_accuracy_drop - max_acceptable_clean_drop) / max(1e-5, (severe_clean_drop_threshold - max_acceptable_clean_drop))
                    clean_preservation_score = round(0.85 - (fraction * 0.45), 4)
                else:
                    # Severe degradation > 10%: steep penalty towards 0.0
                    excess = clean_accuracy_drop - severe_clean_drop_threshold
                    clean_preservation_score = max(0.0, round(0.40 - (excess * 4.0), 4))
            else:
                clean_available = False
                clean_accuracy_drop = None
                clean_accuracy_drop_pct_points = None
                clean_preservation_score = 1.0

            # 3. Latency Evaluation (guaranteed millisecond consistency)
            if d_lat_ms is not None:
                latency_available = True
                latency_overhead_ms = round(float(d_lat_ms) - float(cand_b_lat_ms), 4)
                if latency_overhead_ms <= 0.0:
                    latency_efficiency_score = 1.0
                else:
                    latency_efficiency_score = max(0.0, round(1.0 - (latency_overhead_ms / max_acceptable_latency_overhead), 4))
            else:
                latency_available = False
                latency_overhead_ms = 0.0
                latency_efficiency_score = 1.0

            # 4. Composite Score Calculation
            if not robustness_available:
                # Dynamically rebalance remaining weights (performance + latency) without faking zero ASR
                total_remaining = w_perf + w_lat
                norm_w_perf = w_perf / total_remaining if total_remaining > 0 else 0.5
                norm_w_lat = w_lat / total_remaining if total_remaining > 0 else 0.5
                composite_score = round(norm_w_perf * clean_preservation_score + norm_w_lat * latency_efficiency_score, 4)
            else:
                rob_contrib = w_rob * norm_robustness
                composite_score = round(rob_contrib + w_perf * clean_preservation_score + w_lat * latency_efficiency_score, 4)
                composite_score = max(0.0, min(1.0, composite_score))

            evaluations[def_name] = {
                "defense_name": def_name,
                "composite_score": composite_score,
                "status": eval_status,
                "robustness_available": robustness_available,
                "robustness_improvement": robustness_improvement,
                "relative_robustness_improvement": relative_robustness,
                "is_effective": is_effective,
                "clean_available": clean_available,
                "clean_accuracy_drop": clean_accuracy_drop,
                "clean_accuracy_drop_pct_points": clean_accuracy_drop_pct_points,
                "clean_preservation_score": clean_preservation_score,
                "latency_available": latency_available,
                "latency_overhead_ms": latency_overhead_ms,
                "latency_overhead": latency_overhead_ms,
                "latency_efficiency_score": latency_efficiency_score,
                "defended_asr": d_asr,
                "baseline_asr": cand_b_asr,
                "defended_clean_accuracy": d_clean,
                "baseline_clean_accuracy": cand_b_clean,
                "defended_latency_ms": d_lat_ms,
                "baseline_latency_ms": cand_b_lat_ms,
                "defended_latency": d_lat_ms,
                "baseline_latency": cand_b_lat_ms,
            }

        return evaluations

    def select_best_evaluated(
        self,
        candidate_evaluations: Dict[str, Any],
        max_acceptable_clean_drop: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Final Selection: Identify the best defense candidate based on empirical evaluation results.

        Args:
            candidate_evaluations: Dictionary returned by evaluate_candidate_results.
            max_acceptable_clean_drop: Acceptable threshold for clean drop in rationale generation.

        Returns:
            Dict containing best_defense, best_composite_score, selection_status, ranked_defenses, and supported selection_rationale.
        """
        if not candidate_evaluations:
            return {
                "best_defense": None,
                "best_composite_score": 0.0,
                "selection_status": "no_candidates",
                "ranked_defenses": [],
                "evaluations": {},
                "selection_rationale": "No candidate defense evaluations provided.",
            }

        # Sort candidates descending by composite_score, breaking ties by robustness_improvement
        ranked = sorted(
            candidate_evaluations.keys(),
            key=lambda k: (
                candidate_evaluations[k].get("composite_score", 0.0),
                candidate_evaluations[k].get("robustness_improvement") or 0.0,
            ),
            reverse=True,
        )

        best = ranked[0]
        best_data = candidate_evaluations[best]
        status = best_data.get("status", "unknown")

        # Construct fully data-supported rationale
        parts = []
        if best_data.get("robustness_available"):
            if best_data.get("is_effective"):
                parts.append(
                    f"reduced ASR by {best_data['robustness_improvement'] * 100:.1f}% "
                    f"({best_data['baseline_asr']} -> {best_data['defended_asr']})"
                )
            else:
                parts.append(
                    f"ineffective against attack: ASR increased by {abs(best_data['robustness_improvement']) * 100:.1f}% "
                    f"({best_data['baseline_asr']} -> {best_data['defended_asr']})"
                )
        else:
            parts.append("baseline ASR was unavailable (evaluated on clean preservation and latency efficiency)")

        if best_data.get("clean_available"):
            drop = best_data.get("clean_accuracy_drop") or 0.0
            drop_pts = best_data.get("clean_accuracy_drop_pct_points") or 0.0
            if drop <= 0.0:
                parts.append(f"clean accuracy fully preserved ({best_data['baseline_clean_accuracy']} -> {best_data['defended_clean_accuracy']})")
            elif drop <= max_acceptable_clean_drop:
                parts.append(f"clean accuracy preserved within {drop_pts:.1f} percentage points ({best_data['baseline_clean_accuracy']} -> {best_data['defended_clean_accuracy']})")
            else:
                parts.append(f"incurred clean accuracy degradation of {drop_pts:.1f} percentage points ({best_data['baseline_clean_accuracy']} -> {best_data['defended_clean_accuracy']})")

        if best_data.get("latency_available"):
            overhead = best_data.get("latency_overhead_ms", 0.0)
            sign = "+" if overhead >= 0 else ""
            parts.append(f"latency overhead: {sign}{overhead:.1f}ms")

        summary_clause = "; ".join(parts)

        if status == "robustness_verified":
            if (best_data.get("clean_accuracy_drop") or 0.0) <= max_acceptable_clean_drop:
                conclusion = "Defense successfully reinforced model robustness with acceptable clean performance and latency trade-offs."
            else:
                conclusion = "Defense improved robustness, but incurred notable clean accuracy degradation."
        elif status == "partial_evaluation_no_asr":
            conclusion = "Defense selected based on operational and clean metrics; robustness unverified due to missing baseline ASR."
        elif status == "ineffective_defense":
            conclusion = "Warning: Defense failed to improve robustness for this attack."
        else:
            conclusion = "Empirical evaluation completed."

        rationale = (
            f"Empirically selected '{best}' (composite score: {best_data['composite_score']:.2f}, status: {status}): "
            f"{summary_clause}. {conclusion}"
        )

        return {
            "best_defense": best,
            "best_composite_score": best_data["composite_score"],
            "selection_status": status,
            "ranked_defenses": ranked,
            "evaluations": candidate_evaluations,
            "selection_rationale": rationale,
        }


