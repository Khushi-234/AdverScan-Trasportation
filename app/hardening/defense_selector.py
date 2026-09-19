"""
Context-Aware Dynamic Defense Selector for Module 7 (Hardening) in AdverScan.

Analyzes attack parameters, perturbation scale, risk levels, and model details to select
and recommend appropriate defensive strategies, rank candidate defenses, and provide hyperparameters.
Supports Stage 1 (heuristic recommendation) and Stage 2 (empirical candidate evaluation).
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
    Supports candidate ranking for iterative defense evaluation and empirical post-test selection.
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
        self.capabilities = capabilities if capabilities is not None else DEFENSE_CAPABILITIES
        self.weights = weights if weights is not None else SCORING_WEIGHTS
        self.eval_weights = eval_weights if eval_weights is not None else {
            "robustness": 0.50,
            "accuracy_preservation": 0.35,
            "latency": 0.15,
        }

    def _build_context(self, context: Optional[HardeningContext] = None, **kwargs: Any) -> HardeningContext:
        """Construct or normalize a HardeningContext instance from arguments."""
        if context is not None:
            return context

        valid_fields = {
            "attack_name",
            "is_iterative",
            "perturbation_norm",
            "epsilon",
            "attack_success_rate",
            "parameter_count",
            "architecture_type",
            "has_training_data",
            "device",
            "supports_gradients",
            "input_domain",
            "latency_sensitive",
            "resource_limits",
            "max_hardening_time",
            "allow_retraining",
            "has_labels",
            "risk_level",
            "vulnerability_score",
            "accuracy_drop",
            "confidence_drop",
            "perturbation_magnitude",
        }
        ctx_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields and v is not None}
        return HardeningContext(**ctx_kwargs)

    def filter_incompatible(self, context: HardeningContext) -> Tuple[List[str], Dict[str, str]]:
        """
        Filter out defense strategies incompatible with given context and constraints.

        Returns:
            Tuple of (eligible_defense_keys, rejected_defenses_dict_with_reasons)
        """
        eligible: List[str] = []
        rejected: Dict[str, str] = {}

        for def_key, meta in self.capabilities.items():
            # 1. Operational Retraining constraints
            requires_retraining = meta.get("requires_retraining", False)
            if requires_retraining:
                if not context.allow_retraining:
                    rejected[def_key] = "Retraining is disabled by operational constraints"
                    continue
                if not context.has_training_data:
                    rejected[def_key] = "Training dataset is unavailable"
                    continue
                if not context.has_labels:
                    rejected[def_key] = "Ground truth labels are unavailable"
                    continue

            # 2. Input domain check
            supported_domains = meta.get("supported_domains", ["image"])
            if "*" not in supported_domains and context.input_domain not in supported_domains:
                rejected[def_key] = f"Input domain '{context.input_domain}' not supported"
                continue

            # 3. Latency constraints
            latency_cost = meta.get("latency_cost", 10.0)
            if context.latency_sensitive and latency_cost >= 30.0:
                rejected[def_key] = "High inference latency overhead"
                continue

            eligible.append(def_key)

        return eligible, rejected

    def score_defense(self, defense_name: str, context: HardeningContext) -> float:
        """Compute multi-criteria heuristic score for a candidate defense."""
        meta = self.capabilities.get(defense_name, {})
        w = self.weights

        # Robustness based on attack nature
        if context.is_iterative:
            expected_rob = meta.get("robustness_against_iterative", 50.0)
        else:
            expected_rob = meta.get("robustness_against_single_step", 50.0)

        # Domain compatibility
        supported_domains = meta.get("supported_domains", ["image"])
        domain_compat = 1.0 if ("*" in supported_domains or context.input_domain in supported_domains) else 0.0

        # Operational resource suitability
        requires_retraining = meta.get("requires_retraining", False)
        resource_compat = 0.0 if (requires_retraining and not context.allow_retraining) else 1.0

        # Latency & training costs
        latency_cost = meta.get("latency_cost", 10.0)
        training_cost = meta.get("training_cost", 0.0)

        # Attack compatibility bonus
        attack = (context.attack_name or "").lower().strip()
        attack_compat = 0.0
        if attack in ("fgsm", "single_step", "fast_gradient") and meta.get("defense_type") == "preprocessing":
            attack_compat = 20.0
        elif attack in ("pgd", "bim", "iterative") and (expected_rob >= 60.0 or meta.get("defense_family") in ("smoothing", "feature", "detection")):
            attack_compat = 25.0
        elif attack in ("cw", "carlini_wagner") and meta.get("defense_family") in ("feature", "detection", "smoothing"):
            attack_compat = 20.0
        elif attack == "deepfool" and meta.get("defense_family") in ("smoothing", "feature", "detection"):
            attack_compat = 20.0

        # Risk suitability bonus
        risk_bonus = 0.0
        if context.risk_level in ("CRITICAL", "HIGH") or context.vulnerability_score >= 70.0:
            risk_bonus = expected_rob * 0.5
        elif context.risk_level == "LOW" or context.vulnerability_score < 40.0:
            risk_bonus = (50.0 - latency_cost) * 0.2

        # Latency penalty multiplier if latency sensitive
        lat_weight = w.get("latency_cost", 0.5) * (3.0 if context.latency_sensitive else 1.0)

        score = (
            attack_compat * w.get("attack_compatibility", 1.0)
            + expected_rob * w.get("expected_robustness", 0.3)
            + domain_compat * w.get("domain_compatibility", 10.0)
            + resource_compat * w.get("resource_suitability", 10.0)
            + risk_bonus * w.get("risk_suitability", 0.2)
            - latency_cost * lat_weight
            - training_cost * w.get("training_cost", 0.8)
        )
        return round(score, 4)

    def rank_candidates(
        self,
        context: HardeningContext,
        eligible_defenses: Optional[List[str]] = None,
        min_candidates: int = 3,
        max_candidates: int = 5,
    ) -> List[str]:
        """
        Rank candidate defenses for iterative evaluation based on context.

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

        # Attack-specific prioritization takes precedence over generic epsilon intervals
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

        # Latency sensitivity sorts lightweight defenses forward without blindly overriding attack severity
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

        # Sort candidate list so candidates[0] has top composite score among candidates
        candidates = candidates[:max_candidates]
        candidates.sort(key=lambda d: self.score_defense(d, context), reverse=True)
        return candidates

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
            return {
                "epochs": 2 if context.max_hardening_time < 120.0 else 3,
                "lr": 1e-4,
                "epsilon": max(eps, 0.03),
                "attack_type": "pgd" if context.is_iterative else "fgsm",
            }
        elif defense_key == "confidence_rejection":
            return {"threshold": 0.6 if context.risk_level in ("CRITICAL", "HIGH") else 0.5}
        elif defense_key == "adversarial_detection":
            return {"threshold": 0.5, "method": "sensitivity" if context.is_iterative else "margin"}
        return {}

    def _suggest_params(
        self,
        defense_name: str,
        attack: str,
        eps: float,
        risk: str,
        score: float,
        latency_sensitive: bool = False,
    ) -> Dict[str, Any]:
        """Legacy helper for default hyperparameters generation."""
        ctx = HardeningContext(
            attack_name=attack,
            epsilon=eps,
            risk_level=risk,
            vulnerability_score=score,
            latency_sensitive=latency_sensitive,
        )
        return self._suggest_parameters(defense_name, ctx)

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
        Generate context-aware dynamic defense recommendations (Stage 1).

        Returns:
            Dict containing:
                - primary_defense: Recommended defense identifier
                - secondary_defenses: Alternative candidate options
                - candidate_defenses: Complete ordered candidate pool for iterative evaluation
                - candidate_scores: Score mapping for candidates
                - rejected_defenses: Defenses rejected by constraints with explanations
                - suggested_params: Parameter dictionary for primary defense
                - all_candidate_params: Parameter dictionary for all candidate defenses
                - rationale: Explanation of defense selection reasoning
                - selection_basis: Contextual metrics used during selection
        """
        context = self._build_context(
            attack_name=attack_name,
            risk_level=risk_level,
            epsilon=epsilon,
            vulnerability_score=vulnerability_score,
            latency_sensitive=latency_sensitive,
            **kwargs,
        )

        eligible_defenses, rejected_defenses = self.filter_incompatible(context)
        candidates = self.rank_candidates(context, eligible_defenses=eligible_defenses)

        if not candidates:
            # Fallback to default defense if provided and eligible
            if self.default_defense and self.default_defense in self.capabilities:
                primary = self.default_defense
                secondaries = []
                candidates = [primary]
                candidate_scores = {primary: self.score_defense(primary, context)}
            else:
                return {
                    "primary_defense": None,
                    "secondary_defenses": [],
                    "candidate_defenses": [],
                    "candidate_scores": {},
                    "rejected_defenses": rejected_defenses,
                    "suggested_params": {},
                    "all_candidate_params": {},
                    "rationale": "No compatible defenses found matching the context and constraints.",
                    "selection_basis": {
                        "attack": context.attack_name,
                        "risk_level": context.risk_level,
                        "epsilon": context.epsilon,
                        "vulnerability_score": context.vulnerability_score,
                        "latency_sensitive": context.latency_sensitive,
                    },
                }
        else:
            primary = candidates[0]
            secondaries = candidates[1:]
            candidate_scores = {d: self.score_defense(d, context) for d in candidates}

        suggested_params = self._suggest_parameters(primary, context) if primary else {}
        all_candidate_params = {d: self._suggest_parameters(d, context) for d in candidates}

        rationale = (
            f"Selected primary defense '{primary}' for {context.risk_level} risk {context.attack_name.upper()} attack "
            f"(vulnerability score: {context.vulnerability_score:.1f}, eps: {context.epsilon:.4f}). "
            f"Evaluated {len(candidates)} compatible candidates."
        )

        return {
            "primary_defense": primary,
            "secondary_defenses": secondaries,
            "candidate_defenses": candidates,
            "candidate_scores": candidate_scores,
            "rejected_defenses": rejected_defenses,
            "suggested_params": suggested_params,
            "all_candidate_params": all_candidate_params,
            "rationale": rationale,
            "selection_basis": {
                "attack": context.attack_name,
                "risk_level": context.risk_level,
                "epsilon": context.epsilon,
                "vulnerability_score": context.vulnerability_score,
                "latency_sensitive": context.latency_sensitive,
            },
        }

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

        Returns:
            BaseDefense: Instantiated defense instance ready for execution.
        """
        rec = self.recommend(
            attack_name=attack_name,
            risk_level=risk_level,
            epsilon=epsilon,
            vulnerability_score=vulnerability_score,
            latency_sensitive=latency_sensitive,
            **kwargs,
        )

        defense_name = rec.get("primary_defense")
        if not defense_name:
            raise DefenseNotFoundError("No compatible defense could be selected for the given context.")

        params = rec.get("suggested_params", {})
        defense_cls = get_defense_class(defense_name)
        return defense_cls(**params)

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
        candidate_metrics: Dict[str, Dict[str, Any]],
        baseline_metrics: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Stage 2: Empirically evaluate re-testing results for each candidate defense.

        Args:
            candidate_metrics: Dictionary mapping candidate defense key to its re-tested metrics.
            baseline_metrics: Baseline metrics before hardening.

        Returns:
            Dictionary mapping candidate defense key to detailed empirical evaluation metrics.
        """
        evaluations: Dict[str, Dict[str, Any]] = {}

        base_acc = self._extract_metric(baseline_metrics, "clean_accuracy", "accuracy")
        base_asr = self._extract_metric(baseline_metrics, "attack_success_rate", "asr")
        base_lat = self._extract_latency_ms(baseline_metrics) or 0.0

        for cand_name, cand_m in candidate_metrics.items():
            cand_acc = self._extract_metric(cand_m, "clean_accuracy", "accuracy")
            cand_asr = self._extract_metric(cand_m, "attack_success_rate", "asr")
            cand_lat = self._extract_latency_ms(cand_m)
            defended_lat = cand_lat if cand_lat is not None else base_lat

            # Latency overhead calculation
            latency_overhead = round(max(0.0, defended_lat - base_lat), 4) if base_lat is not None else 0.0
            if latency_overhead <= 0.0:
                latency_efficiency_score = 1.0
            else:
                latency_efficiency_score = round(max(0.0, 1.0 / (1.0 + latency_overhead / 20.0)), 4)

            # Clean accuracy drop calculation
            if base_acc is None or base_acc <= 0.0 or cand_acc is None:
                acc_drop = 0.0
                acc_drop_pct_points = 0.0
                clean_preservation_score = 1.0
            else:
                acc_drop = round(max(0.0, base_acc - cand_acc), 4)
                acc_drop_pct_points = round(acc_drop * 100.0, 4)
                if acc_drop_pct_points <= 5.0:
                    clean_preservation_score = max(0.0, 1.0 - (acc_drop_pct_points / 20.0))
                else:
                    clean_preservation_score = max(0.0, 0.75 - ((acc_drop_pct_points - 5.0) / 25.0))

            # Robustness evaluation
            if base_asr is None or cand_asr is None:
                robustness_available = False
                robustness_improvement = None
                is_effective = None
                status = "partial_evaluation_no_asr"
                composite_score = round(
                    clean_preservation_score * 0.70 + latency_efficiency_score * 0.30,
                    4,
                )
            else:
                robustness_available = True
                robustness_improvement = round(base_asr - cand_asr, 4)
                is_effective = robustness_improvement > 0.0

                if not is_effective:
                    status = "ineffective_defense"
                    # Penalize composite score heavily for ineffective defenses (< 0.40)
                    composite_score = round(
                        max(0.0, (robustness_improvement * 0.50 + clean_preservation_score * 0.35 + latency_efficiency_score * 0.15) * 0.5),
                        4,
                    )
                else:
                    status = "robustness_verified"
                    rob_norm = max(0.0, min(1.0, robustness_improvement))
                    composite_score = round(
                        rob_norm * self.eval_weights.get("robustness", 0.50)
                        + clean_preservation_score * self.eval_weights.get("accuracy_preservation", 0.35)
                        + latency_efficiency_score * self.eval_weights.get("latency", 0.15),
                        4,
                    )

            evaluations[cand_name] = {
                "defense_name": cand_name,
                "baseline_clean_accuracy": base_acc,
                "defended_clean_accuracy": cand_acc,
                "clean_accuracy_drop": acc_drop,
                "clean_accuracy_drop_pct_points": acc_drop_pct_points,
                "clean_preservation_score": clean_preservation_score,
                "baseline_asr": base_asr,
                "defended_asr": cand_asr,
                "robustness_available": robustness_available,
                "robustness_improvement": robustness_improvement,
                "is_effective": is_effective,
                "baseline_latency_ms": base_lat,
                "defended_latency_ms": defended_lat,
                "latency_overhead_ms": latency_overhead,
                "latency_efficiency_score": latency_efficiency_score,
                "composite_score": composite_score,
                "status": status,
            }

        return evaluations

    def select_best_evaluated(self, evaluations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select the winning defense based on empirical evaluation results.
        """
        if not evaluations:
            return {
                "best_defense": None,
                "selection_status": "no_candidates",
                "ranked_defenses": [],
                "selection_rationale": "No evaluated candidate defenses provided.",
            }

        # Sort candidates by composite_score descending
        ranked = sorted(
            evaluations.keys(),
            key=lambda k: evaluations[k].get("composite_score", 0.0),
            reverse=True,
        )
        winner_key = ranked[0]
        winner_info = evaluations[winner_key]
        status = winner_info.get("status", "evaluated")

        # Construct data-supported rationale
        if status == "partial_evaluation_no_asr":
            rationale = (
                f"Selected '{winner_key}' based on clean accuracy preservation and latency overhead. "
                f"However, baseline ASR was unavailable, leaving empirical robustness unverified."
            )
        elif status == "ineffective_defense":
            rationale = (
                f"Warning: Defense failed to improve robustness for '{winner_key}' "
                f"(ASR increased or remained unchanged by {winner_info.get('robustness_improvement')})."
            )
        else:
            rob_imp = winner_info.get("robustness_improvement", 0.0) or 0.0
            rob_pct = rob_imp * 100.0
            acc_drop_pct = winner_info.get("clean_accuracy_drop_pct_points", 0.0) or 0.0
            lat_ovh = winner_info.get("latency_overhead_ms", 0.0) or 0.0

            rationale = (
                f"Empirically selected '{winner_key}' as best defense (composite score: {winner_info.get('composite_score', 0.0):.4f}). "
                f"It reduced ASR by {rob_pct:.1f}% with {acc_drop_pct:.1f} percentage points clean accuracy drop "
                f"and {lat_ovh:.1f} ms latency overhead. Defense successfully reinforced model robustness."
            )

        return {
            "best_defense": winner_key,
            "selection_status": status,
            "ranked_defenses": ranked,
            "winner_evaluation": winner_info,
            "selection_rationale": rationale,
        }
