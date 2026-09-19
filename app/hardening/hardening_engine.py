"""
Main Coordinator Module for Module 7 (Hardening) in AdverScan.

Coordinates the complete model hardening process using DefenseSelector, defensive strategies,
and iterative defense selection with configurable acceptance thresholds.
"""

import time
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

from app.hardening.defense_selector import DefenseSelector
from app.hardening.defenses import BaseDefense, get_defense_class
from app.hardening.hardening_result import DefenseAttemptResult, HardeningMetadata, HardeningResult
from app.hardening.exceptions import DefenseNotFoundError, HardeningConfigurationError, HardeningError
from app.retest.threshold import RetestThresholds, Thresholds




class HardeningEngine:
    """
    Main coordinator engine for model hardening and defensive reinforcement.

    Accepts target PyTorch models, input/label tensors, and vulnerability profiles to select,
    instantiate, execute, and evaluate defenses, supporting both single-defense application
    and iterative defense selection against configurable acceptance thresholds.
    """

    def __init__(
        self,
        defense_selector: Optional[DefenseSelector] = None,
    ) -> None:
        """
        Initialize HardeningEngine.

        Args:
            defense_selector: Optional custom DefenseSelector instance.
        """
        self.selector = defense_selector or DefenseSelector()

    def harden_iterative(
        self,
        model: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attack_name: Optional[str] = None,
        risk_level: Optional[str] = None,
        epsilon: Optional[float] = None,
        vulnerability_score: Optional[float] = None,
        thresholds: Optional[Union[RetestThresholds, Dict[str, Any]]] = None,
        candidate_defenses: Optional[List[str]] = None,
        candidate_params: Optional[Dict[str, Dict[str, Any]]] = None,
        eval_fn: Optional[Any] = None,
        baseline_metrics: Optional[Dict[str, Any]] = None,
        latency_sensitive: Optional[bool] = None,
        context: Optional[Any] = None,
        max_attempts: Optional[int] = None,
        **kwargs: Any,
    ) -> HardeningResult:
        """
        Execute iterative defense selection workflow.

        Iteratively evaluates ranked candidate defenses against the original baseline model:
        1. Ranks candidate defenses using DefenseSelector.
        2. Applies each candidate independently to the baseline model (no automatic stacking).
        3. Re-tests candidate via eval_fn and compares metrics against configurable acceptance thresholds.
        4. If a candidate satisfies the thresholds, accepts it and stops the loop.
        5. If a candidate is insufficient or harmful, automatically tries the next ranked candidate.
        6. If all candidates fail acceptance, reports that no defense provided sufficient improvement.

        Args:
            model: PyTorch model to harden.
            inputs: Optional input tensor.
            labels: Optional ground truth target label tensor.
            attack_name: Attack identifier (e.g. 'fgsm', 'pgd', 'cw').
            risk_level: Risk level ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW').
            epsilon: Perturbation magnitude.
            vulnerability_score: Baseline vulnerability score (0-100).
            thresholds: Configurable RetestThresholds instance or parameter dictionary.
            candidate_defenses: Optional explicit list of candidate defense names to test in order.
            candidate_params: Optional dictionary mapping defense names to parameters.
            eval_fn: Evaluation callable: taking model and returning metrics dict (asr, clean_accuracy, latency, etc.).
            baseline_metrics: Optional pre-computed baseline metrics.
            latency_sensitive: Whether operational context requires strict latency compliance.
            context: Optional pre-built HardeningContext.
            max_attempts: Optional cap on candidate attempts.

        Returns:
            HardeningResult: Comprehensive result detailing attempts, selected defense, and evaluation metrics.
        """
        if not isinstance(model, nn.Module):
            raise HardeningConfigurationError(f"Expected model to be PyTorch nn.Module, got {type(model)}")

        # Parse or default acceptance thresholds
        if isinstance(thresholds, dict):
            thresh_obj = RetestThresholds(**thresholds)
        elif isinstance(thresholds, RetestThresholds):
            thresh_obj = thresholds
        else:
            thresh_obj = RetestThresholds()

        is_lat_sensitive = bool(latency_sensitive)

        # 1. Rank candidate defenses via DefenseSelector if not explicitly provided
        all_candidate_params: Dict[str, Dict[str, Any]] = candidate_params or {}
        if candidate_defenses is None:
            rec_kwargs = {
                "attack_name": attack_name,
                "risk_level": risk_level,
                "epsilon": epsilon,
                "vulnerability_score": vulnerability_score,
                "latency_sensitive": is_lat_sensitive,
            }
            rec_kwargs.update(kwargs)
            recommendation = self.selector.recommend(**rec_kwargs)
            candidate_defenses = recommendation.get("candidate_defenses", [])
            for def_key, p in recommendation.get("all_candidate_params", {}).items():
                if def_key not in all_candidate_params:
                    all_candidate_params[def_key] = p


        if not candidate_defenses:
            return HardeningResult(
                hardened_model=model,
                metadata=HardeningMetadata(
                    defense_name="none",
                    defense_type="none",
                    extra_metadata={"reason": "No compatible candidate defenses found for context"},
                ),
                success=False,
                selected_defense=None,
                num_attempts=0,
                is_improved=False,
                improvement_sufficient=False,
                status="insufficient",
                thresholds=thresh_obj.to_dict(),
                recommendations=["No compatible defense candidates could be generated for the given context."],
            )

        # Cap candidates if max_attempts specified
        if max_attempts is not None and max_attempts > 0:
            candidate_defenses = candidate_defenses[:max_attempts]

        # 2. Extract / compute baseline metrics before hardening
        base_metrics: Dict[str, Any] = dict(baseline_metrics or {})
        if eval_fn is not None and callable(eval_fn) and not base_metrics:
            try:
                base_metrics = eval_fn(model) or {}
            except Exception as e:
                base_metrics = {"eval_error": str(e)}

        if vulnerability_score is not None and "vulnerability_score" not in base_metrics and "vuln_score" not in base_metrics:
            base_metrics["vulnerability_score"] = float(vulnerability_score)

        attempts: List[DefenseAttemptResult] = []
        any_improved: bool = False

        # 3. Iterative defense evaluation loop (apply -> re-test -> compare -> evaluate)
        for candidate in candidate_defenses:
            params = all_candidate_params.get(candidate, {})
            try:
                defense_cls = get_defense_class(candidate)
                defense_obj = defense_cls(**params)
            except Exception as e:
                attempt = DefenseAttemptResult(
                    defense_name=candidate,
                    status="harmful",
                    is_improved=False,
                    improvement_sufficient=False,
                    parameters=params,
                    reason=f"Failed to instantiate defense '{candidate}': {e}",
                )
                attempts.append(attempt)
                continue

            # Apply candidate defense INDEPENDENTLY to the original baseline model (Rule 8: no auto-stacking)
            t_start = time.time()
            try:
                cand_result: HardeningResult = defense_obj.apply(
                    model=model,
                    inputs=inputs,
                    labels=labels,
                )
                exec_time = time.time() - t_start
                cand_result.metadata.execution_time_seconds = exec_time
            except Exception as e:
                attempt = DefenseAttemptResult(
                    defense_name=candidate,
                    status="harmful",
                    is_improved=False,
                    improvement_sufficient=False,
                    parameters=params,
                    reason=f"Defense execution failed: {e}",
                )
                attempts.append(attempt)
                continue

            # M8 re-test: evaluate metrics after applying candidate defense
            after_metrics: Dict[str, Any] = {}
            if eval_fn is not None and callable(eval_fn):
                try:
                    after_metrics = eval_fn(cand_result.hardened_model) or {}
                except Exception as e:
                    after_metrics = {"eval_error": str(e)}

            # Compare before vs after metrics against configured acceptance thresholds
            eval_eval = thresh_obj.evaluate(
                before_metrics=base_metrics,
                after_metrics=after_metrics,
                latency_sensitive=is_lat_sensitive,
            )

            # Record attempt details
            attempt = DefenseAttemptResult(
                defense_name=candidate,
                status=eval_eval["status"],
                is_improved=eval_eval["is_improved"],
                improvement_sufficient=eval_eval["improvement_sufficient"],
                vuln_score_before=eval_eval["vulnerability_score_before"],
                vuln_score_after=eval_eval["vulnerability_score_after"],
                vuln_score_improvement=eval_eval["vulnerability_score_improvement"],
                asr_before=eval_eval["asr_before"],
                asr_after=eval_eval["asr_after"],
                asr_reduction=eval_eval["asr_reduction"],
                clean_accuracy_before=eval_eval["clean_accuracy_before"],
                clean_accuracy_after=eval_eval["clean_accuracy_after"],
                clean_accuracy_drop=eval_eval["clean_accuracy_drop"],
                clean_accuracy_drop_pct_points=eval_eval["clean_accuracy_drop_pct_points"],
                latency_ms=eval_eval["defended_latency_ms"],
                latency_overhead_ms=eval_eval["latency_overhead_ms"],
                parameters=params,
                reason=eval_eval["reason"],
            )
            attempts.append(attempt)

            if attempt.is_improved:
                any_improved = True

            # Rule 5: If defense satisfies acceptance thresholds, accept it and STOP the defense loop!
            if attempt.improvement_sufficient:
                cand_result.selected_defense = candidate
                cand_result.defense_attempts = attempts
                cand_result.num_attempts = len(attempts)
                cand_result.is_improved = True
                cand_result.improvement_sufficient = True
                cand_result.status = "accepted"
                cand_result.thresholds = thresh_obj.to_dict()
                cand_result.metrics_before = base_metrics
                cand_result.metrics_after = after_metrics
                cand_result.recommendations.append(
                    f"Iterative Defense Selection: Accepted '{candidate}' after {len(attempts)} attempt(s) "
                    f"(satisfied acceptance thresholds: {attempt.reason})."
                )
                return cand_result

            # Rule 6, 7: Defense did not satisfy thresholds, continue to next candidate

        # Rule 9: If no candidate satisfies acceptance criteria, report that no tested defense provided sufficient improvement
        all_harmful = all(a.status == "harmful" for a in attempts)
        overall_status = "harmful" if (all_harmful and attempts) else "insufficient"

        summary_recs = [
            f"Iterative Defense Selection: No tested defense ({len(attempts)} attempted: {[a.defense_name for a in attempts]}) "
            "provided sufficient improvement meeting the configured acceptance criteria. "
            "The baseline model was retained without modification."
        ]

        return HardeningResult(
            hardened_model=model,
            metadata=HardeningMetadata(
                defense_name="none",
                defense_type="none",
                parameters={},
                extra_metadata={
                    "tested_candidates": [a.defense_name for a in attempts],
                    "total_attempts": len(attempts),
                    "selection_outcome": "no_defense_accepted",
                },
            ),
            success=True,
            selected_defense=None,
            defense_attempts=attempts,
            num_attempts=len(attempts),
            is_improved=any_improved,
            improvement_sufficient=False,
            status=overall_status,
            thresholds=thresh_obj.to_dict(),
            metrics_before=base_metrics,
            metrics_after=after_metrics if attempts else {},
            recommendations=summary_recs,
        )

    def harden(
        self,
        model: nn.Module,
        defense: Union[str, BaseDefense] = "auto",
        inputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attack_name: Optional[str] = None,
        risk_level: Optional[str] = None,
        epsilon: Optional[float] = None,
        vulnerability_score: Optional[float] = None,
        defense_config: Optional[Dict[str, Any]] = None,
        eval_fn: Optional[Any] = None,
        iterative: bool = False,
        thresholds: Optional[Union[RetestThresholds, Dict[str, Any]]] = None,
        latency_sensitive: Optional[bool] = None,
        candidate_defenses: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> HardeningResult:
        """
        Execute model hardening pipeline.

        Supports both single defense execution and iterative defense selection:
        - If iterative=True or defense in ('iterative', 'loop'): runs iterative defense selection.
        - Otherwise: selects and applies the requested or auto-recommended defense.

        Args:
            model: PyTorch model to harden.
            defense: Registered defense name, 'auto' for smart selection, 'iterative' for iterative selection,
                     or a BaseDefense instance.
            inputs: Optional clean or adversarial input tensor.
            labels: Optional ground truth target label tensor.
            attack_name: Attack identifier for auto-selection.
            risk_level: Risk level for auto-selection.
            epsilon: Perturbation magnitude for auto-selection.
            vulnerability_score: Vulnerability score for auto-selection.
            defense_config: Optional kwargs passed to the defense instantiation.
            eval_fn: Optional evaluation function taking model and returning metrics dict.
            iterative: Whether to execute iterative defense selection with acceptance threshold evaluation.
            thresholds: Optional acceptance thresholds (RetestThresholds or dict).
            latency_sensitive: Operational latency constraint flag.
            candidate_defenses: Optional explicit candidate list for iterative mode.
            **kwargs: Additional context parameters.

        Returns:
            HardeningResult: DTO containing hardened model, execution metadata, evaluation metrics, and attempts.
        """
        if not isinstance(model, nn.Module):
            raise HardeningConfigurationError(f"Expected model to be PyTorch nn.Module, got {type(model)}")

        # Check for iterative defense selection request
        is_iterative_request = iterative or (isinstance(defense, str) and defense.lower().strip() in ("iterative", "loop"))
        if is_iterative_request:
            return self.harden_iterative(
                model=model,
                inputs=inputs,
                labels=labels,
                attack_name=attack_name,
                risk_level=risk_level,
                epsilon=epsilon,
                vulnerability_score=vulnerability_score,
                thresholds=thresholds,
                candidate_defenses=candidate_defenses,
                eval_fn=eval_fn,
                latency_sensitive=latency_sensitive,
                **kwargs,
            )

        # Standard single defense pipeline
        defense_config = defense_config or {}

        # Parse thresholds for evaluation
        if isinstance(thresholds, dict):
            thresh_obj = RetestThresholds(**thresholds)
        elif isinstance(thresholds, RetestThresholds):
            thresh_obj = thresholds
        else:
            thresh_obj = RetestThresholds()

        # Evaluate metrics before hardening if evaluation function is provided
        metrics_before: Dict[str, Any] = {}
        if eval_fn is not None and callable(eval_fn):
            try:
                metrics_before = eval_fn(model) or {}
            except Exception as e:
                metrics_before = {"eval_error": str(e)}

        if vulnerability_score is not None and "vulnerability_score" not in metrics_before and "vuln_score" not in metrics_before:
            metrics_before["vulnerability_score"] = float(vulnerability_score)

        # Resolve Defense instance
        if isinstance(defense, BaseDefense):
            defense_obj = defense
        elif isinstance(defense, str):
            def_str = defense.lower().strip()
            if def_str == "auto":
                defense_obj = self.selector.select(
                    attack_name=attack_name,
                    risk_level=risk_level,
                    epsilon=epsilon,
                    vulnerability_score=vulnerability_score,
                    latency_sensitive=latency_sensitive,
                    **kwargs,
                )
            else:
                defense_cls = get_defense_class(def_str)
                defense_obj = defense_cls(**defense_config)
        else:
            raise HardeningConfigurationError(f"Invalid defense parameter type: {type(defense)}")

        # Execute defense application
        t_start = time.time()
        result: HardeningResult = defense_obj.apply(
            model=model,
            inputs=inputs,
            labels=labels,
        )
        result.metadata.execution_time_seconds = time.time() - t_start

        # Attach metrics before
        result.metrics_before = metrics_before

        # Evaluate metrics after hardening if evaluation function is provided
        metrics_after: Dict[str, Any] = {}
        if eval_fn is not None and callable(eval_fn):
            try:
                metrics_after = eval_fn(result.hardened_model) or {}
            except Exception as e:
                metrics_after = {"eval_error": str(e)}

        result.metrics_after = metrics_after

        # Add selector recommendations if available
        try:
            rec_info = self.selector.recommend(
                attack_name=attack_name,
                risk_level=risk_level,
                epsilon=epsilon,
                vulnerability_score=vulnerability_score,
                latency_sensitive=latency_sensitive,
                **kwargs,
            )
            if rec_info.get("rationale") and f"Selector Context: {rec_info['rationale']}" not in result.recommendations:
                result.recommendations.append(f"Selector Context: {rec_info['rationale']}")
        except Exception:
            pass

        # Evaluate against thresholds if metrics are present
        result.selected_defense = result.metadata.defense_name
        result.num_attempts = 1
        result.thresholds = thresh_obj.to_dict()

        if metrics_before and metrics_after:
            eval_eval = thresh_obj.evaluate(
                before_metrics=metrics_before,
                after_metrics=metrics_after,
                latency_sensitive=bool(latency_sensitive),
            )
            result.is_improved = eval_eval["is_improved"]
            result.improvement_sufficient = eval_eval["improvement_sufficient"]
            result.status = eval_eval["status"]

            attempt = DefenseAttemptResult(
                defense_name=result.metadata.defense_name,
                status=eval_eval["status"],
                is_improved=eval_eval["is_improved"],
                improvement_sufficient=eval_eval["improvement_sufficient"],
                vuln_score_before=eval_eval["vulnerability_score_before"],
                vuln_score_after=eval_eval["vulnerability_score_after"],
                vuln_score_improvement=eval_eval["vulnerability_score_improvement"],
                asr_before=eval_eval["asr_before"],
                asr_after=eval_eval["asr_after"],
                asr_reduction=eval_eval["asr_reduction"],
                clean_accuracy_before=eval_eval["clean_accuracy_before"],
                clean_accuracy_after=eval_eval["clean_accuracy_after"],
                clean_accuracy_drop=eval_eval["clean_accuracy_drop"],
                clean_accuracy_drop_pct_points=eval_eval["clean_accuracy_drop_pct_points"],
                latency_ms=eval_eval["defended_latency_ms"],
                latency_overhead_ms=eval_eval["latency_overhead_ms"],
                parameters=result.metadata.parameters,
                reason=eval_eval["reason"],
            )
            result.defense_attempts = [attempt]
        else:
            result.is_improved = True
            result.improvement_sufficient = True
            result.status = "accepted"

        return result
