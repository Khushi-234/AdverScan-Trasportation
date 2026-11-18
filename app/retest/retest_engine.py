"""
RetestEngine for Module 8 (Re-Test & Comparison) in AdverScan.

Coordinates controlled re-evaluation of hardened models using identical
dataset, attack suite, and configuration parameters as used prior to hardening.
Reuses existing AttackEngine, BaselineEvaluator, VulnerabilityEngine, and ComparisonEngine.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn

from app.ingestion.adapters.base_adapter import BaseModelAdapter
from app.ingestion.adapters.pytorch_adapter import PyTorchAdapter
from app.evaluation.evaluator import BaselineEvaluator
from app.evaluation.dataset_loader import BaseDatasetLoader
from app.evaluation.results import EvaluationResult
from app.attack_engine.attack_engine import AttackEngine
from app.attack_engine.config import AttackConfig
from app.attack_engine.models import AttackMetadata, AttackResult, AttackResults
from app.vulnerability_analysis.vulnerability_engine import VulnerabilityEngine
from app.orchestration.dataset_adapter import InMemoryDatasetLoader
from app.retest.comparison import ComparisonEngine
from app.retest.retest_result import RetestResult, ComparisonResult
from app.retest.threshold import RetestThresholds, Thresholds


class RetestEngine:
    """
    Main Module 8 coordinator.

    Receives hardened model (or adapter / HardeningResult DTO) and evaluation context,
    runs baseline evaluation, executes attacks via AttackEngine, runs VulnerabilityEngine,
    and performs BEFORE vs AFTER comparison.
    """

    def __init__(
        self,
        vulnerability_engine: Optional[VulnerabilityEngine] = None,
        comparison_engine: Optional[ComparisonEngine] = None,
    ) -> None:
        """
        Initialize RetestEngine.

        Args:
            vulnerability_engine: Optional VulnerabilityEngine instance.
            comparison_engine: Optional ComparisonEngine instance.
        """
        self.vulnerability_engine = vulnerability_engine or VulnerabilityEngine()
        self.comparison_engine = comparison_engine or ComparisonEngine()

    def _resolve_adapter(self, hardened_model: Any, device: str = "cpu") -> BaseModelAdapter:
        """Helper to standardize hardened model into a BaseModelAdapter."""
        while (
            hasattr(hardened_model, "hardened_model")
            and getattr(hardened_model, "hardened_model") is not None
            and getattr(hardened_model, "hardened_model") is not hardened_model
        ):
            hardened_model = getattr(hardened_model, "hardened_model")

        if isinstance(hardened_model, BaseModelAdapter):
            return hardened_model
        elif isinstance(hardened_model, (nn.Module, torch.jit.ScriptModule)):
            return PyTorchAdapter(hardened_model, device=device)
        elif hasattr(hardened_model, "predict") and callable(getattr(hardened_model, "predict")):
            from app.ingestion.adapters.sklearn_adapter import SklearnAdapter
            return SklearnAdapter(hardened_model, device=device)
        else:
            raise TypeError(f"Unsupported model type for RetestEngine: {type(hardened_model)}")

    def retest(
        self,
        hardened_model: Any,
        dataset_loader: BaseDatasetLoader,
        attacks: List[str],
        before_baseline_result: Union[EvaluationResult, Dict[str, Any]],
        before_vulnerability_analysis: Dict[str, Dict[str, Any]],
        attack_configs: Optional[Dict[str, Union[AttackConfig, Dict[str, Any]]]] = None,
        before_attack_results: Optional[Dict[str, Any]] = None,
        num_classes: Optional[int] = None,
        model_name: str = "hardened_model",
        device: str = "cpu",
        batch_size: int = 32,
        thresholds: Optional[Union[RetestThresholds, Dict[str, Any]]] = None,
        defense_config: Optional[Dict[str, Any]] = None,
        defense_metadata: Optional[Dict[str, Any]] = None,
        defense_name: Optional[str] = None,
        original_model: Optional[Any] = None,
        latency_sensitive: bool = False,
        **kwargs: Any,
    ) -> RetestResult:
        """
        Execute controlled re-testing of the hardened model under identical conditions.

        Args:
            hardened_model: PyTorch model, BaseModelAdapter, or HardeningResult DTO.
            dataset_loader: BaseDatasetLoader instance used during original evaluation.
            attacks: List of attack identifiers (e.g. ['fgsm', 'pgd', 'deepfool']).
            before_baseline_result: Baseline clean EvaluationResult or dict from before hardening.
            before_vulnerability_analysis: Dict mapping attack_name -> {"assessment": ..., "scoring": ...} from before.
            attack_configs: Optional dict mapping attack_name -> AttackConfig or dict parameters.
            before_attack_results: Optional dict of attack results from before hardening.
            num_classes: Optional number of output classes.
            model_name: Descriptive model identifier name.
            device: Computing device ('cpu', 'cuda').
            batch_size: Evaluation batch size.
            thresholds: Optional acceptance thresholds for validating robustness.
            defense_config: Optional defense hyperparameter dictionary.
            defense_metadata: Optional metadata describing applied defense.
            defense_name: Optional identifier name of applied defense.
            original_model: Optional baseline model prior to defense application.
            latency_sensitive: Flag indicating strict latency constraint.

        Returns:
            RetestResult: Structured DTO containing complete re-test and comparative delta analysis.
        """
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Resolve thresholds
        if thresholds is None and hasattr(hardened_model, "thresholds") and getattr(hardened_model, "thresholds"):
            thresh_candidate = getattr(hardened_model, "thresholds")
            threshold_obj = RetestThresholds(**thresh_candidate) if isinstance(thresh_candidate, dict) else thresh_candidate
        elif isinstance(thresholds, dict):
            threshold_obj = RetestThresholds(**thresholds)
        elif isinstance(thresholds, RetestThresholds):
            threshold_obj = thresholds
        else:
            threshold_obj = RetestThresholds()

        # Extract defense metadata from HardeningResult if available
        if defense_name is None and hasattr(hardened_model, "selected_defense"):
            defense_name = getattr(hardened_model, "selected_defense")
        if defense_metadata is None and hasattr(hardened_model, "metadata"):
            raw_meta = getattr(hardened_model, "metadata")
            if hasattr(raw_meta, "to_dict"):
                defense_metadata = raw_meta.to_dict()
            elif hasattr(raw_meta, "__dict__"):
                defense_metadata = raw_meta.__dict__
            elif isinstance(raw_meta, dict):
                defense_metadata = raw_meta
        if defense_config is None and defense_metadata and "parameters" in defense_metadata:
            defense_config = defense_metadata["parameters"]

        # Preserve model training/eval state
        orig_training = (
            getattr(original_model, "training", False)
            if (original_model is not None and hasattr(original_model, "training"))
            else False
        )
        hardened_training = (
            getattr(hardened_model, "training", False)
            if hasattr(hardened_model, "training")
            else False
        )

        try:
            adapter = self._resolve_adapter(hardened_model, device=device)
            attack_configs = attack_configs or {}
            before_attack_results = before_attack_results or {}

            # Standardize before_baseline_result dict representation
            if isinstance(before_baseline_result, dict):
                before_baseline_dict = dict(before_baseline_result)
            elif hasattr(before_baseline_result, "to_dict"):
                before_baseline_dict = before_baseline_result.to_dict()
            else:
                before_baseline_dict = {}

            # Step 1: Baseline Evaluation of Hardened Model
            baseline_evaluator = BaselineEvaluator(
                adapter=adapter,
                dataset_loader=dataset_loader,
                num_classes=num_classes,
                model_name=model_name,
            )
            after_baseline_result: EvaluationResult = baseline_evaluator.evaluate(output_dir=None, show_progress=False)
            after_baseline_dict = after_baseline_result.to_dict()

            # Extract all batches for attack execution
            all_batches = []
            for batch_pixels, batch_targets, _ in dataset_loader.iterate_batches():
                all_batches.append((batch_pixels, batch_targets))

            if not all_batches:
                raise ValueError("Provided dataset_loader produced no batches.")

            # Step 2: Execute Attacks on Hardened Model using AttackEngine
            attack_engine = AttackEngine(adapter)
            after_attack_results_coll = AttackResults()
            after_attack_results_dict: Dict[str, Any] = {}
            after_adv_evaluations_dict: Dict[str, EvaluationResult] = {}
            failed_or_unsupported_attacks: Dict[str, Dict[str, Any]] = {}

            for attack_name in attacks:
                atk_cfg_raw = attack_configs.get(attack_name.lower())
                atk_config_obj = None
                if isinstance(atk_cfg_raw, AttackConfig):
                    atk_config_obj = atk_cfg_raw
                elif isinstance(atk_cfg_raw, dict):
                    known_fields = {"epsilon", "clip_min", "clip_max", "loss_fn", "params"}
                    config_kwargs = {k: v for k, v in atk_cfg_raw.items() if k in known_fields}
                    extra_params = {k: v for k, v in atk_cfg_raw.items() if k not in known_fields}
                    if "params" in config_kwargs and isinstance(config_kwargs["params"], dict):
                        config_kwargs["params"] = {**extra_params, **config_kwargs["params"]}
                    elif extra_params:
                        config_kwargs["params"] = extra_params
                    atk_config_obj = AttackConfig(**config_kwargs)

                try:
                    # Validate compatibility with target adapter
                    from app.attack_engine.attack_selector import select_compatible_attacks
                    compat_classes = select_compatible_attacks(adapter, attack_name)
                    if not compat_classes:
                        raise ValueError(f"No compatible implementation found for attack '{attack_name}'.")

                    batch_orig_list = []
                    batch_adv_list = []
                    batch_labels_list = []
                    total_atk_time = 0.0
                    last_metadata = None

                    for b_pixels, b_targets in all_batches:
                        atk_res_b: AttackResult = attack_engine.run_attack(
                            attack_name=attack_name,
                            inputs=b_pixels,
                            labels=b_targets,
                            config=atk_config_obj,
                        )
                        batch_orig_list.append(atk_res_b.original_inputs if atk_res_b.original_inputs is not None else b_pixels)
                        batch_adv_list.append(atk_res_b.adversarial_examples)
                        b_lbls = atk_res_b.labels if atk_res_b.labels is not None else b_targets
                        batch_labels_list.append(b_lbls)
                        total_atk_time += atk_res_b.metadata.execution_time_seconds
                        last_metadata = atk_res_b.metadata

                    # Combine per-batch results
                    first_adv = batch_adv_list[0]
                    if isinstance(first_adv, torch.Tensor):
                        try:
                            combined_adv = torch.cat(batch_adv_list, dim=0)
                        except RuntimeError:
                            combined_adv = batch_adv_list
                    elif isinstance(first_adv, np.ndarray):
                        try:
                            combined_adv = np.concatenate(batch_adv_list, axis=0)
                        except ValueError:
                            combined_adv = batch_adv_list
                    elif isinstance(first_adv, list):
                        combined_adv = [item for sub in batch_adv_list for item in sub]
                    else:
                        combined_adv = batch_adv_list[0]

                    first_orig = batch_orig_list[0]
                    if isinstance(first_orig, torch.Tensor):
                        try:
                            combined_orig = torch.cat(batch_orig_list, dim=0)
                        except RuntimeError:
                            combined_orig = batch_orig_list
                    elif isinstance(first_orig, np.ndarray):
                        try:
                            combined_orig = np.concatenate(batch_orig_list, axis=0)
                        except ValueError:
                            combined_orig = batch_orig_list
                    elif isinstance(first_orig, list):
                        combined_orig = [item for sub in batch_orig_list for item in sub]
                    else:
                        combined_orig = batch_orig_list[0]

                    first_lbl = batch_labels_list[0]
                    if isinstance(first_lbl, torch.Tensor):
                        try:
                            combined_labels = torch.cat(batch_labels_list, dim=0)
                        except RuntimeError:
                            combined_labels = batch_labels_list
                    elif isinstance(first_lbl, np.ndarray):
                        try:
                            combined_labels = np.concatenate(batch_labels_list, axis=0)
                        except ValueError:
                            combined_labels = batch_labels_list
                    elif isinstance(first_lbl, list):
                        combined_labels = [item for sub in batch_labels_list for item in sub]
                    else:
                        combined_labels = batch_labels_list[0]

                    merged_params = dict(last_metadata.parameters) if (last_metadata and last_metadata.parameters) else {}
                    if atk_config_obj:
                        if atk_config_obj.params:
                            merged_params.update(atk_config_obj.params)
                        if atk_config_obj.epsilon is not None and "epsilon" not in merged_params:
                            merged_params["epsilon"] = atk_config_obj.epsilon

                    combined_metadata = AttackMetadata(
                        name=attack_name,
                        attack_name=last_metadata.attack_name if last_metadata else attack_name,
                        attack_class=last_metadata.attack_class if last_metadata else attack_name.upper(),
                        execution_time_seconds=round(total_atk_time, 4),
                        parameters=merged_params,
                        epsilon=atk_config_obj.epsilon if (atk_config_obj and atk_config_obj.epsilon is not None) else (last_metadata.epsilon if last_metadata else None),
                    )
                    combined_atk_res = AttackResult(
                        adversarial_examples=combined_adv,
                        metadata=combined_metadata,
                        original_inputs=combined_orig,
                        labels=combined_labels,
                    )

                    after_attack_results_coll[attack_name] = combined_atk_res
                    after_attack_results_dict[attack_name] = {
                        "attack_name": combined_atk_res.metadata.attack_name,
                        "attack_class": combined_atk_res.metadata.attack_class,
                        "execution_time_seconds": combined_atk_res.metadata.execution_time_seconds,
                        "parameters": combined_atk_res.metadata.parameters,
                        "status": "SUCCESS",
                    }

                    # Evaluate Hardened Model on Adversarial Examples
                    adv_targets = combined_atk_res.labels
                    in_mem_loader = InMemoryDatasetLoader(
                        inputs=combined_atk_res.adversarial_examples,
                        targets=adv_targets,
                        dataset_name=f"hardened_adv_{attack_name}",
                        batch_size=batch_size,
                    )
                    adv_evaluator = BaselineEvaluator(
                        adapter=adapter,
                        dataset_loader=in_mem_loader,
                        num_classes=num_classes,
                        model_name=model_name,
                    )
                    adv_eval_res: EvaluationResult = adv_evaluator.evaluate(output_dir=None, show_progress=False)
                    after_adv_evaluations_dict[attack_name] = adv_eval_res

                except Exception as atk_err:
                    from app.attack_engine.exceptions import UnsupportedModelError
                    is_unsupported = isinstance(atk_err, UnsupportedModelError) or "unsupported" in str(atk_err).lower()
                    err_status = "unsupported" if is_unsupported else "failed"
                    after_attack_results_dict[attack_name] = {
                        "attack_name": attack_name,
                        "status": err_status,
                        "error": str(atk_err),
                        "error_type": type(atk_err).__name__,
                        "execution_time_seconds": 0.0,
                        "parameters": attack_configs.get(attack_name.lower(), {}) if (attack_configs and isinstance(attack_configs.get(attack_name.lower()), dict)) else {},
                    }
                    failed_or_unsupported_attacks[attack_name] = {
                        "status": err_status,
                        "error": str(atk_err),
                    }

            # Step 3: Run Vulnerability Analysis on Hardened Model
            after_vulnerability_analysis: Dict[str, Dict[str, Any]] = {}

            if len(after_adv_evaluations_dict) > 0:
                predictions_dict = {}
                for atk_name, adv_res in after_adv_evaluations_dict.items():
                    y_true = adv_res.targets or after_baseline_result.targets
                    y_clean_pred = after_baseline_result.predictions
                    y_adv_pred = adv_res.predictions
                    predictions_dict[atk_name] = {
                        "y_true": y_true,
                        "y_clean_pred": y_clean_pred,
                        "y_adv_pred": y_adv_pred,
                    }

                after_vuln_raw = self.vulnerability_engine.analyze_pipeline(
                    baseline_result=after_baseline_result,
                    attack_results=after_attack_results_coll,
                    adversarial_results=after_adv_evaluations_dict,
                    predictions_dict=predictions_dict,
                )

                for atk_k, atk_v in after_vuln_raw.items():
                    after_vulnerability_analysis[atk_k] = {
                        "assessment": atk_v["assessment"].to_dict() if hasattr(atk_v["assessment"], "to_dict") else atk_v["assessment"],
                        "scoring": atk_v["scoring"].to_dict() if hasattr(atk_v["scoring"], "to_dict") else atk_v["scoring"],
                    }

            # Record failed or unsupported attacks into after_vulnerability_analysis
            for fail_k, fail_v in failed_or_unsupported_attacks.items():
                after_vulnerability_analysis[fail_k] = {
                    "assessment": {
                        "attack_name": fail_k,
                        "status": fail_v["status"],
                        "error": fail_v["error"],
                        "attack_success_rate": None,
                        "clean_accuracy": after_baseline_result.accuracy,
                        "adversarial_accuracy": None,
                        "accuracy_drop": None,
                        "f1_drop": None,
                        "confidence_drop": None,
                        "model_degradation": None,
                        "perturbation": {},
                    },
                    "scoring": {
                        "attack_name": fail_k,
                        "status": fail_v["status"],
                        "error": fail_v["error"],
                        "vulnerability_score": None,
                        "risk_level": "UNKNOWN",
                    },
                }

            # Step 4: Perform BEFORE vs AFTER Comparison
            comparisons: Dict[str, ComparisonResult] = self.comparison_engine.compare_pipeline(
                before_vulnerability_analysis=before_vulnerability_analysis,
                after_vulnerability_analysis=after_vulnerability_analysis,
                thresholds=threshold_obj,
                latency_sensitive=latency_sensitive,
            )

            # Evaluate overall retest status
            if not comparisons:
                overall_improved = True
                overall_status = "COMPLETED"
            elif all(comp.status == "successful" for comp in comparisons.values()):
                overall_improved = True
                overall_status = "accepted"
            elif any(comp.status == "failed_defense" for comp in comparisons.values()):
                overall_improved = False
                overall_status = "failed_defense"
            elif any(comp.status == "attack_failed" for comp in comparisons.values()):
                overall_improved = False
                overall_status = "PARTIAL_SUCCESS"
            elif all(comp.status == "unsupported" for comp in comparisons.values()):
                overall_improved = False
                overall_status = "unsupported"
            elif all(comp.status == "inconclusive" for comp in comparisons.values()):
                overall_improved = False
                overall_status = "inconclusive"
            else:
                overall_improved = all(comp.is_improved for comp in comparisons.values())
                overall_status = "insufficient" if not overall_improved else "accepted"

            # Global threshold evaluation
            overall_thresh_eval = threshold_obj.evaluate(
                before_metrics={
                    "clean_accuracy": before_baseline_dict.get("accuracy"),
                    "latency_ms": before_baseline_dict.get("average_inference_time_ms"),
                },
                after_metrics={
                    "clean_accuracy": after_baseline_dict.get("accuracy"),
                    "latency_ms": after_baseline_dict.get("average_inference_time_ms"),
                },
                latency_sensitive=latency_sensitive,
            )

            exec_time = round(time.time() - start_time, 4)

            return RetestResult(
                hardened_model_name=model_name,
                dataset_name=dataset_loader.dataset_name if hasattr(dataset_loader, "dataset_name") else "dataset",
                num_samples=after_baseline_result.num_samples,
                before_baseline_evaluation=before_baseline_dict,
                after_baseline_evaluation=after_baseline_dict,
                before_attack_results=before_attack_results,
                after_attack_results=after_attack_results_dict,
                before_vulnerability_analysis=before_vulnerability_analysis,
                after_vulnerability_analysis=after_vulnerability_analysis,
                comparisons=comparisons,
                overall_improved=overall_improved,
                timestamp=timestamp,
                execution_time_seconds=exec_time,
                status=overall_status,
                defense_name=defense_name,
                defense_config=defense_config or {},
                defense_metadata=defense_metadata or {},
                thresholds=threshold_obj.to_dict(),
                threshold_evaluation=overall_thresh_eval,
                overall_status=overall_status,
            )

        finally:
            # Restore model training states
            if original_model is not None and hasattr(original_model, "train"):
                original_model.train(orig_training)
            if hasattr(hardened_model, "train"):
                hardened_model.train(hardened_training)
