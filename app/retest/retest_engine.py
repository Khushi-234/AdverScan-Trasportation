"""
RetestEngine for Module 8 (Re-Test & Comparison) in AdverScan.

Coordinates controlled re-evaluation of hardened models using identical
dataset, attack suite, and configuration parameters as used prior to hardening.
Reuses existing AttackEngine, BaselineEvaluator, VulnerabilityEngine, and ComparisonEngine.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

from app.ingestion.adapters.base_adapter import BaseModelAdapter
from app.ingestion.adapters.pytorch_adapter import PyTorchAdapter
from app.evaluation.evaluator import BaselineEvaluator
from app.evaluation.dataset_loader import BaseDatasetLoader
from app.evaluation.results import EvaluationResult
from app.attack_engine.attack_engine import AttackEngine
from app.attack_engine.config import AttackConfig
from app.attack_engine.models import AttackResult, AttackResults
from app.vulnerability_analysis.vulnerability_engine import VulnerabilityEngine
from app.orchestration.dataset_adapter import InMemoryDatasetLoader
from app.retest.comparison import ComparisonEngine
from app.retest.retest_result import RetestResult, ComparisonResult


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
        if hasattr(hardened_model, "hardened_model"):
            hardened_model = hardened_model.hardened_model

        if isinstance(hardened_model, BaseModelAdapter):
            return hardened_model
        elif isinstance(hardened_model, (nn.Module, torch.jit.ScriptModule)):
            return PyTorchAdapter(hardened_model, device=device)
        elif hasattr(hardened_model, "predict") and callable(getattr(hardened_model, "predict")):
            if isinstance(hardened_model, nn.Module):
                return PyTorchAdapter(hardened_model, device=device)
            return hardened_model
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

        Returns:
            RetestResult: Structured DTO containing complete re-test and comparative delta analysis.
        """
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        adapter = self._resolve_adapter(hardened_model, device=device)
        attack_configs = attack_configs or {}
        before_attack_results = before_attack_results or {}

        # Standardize before_baseline_result dict representation
        before_baseline_dict = (
            before_baseline_result.to_dict()
            if hasattr(before_baseline_result, "to_dict")
            else dict(before_baseline_result)
        )

        # Step 1: Baseline Evaluation of Hardened Model
        baseline_evaluator = BaselineEvaluator(
            adapter=adapter,
            dataset_loader=dataset_loader,
            num_classes=num_classes,
            model_name=model_name,
        )
        after_baseline_result: EvaluationResult = baseline_evaluator.evaluate(output_dir=None)
        after_baseline_dict = after_baseline_result.to_dict()

        # Extract sample batch for attack execution
        sample_batch = None
        for batch_pixels, batch_targets, _ in dataset_loader.iterate_batches():
            sample_batch = (batch_pixels, batch_targets)
            break

        if sample_batch is None:
            raise ValueError("Provided dataset_loader produced no batches.")

        inputs, labels = sample_batch

        # Step 2: Execute Attacks on Hardened Model using AttackEngine
        attack_engine = AttackEngine(adapter)
        after_attack_results_coll = AttackResults()
        after_attack_results_dict: Dict[str, Any] = {}
        after_adv_evaluations_dict: Dict[str, EvaluationResult] = {}

        for attack_name in attacks:
            atk_cfg_raw = attack_configs.get(attack_name.lower())
            atk_config_obj = None
            if isinstance(atk_cfg_raw, AttackConfig):
                atk_config_obj = atk_cfg_raw
            elif isinstance(atk_cfg_raw, dict):
                atk_config_obj = AttackConfig(**atk_cfg_raw)

            # Execute attack via AttackEngine
            atk_res: AttackResult = attack_engine.run_attack(
                attack_name=attack_name,
                inputs=inputs,
                labels=labels,
                config=atk_config_obj,
            )
            after_attack_results_coll[attack_name] = atk_res
            after_attack_results_dict[attack_name] = {
                "attack_name": atk_res.metadata.attack_name,
                "attack_class": atk_res.metadata.attack_class,
                "execution_time_seconds": atk_res.metadata.execution_time_seconds,
                "parameters": atk_res.metadata.parameters,
            }

            # Evaluate Hardened Model on Adversarial Examples
            adv_targets = atk_res.labels if atk_res.labels is not None else labels
            in_mem_loader = InMemoryDatasetLoader(
                inputs=atk_res.adversarial_examples,
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
            adv_eval_res: EvaluationResult = adv_evaluator.evaluate(output_dir=None)
            after_adv_evaluations_dict[attack_name] = adv_eval_res

        # Step 3: Run Vulnerability Analysis on Hardened Model
        after_vuln_raw = self.vulnerability_engine.analyze_pipeline(
            baseline_result=after_baseline_result,
            attack_results=after_attack_results_coll,
            adversarial_results=after_adv_evaluations_dict,
        )

        after_vulnerability_analysis: Dict[str, Dict[str, Any]] = {}
        for atk_k, atk_v in after_vuln_raw.items():
            after_vulnerability_analysis[atk_k] = {
                "assessment": atk_v["assessment"].to_dict() if hasattr(atk_v["assessment"], "to_dict") else atk_v["assessment"],
                "scoring": atk_v["scoring"].to_dict() if hasattr(atk_v["scoring"], "to_dict") else atk_v["scoring"],
            }

        # Step 4: Perform BEFORE vs AFTER Comparison
        comparisons: Dict[str, ComparisonResult] = self.comparison_engine.compare_pipeline(
            before_vulnerability_analysis=before_vulnerability_analysis,
            after_vulnerability_analysis=after_vulnerability_analysis,
        )

        overall_improved = all(comp.is_improved for comp in comparisons.values()) if comparisons else True
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
        )
