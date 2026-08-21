"""
Central AdverScanOrchestrator coordinator module for Module 8 (Orchestration).
"""

from datetime import datetime
import time
from typing import Any, Dict, List, Optional
import torch

from app.ingestion.pipeline import ingest_model
from app.evaluation.evaluator import BaselineEvaluator
from app.evaluation.dataset_loader import GTSRBDatasetLoader
from app.evaluation.results import EvaluationResult
from app.attack_engine.attack_engine import AttackEngine
from app.attack_engine.config import AttackConfig
from app.attack_engine.models import AttackResult, AttackResults
from app.vulnerability_analysis.vulnerability_engine import VulnerabilityEngine
from app.explainability.explainer import XAIExplainer
from app.hardening.hardening_engine import HardeningEngine
from app.orchestration.dataset_adapter import InMemoryDatasetLoader
from app.orchestration.orchestration_result import OrchestrationResult
from app.orchestration.pipeline_config import PipelineConfig


class AdverScanOrchestrator:
    """
    Central orchestration and integration coordinator for AdverScan.
    Coordinates M1 → M2 → M3 → M2(Adv) → M5 → M6 / M7 workflow pipeline.
    """

    def run(self, config: PipelineConfig) -> OrchestrationResult:
        """
        Execute the configured AdverScan workflow pipeline.

        Args:
            config: PipelineConfig instance detailing execution parameters.

        Returns:
            OrchestrationResult containing aggregated status, timing, and module DTOs.
        """
        config.validate()
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mode = config.mode.lower()

        result = OrchestrationResult(
            status="SUCCESS",
            execution_mode=mode,
            timestamp=timestamp,
        )

        # Step 1: M1 Model Ingestion
        try:
            adapter, metadata = ingest_model(
                model_path=config.model_path,
                sample_input=config.sample_input,
                device=config.device,
                model_class=config.model_class,
                model_name=config.model_name,
                task_type=config.task_type,
            )
            result.model_metadata = metadata.to_dict() if hasattr(metadata, "to_dict") else metadata
        except Exception as e:
            result.status = "FAILED"
            result.errors.append({
                "module": "M1_ingestion",
                "error_type": type(e).__name__,
                "message": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            result.execution_time_seconds = round(time.time() - start_time, 4)
            return result

        # Step 2: M2 Baseline Evaluation
        try:
            if config.custom_dataset_loader is not None:
                dataset_loader = config.custom_dataset_loader
            else:
                dataset_loader = GTSRBDatasetLoader(
                    dataset_name=config.dataset_name,
                    processor_name=config.processor_name,
                    split=config.split,
                    batch_size=config.batch_size,
                )

            baseline_evaluator = BaselineEvaluator(
                adapter=adapter,
                dataset_loader=dataset_loader,
                num_classes=config.num_classes,
                model_name=config.model_name,
            )
            baseline_result: EvaluationResult = baseline_evaluator.evaluate(output_dir=None)
            result.baseline_evaluation = baseline_result.to_dict()
        except Exception as e:
            result.status = "FAILED"
            result.errors.append({
                "module": "M2_baseline_evaluation",
                "error_type": type(e).__name__,
                "message": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            result.execution_time_seconds = round(time.time() - start_time, 4)
            return result

        if mode == "baseline_only":
            result.execution_time_seconds = round(time.time() - start_time, 4)
            return result

        # Extract sample batch for attack execution
        sample_batch = None
        for batch_pixels, batch_targets, _ in dataset_loader.iterate_batches():
            sample_batch = (batch_pixels, batch_targets)
            break

        if sample_batch is None:
            result.status = "FAILED"
            result.errors.append({
                "module": "dataset_loader",
                "error_type": "ValueError",
                "message": "Dataset loader produced no batches.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            result.execution_time_seconds = round(time.time() - start_time, 4)
            return result

        inputs, labels = sample_batch

        # Step 3: M3 Attack Engine Execution
        attack_engine = AttackEngine(adapter)
        attack_results_coll = AttackResults()
        adv_evaluations_dict: Dict[str, EvaluationResult] = {}

        for attack_name in config.attacks:
            atk_cfg_raw = config.attack_configs.get(attack_name.lower())
            atk_config_obj = None
            if isinstance(atk_cfg_raw, AttackConfig):
                atk_config_obj = atk_cfg_raw
            elif isinstance(atk_cfg_raw, dict):
                atk_config_obj = AttackConfig(**atk_cfg_raw)

            try:
                atk_res: AttackResult = attack_engine.run_attack(
                    attack_name=attack_name,
                    inputs=inputs,
                    labels=labels,
                    config=atk_config_obj,
                )
                attack_results_coll[attack_name] = atk_res
                result.attack_results[attack_name] = {
                    "attack_name": atk_res.metadata.attack_name,
                    "attack_class": atk_res.metadata.attack_class,
                    "execution_time_seconds": atk_res.metadata.execution_time_seconds,
                    "parameters": atk_res.metadata.parameters,
                }
            except Exception as e:
                result.status = "PARTIAL_SUCCESS"
                result.errors.append({
                    "module": "M3_attack_engine",
                    "attack_name": attack_name,
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
                continue

            # Step 4: M2 Adversarial Evaluation via InMemoryDatasetLoader helper
            try:
                adv_targets = atk_res.labels if atk_res.labels is not None else labels
                in_mem_loader = InMemoryDatasetLoader(
                    inputs=atk_res.adversarial_examples,
                    targets=adv_targets,
                    dataset_name=f"adv_{attack_name}",
                    batch_size=config.batch_size,
                )
                adv_evaluator = BaselineEvaluator(
                    adapter=adapter,
                    dataset_loader=in_mem_loader,
                    num_classes=config.num_classes,
                    model_name=config.model_name,
                )
                adv_eval_res: EvaluationResult = adv_evaluator.evaluate(output_dir=None)
                adv_evaluations_dict[attack_name] = adv_eval_res
                result.adversarial_evaluations[attack_name] = adv_eval_res.to_dict()
            except Exception as e:
                result.status = "PARTIAL_SUCCESS"
                result.errors.append({
                    "module": "M2_adversarial_evaluation",
                    "attack_name": attack_name,
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

        if len(attack_results_coll) == 0:
            result.status = "FAILED" if result.status != "PARTIAL_SUCCESS" else "PARTIAL_SUCCESS"
            result.execution_time_seconds = round(time.time() - start_time, 4)
            return result

        # Step 5: M5 Vulnerability Analysis
        try:
            vuln_engine = VulnerabilityEngine()
            vuln_out = vuln_engine.analyze_pipeline(
                baseline_result=baseline_result,
                attack_results=attack_results_coll,
                adversarial_results=adv_evaluations_dict,
            )
            for atk_k, atk_v in vuln_out.items():
                result.vulnerability_analysis[atk_k] = {
                    "assessment": atk_v["assessment"].to_dict(),
                    "scoring": atk_v["scoring"].to_dict(),
                }
        except Exception as e:
            result.status = "PARTIAL_SUCCESS"
            result.errors.append({
                "module": "M5_vulnerability_analysis",
                "error_type": type(e).__name__,
                "message": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

        if mode == "attack_assessment":
            result.execution_time_seconds = round(time.time() - start_time, 4)
            return result

        # Step 6: M6 XAI Explainability (Optional)
        if config.enable_xai:
            try:
                xai_explainer = XAIExplainer()
                for atk_name, atk_res in attack_results_coll.items():
                    vuln_info = result.vulnerability_analysis.get(atk_name, {})
                    assess_res = vuln_info.get("assessment")
                    for tech in config.xai_techniques:
                        exp_res = xai_explainer.explain_attack_result(
                            model=adapter,
                            attack_result=atk_res,
                            assessment_result=assess_res,
                            technique=tech,
                        )
                        result.xai_results[f"{atk_name}_{tech}"] = exp_res.to_dict()
            except Exception as e:
                result.status = "PARTIAL_SUCCESS"
                result.errors.append({
                    "module": "M6_explainability",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

        # Step 7: M7 Hardening Engine (Optional)
        if config.enable_hardening:
            try:
                hardening_engine = HardeningEngine()
                raw_model = adapter.get_model()

                first_atk = list(attack_results_coll.keys())[0] if len(attack_results_coll) > 0 else None
                score_val = None
                risk_lvl = None
                if first_atk and first_atk in result.vulnerability_analysis:
                    score_info = result.vulnerability_analysis[first_atk].get("scoring", {})
                    score_val = score_info.get("vulnerability_score")
                    risk_lvl = score_info.get("risk_level")

                hard_res = hardening_engine.harden(
                    model=raw_model,
                    defense=config.defense,
                    inputs=inputs,
                    labels=labels,
                    attack_name=first_atk,
                    risk_level=risk_lvl,
                    vulnerability_score=score_val,
                    defense_config=config.defense_config,
                )
                result.hardening_results = hard_res.to_dict()
            except Exception as e:
                result.status = "PARTIAL_SUCCESS"
                result.errors.append({
                    "module": "M7_hardening",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

        result.execution_time_seconds = round(time.time() - start_time, 4)
        return result
