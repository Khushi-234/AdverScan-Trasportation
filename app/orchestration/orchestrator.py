"""
Central AdverScanOrchestrator coordinator module for Module 8 (Orchestration).
"""

from app.report_generator import ReportWriter
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
from app.attack_engine.models import AttackMetadata, AttackResult, AttackResults
from app.vulnerability_analysis.vulnerability_engine import VulnerabilityEngine
from app.explainability.explainer import XAIExplainer
from app.hardening.hardening_engine import HardeningEngine
from app.report_generator import ReportData, ReportGenerator
from app.utils.reproducibility import ReproducibilityManager
from app.utils.resource_monitor import ResourceMonitor
from app.utils.mlflow_tracker import MLflowTracker
from app.orchestration.dataset_adapter import InMemoryDatasetLoader
from app.orchestration.orchestration_result import OrchestrationResult
from app.orchestration.pipeline_config import PipelineConfig
from app.orchestration.failure_registry import FailureRegistry


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

        # Initialize Resource Monitor (A4) and Failure Registry (A5)
        resource_monitor = ResourceMonitor()
        resource_monitor.start()
        failure_registry = FailureRegistry()

        # Step 0: Reproducibility Initialization & Environment Metadata Collection (A2)
        if config.seed is not None:
            ReproducibilityManager.set_seed(
                seed=config.seed,
                deterministic=getattr(config, "deterministic", False),
            )
        repro_meta = ReproducibilityManager.collect_environment_metadata(
            seed=config.seed,
            deterministic=getattr(config, "deterministic", False),
        )
        identity_meta = ReproducibilityManager.collect_identity_metadata(config=config)
        repro_meta.update(identity_meta)

        start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mode = config.mode.lower()

        # MLflow experiment tracking initialization (A3)
        mlflow_tracker: Optional[MLflowTracker] = None
        mlflow_run_id: Optional[str] = None
        mlflow_exp_name: Optional[str] = None

        if getattr(config, "enable_mlflow", False):
            mlflow_tracker = MLflowTracker()
            mlflow_exp_name = config.mlflow_experiment_name or config.experiment_name or "Experiment"
            mlflow_run_id = mlflow_tracker.start_run(
                experiment_name=mlflow_exp_name,
                tracking_uri=config.mlflow_tracking_uri,
                run_name=f"run_{config.model_name}",
            )

        result = OrchestrationResult(
            status="SUCCESS",
            execution_mode=mode,
            timestamp=timestamp,
            reproducibility_metadata=repro_meta,
            mlflow_run_id=mlflow_run_id,
            mlflow_experiment_name=mlflow_exp_name,
        )

        device_info = f"Device: {config.device.upper() if config.device else 'CPU'}"
        if config.device and config.device.lower() == "cuda" and torch.cuda.is_available():
            device_info += f" ({torch.cuda.get_device_name(0)})"
        print(f"▶ [Pipeline Init] Starting AdverScan Pipeline | Mode: {mode.upper()} | {device_info}")

        def _finish(r: OrchestrationResult) -> OrchestrationResult:
            r.execution_time_seconds = round(time.time() - start_time, 4)
            try:
                r.resource_summary = resource_monitor.stop()
            except Exception:
                pass
            try:
                r.failure_records = failure_registry.to_dict()
            except Exception:
                pass
            if mlflow_tracker and mlflow_tracker.active_run_id:
                try:
                    mlflow_tracker.log_orchestration_run(
                        result=r,
                        config=config,
                        output_dir=config.output_dir,
                    )
                    mlflow_tracker.end_run(
                        status="FINISHED" if r.status != "FAILED" else "FAILED"
                    )
                except Exception:
                    pass
            return r

        # Step 1: M1 Model Ingestion
        step_start = time.time()
        print("  ⏳ [Step 1/9] M1 Model Ingestion...", end="", flush=True)
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
            resource_monitor.record_stage("M1_ingestion")
            print(f" Done ({time.time() - step_start:.2f}s)")
        except Exception as e:
            print(f" Failed ({time.time() - step_start:.2f}s)")
            result.status = "FAILED"
            failure_registry.register_exception(
                e, module="M1_ingestion", operation="ingest_model", recoverable=False
            )
            result.errors.append({
                "module": "M1_ingestion",
                "error_type": type(e).__name__,
                "message": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            return _finish(result)

        # Step 2: M2 Baseline Evaluation
        step_start = time.time()
        print("\n  ⏳ [Step 2/9] M2 Baseline Clean Evaluation...")
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
            baseline_result: EvaluationResult = baseline_evaluator.evaluate(output_dir=None, show_progress=True)
            result.baseline_evaluation = baseline_result.to_dict()
            resource_monitor.record_stage("M2_baseline")
            print(f"  ✔ M2 Clean Evaluation Completed ({time.time() - step_start:.2f}s) — Accuracy: {baseline_result.accuracy*100:.2f}%")
        except Exception as e:
            print(f"  ❌ M2 Clean Evaluation Failed ({time.time() - step_start:.2f}s)")
            result.status = "FAILED"
            failure_registry.register_exception(
                e, module="M2_baseline_evaluation", operation="evaluate_baseline", recoverable=False
            )
            result.errors.append({
                "module": "M2_baseline_evaluation",
                "error_type": type(e).__name__,
                "message": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            return _finish(result)

        if mode == "baseline_only":
            return _finish(result)

        # Collect dataset loader batches, adhering to sample_count if set (A1.3)
        dataset_batches = []
        samples_collected = 0
        for batch_pixels, batch_targets, _ in dataset_loader.iterate_batches():
            if config.sample_count is not None and samples_collected >= config.sample_count:
                break
            if config.sample_count is not None and samples_collected + len(batch_targets) > config.sample_count:
                limit = config.sample_count - samples_collected
                batch_pixels = batch_pixels[:limit]
                batch_targets = batch_targets[:limit]
            dataset_batches.append((batch_pixels, batch_targets))
            samples_collected += len(batch_targets)
            if config.sample_count is not None and samples_collected >= config.sample_count:
                break

        if len(dataset_batches) == 0:
            result.status = "FAILED"
            failure_registry.register(
                module="dataset_loader",
                operation="iterate_batches",
                error_type="ValueError",
                message="Dataset loader produced no batches.",
                recoverable=False,
            )
            result.errors.append({
                "module": "dataset_loader",
                "error_type": "ValueError",
                "message": "Dataset loader produced no batches.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            return _finish(result)

        first_batch_inputs, first_batch_labels = dataset_batches[0]

        # Step 3: M3 Attack Engine Execution
        step_start = time.time()
        print(f"\n  ⏳ [Step 3/9] M3 Attack Engine Execution ({', '.join(config.attacks).upper()})...")
        attack_engine = AttackEngine(adapter)
        attack_results_coll = AttackResults()
        adv_evaluations_dict: Dict[str, EvaluationResult] = {}

        for attack_name in config.attacks:
            atk_start = time.time()
            print(f"    ▶ Executing attack '{attack_name.upper()}'...", end="", flush=True)
            atk_cfg_raw = config.attack_configs.get(attack_name.lower())
            atk_config_obj = None
            if isinstance(atk_cfg_raw, AttackConfig):
                atk_config_obj = atk_cfg_raw
            elif isinstance(atk_cfg_raw, dict):
                atk_config_obj = AttackConfig(**atk_cfg_raw)

            batch_orig_list = []
            batch_adv_list = []
            batch_labels_list = []
            total_atk_time = 0.0
            last_metadata = None

            try:
                for b_pixels, b_targets in dataset_batches:
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

                # Combine per-batch results across the full dataset
                if isinstance(batch_adv_list[0], torch.Tensor):
                    combined_orig = torch.cat(batch_orig_list, dim=0)
                    combined_adv = torch.cat(batch_adv_list, dim=0)
                    combined_labels = torch.cat(batch_labels_list, dim=0)
                else:
                    combined_orig = np.concatenate(batch_orig_list, axis=0)
                    combined_adv = np.concatenate(batch_adv_list, axis=0)
                    combined_labels = np.concatenate(batch_labels_list, axis=0)

                combined_metadata = AttackMetadata(
                    attack_name=last_metadata.attack_name if last_metadata else attack_name,
                    attack_class=last_metadata.attack_class if last_metadata else attack_name.upper(),
                    execution_time_seconds=round(total_atk_time, 4),
                    parameters=last_metadata.parameters if last_metadata else {},
                    epsilon=last_metadata.epsilon if last_metadata else None,
                )
                combined_atk_res = AttackResult(
                    adversarial_examples=combined_adv,
                    metadata=combined_metadata,
                    original_inputs=combined_orig,
                    labels=combined_labels,
                )

                attack_results_coll[attack_name] = combined_atk_res
                result.attack_results[attack_name] = {
                    "attack_name": combined_atk_res.metadata.attack_name,
                    "attack_class": combined_atk_res.metadata.attack_class,
                    "execution_time_seconds": combined_atk_res.metadata.execution_time_seconds,
                    "parameters": combined_atk_res.metadata.parameters,
                }
                print(f" Done ({time.time() - atk_start:.2f}s)")
            except Exception as e:
                print(f" Failed ({time.time() - atk_start:.2f}s)")
                result.status = "PARTIAL_SUCCESS"
                failure_registry.register_exception(
                    e, module="M3_attack_engine", operation=f"run_attack_{attack_name}", attack_name=attack_name, recoverable=True
                )
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
                adv_targets = combined_atk_res.labels
                in_mem_loader = InMemoryDatasetLoader(
                    inputs=combined_atk_res.adversarial_examples,
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
                adv_eval_res: EvaluationResult = adv_evaluator.evaluate(output_dir=None, show_progress=False)
                adv_evaluations_dict[attack_name] = adv_eval_res
                result.adversarial_evaluations[attack_name] = adv_eval_res.to_dict()
            except Exception as e:
                result.status = "PARTIAL_SUCCESS"
                failure_registry.register_exception(
                    e, module="M2_adversarial_evaluation", operation=f"eval_adv_{attack_name}", attack_name=attack_name, recoverable=True
                )
                result.errors.append({
                    "module": "M2_adversarial_evaluation",
                    "attack_name": attack_name,
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

        resource_monitor.record_stage("M3_attack_engine")

        if len(attack_results_coll) == 0:
            result.status = "FAILED" if result.status != "PARTIAL_SUCCESS" else "PARTIAL_SUCCESS"
            return _finish(result)

        # Step 5: M5 Vulnerability Analysis
        step_start = time.time()
        print("  ⏳ [Step 5/9] M5 Vulnerability Analysis...", end="", flush=True)
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
            print(f" Done ({time.time() - step_start:.2f}s)")
        except Exception as e:
            print(f" Failed ({time.time() - step_start:.2f}s)")
            result.status = "PARTIAL_SUCCESS"
            failure_registry.register_exception(
                e, module="M5_vulnerability_analysis", operation="analyze_pipeline", recoverable=True
            )
            result.errors.append({
                "module": "M5_vulnerability_analysis",
                "error_type": type(e).__name__,
                "message": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

        if mode == "attack_assessment":
            return _finish(result)

        # Step 6: M6 XAI Explainability (Optional)
        if config.enable_xai:
            step_start = time.time()
            print(f"\n  ⏳ [Step 6/9] M6 XAI Explainability ({', '.join(config.xai_techniques)})...")
            try:
                xai_explainer = XAIExplainer()
                for atk_name, atk_res in attack_results_coll.items():
                    vuln_info = result.vulnerability_analysis.get(atk_name, {})
                    assess_res = vuln_info.get("assessment")
                    for tech in config.xai_techniques:
                        xai_start = time.time()
                        print(f"    ▶ Computing {tech.upper()} explanation for attack '{atk_name.upper()}'...", end="", flush=True)
                        exp_res = xai_explainer.explain_attack_result(
                            model=adapter,
                            attack_result=atk_res,
                            assessment_result=assess_res,
                            technique=tech,
                        )
                        result.xai_results[f"{atk_name}_{tech}"] = exp_res.to_dict()
                        print(f" Done ({time.time() - xai_start:.2f}s)")
            except Exception as e:
                print(f"  ❌ M6 XAI Explainability Error ({time.time() - step_start:.2f}s)")
                result.status = "PARTIAL_SUCCESS"
                failure_registry.register_exception(
                    e, module="M6_explainability", operation="explain_attack_result", recoverable=True
                )
                result.errors.append({
                    "module": "M6_explainability",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

        # Step 7: M7 Hardening Engine (Optional)
        if config.enable_hardening:
            step_start = time.time()
            print(f"\n  ⏳ [Step 7/9] M7 Hardening Engine (Defense: '{config.defense}')...", end="", flush=True)
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
                    inputs=first_batch_inputs,
                    labels=first_batch_labels,
                    attack_name=first_atk,
                    risk_level=risk_lvl,
                    vulnerability_score=score_val,
                    defense_config=config.defense_config,
                )
                result.hardening_results = hard_res.to_dict()
                resource_monitor.record_stage("M7_hardening")
                print(f" Done ({time.time() - step_start:.2f}s)")
            except Exception as e:
                print(f" Failed ({time.time() - step_start:.2f}s)")
                result.status = "PARTIAL_SUCCESS"
                failure_registry.register_exception(
                    e, module="M7_hardening", operation=f"harden_{config.defense}", defense_name=config.defense, recoverable=True
                )
                result.errors.append({
                    "module": "M7_hardening",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

        # Step 8: M8 Re-Test & Comparison Engine (Optional / Post-Hardening)
        retest_res_obj = None
        if config.enable_hardening and config.enable_retest and result.hardening_results:
            step_start = time.time()
            print("\n  ⏳ [Step 8/9] M8 Re-Test & Comparison Engine...")
            try:
                from app.retest.retest_engine import RetestEngine
                retest_engine = RetestEngine()
                hardened_model_obj = result.hardening_results.get("hardened_model") or adapter.get_model()
                retest_res_obj = retest_engine.retest(
                    hardened_model=hardened_model_obj,
                    dataset_loader=dataset_loader,
                    attacks=config.attacks,
                    before_baseline_result=baseline_result,
                    before_vulnerability_analysis=result.vulnerability_analysis,
                    attack_configs=config.attack_configs,
                    before_attack_results=result.attack_results,
                    num_classes=config.num_classes,
                    model_name=config.model_name,
                    device=config.device,
                    batch_size=config.batch_size,
                )
                result.retest_results = retest_res_obj.to_dict()
                print(f"  ✔ M8 Re-Test Completed ({time.time() - step_start:.2f}s)")
            except Exception as e:
                print(f"  ❌ M8 Re-Test Failed ({time.time() - step_start:.2f}s)")
                result.status = "PARTIAL_SUCCESS"
                failure_registry.register_exception(
                    e, module="M8_retest", operation="retest", recoverable=True
                )
                result.errors.append({
                    "module": "M8_retest",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

        # Step 9: M9 Security Report Generator
        if config.enable_report:
            step_start = time.time()
            print("\n  ⏳ [Step 9/9] M9 Security Report Generator...", end="", flush=True)
            try:
                report_data = ReportData.from_orchestration_and_retest(
                    orchestration_result=result,
                    retest_result=retest_res_obj,
                )
                report_gen = ReportGenerator()
                report_res = report_gen.generate(report_data)

                writer = ReportWriter()
                written_files = writer.write(report_res, formats=["md", "json", "csv"])
                
                result.report_result = report_res.to_dict()

                if config.output_dir:
                    report_res.save_json(f"{config.output_dir}/security_report.json")
                    report_res.save_text(f"{config.output_dir}/security_report.txt")
                print(f" Done ({time.time() - step_start:.2f}s)")
            except Exception as e:
                print(f" Failed ({time.time() - step_start:.2f}s)")
                result.status = "PARTIAL_SUCCESS"
                failure_registry.register_exception(
                    e, module="M9_report_generator", operation="generate_report", recoverable=True
                )
                result.errors.append({
                    "module": "M9_report_generator",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

        return _finish(result)


