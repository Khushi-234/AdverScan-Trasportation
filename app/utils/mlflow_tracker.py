"""
MLflow tracking thin adapter for Module 8 (Orchestration).
Provides safe, failure-isolated logging of AdverScan experiments, parameters,
reproducibility metadata, resource summaries, failure records, metrics, tags,
and artifacts to MLflow.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("AdverScan.MLflowTracker")


def is_mlflow_available() -> bool:
    """Check whether MLflow is installed and importable."""
    try:
        import mlflow
        return True
    except ImportError:
        return False


class MLflowTracker:
    """
    Thin adapter over MLflow providing failure-isolated tracking operations
    for AdverScan experiments.
    """

    def __init__(self) -> None:
        self._active_run_id: Optional[str] = None
        self._experiment_name: Optional[str] = None

    @property
    def active_run_id(self) -> Optional[str]:
        """Return currently active MLflow run_id, if any."""
        return self._active_run_id

    @property
    def experiment_name(self) -> Optional[str]:
        """Return configured experiment name, if any."""
        return self._experiment_name

    def start_run(
        self,
        experiment_name: str = "Experiment",
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Start an MLflow run under the specified experiment.

        Returns:
            MLflow run_id string if successful, None if MLflow is unavailable or fails.
        """
        if not is_mlflow_available():
            logger.warning("MLflow is not installed. Experiment tracking disabled.")
            return None

        try:
            import mlflow

            # Ensure MLflow 3.x allows file store backend without throwing maintenance mode error
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)

            exp_name = experiment_name or "Experiment"
            self._experiment_name = exp_name
            mlflow.set_experiment(exp_name)

            run = mlflow.start_run(run_name=run_name, tags=tags)
            self._active_run_id = run.info.run_id
            return self._active_run_id
        except Exception as exc:
            logger.warning(f"Failed to start MLflow run: {exc}")
            self._active_run_id = None
            return None

    def log_orchestration_run(
        self,
        result: Any,
        config: Any,
        output_dir: Optional[str] = None,
    ) -> bool:
        """
        Log concise parameters, scalar metrics, categorical tags, and structured
        JSON/report artifacts from an OrchestrationResult and PipelineConfig.

        Returns:
            True if logging succeeded, False otherwise.
        """
        if not is_mlflow_available() or not self._active_run_id:
            return False

        try:
            import mlflow

            # 1. Log A1 Configuration Parameters
            params: Dict[str, Any] = {}
            if hasattr(config, "to_dict"):
                cfg_dict = config.to_dict()
                for key in [
                    "experiment_id",
                    "experiment_name",
                    "description",
                    "model_name",
                    "model_path",
                    "task_type",
                    "num_classes",
                    "dataset_name",
                    "processor_name",
                    "split",
                    "batch_size",
                    "sample_count",
                    "device",
                    "seed",
                    "deterministic",
                    "mode",
                    "enable_xai",
                    "enable_hardening",
                    "defense",
                    "enable_retest",
                    "enable_report",
                ]:
                    if key in cfg_dict and cfg_dict[key] is not None:
                        params[key] = str(cfg_dict[key])

                if "attacks" in cfg_dict:
                    params["attacks"] = ",".join(cfg_dict["attacks"]) if isinstance(cfg_dict["attacks"], list) else str(cfg_dict["attacks"])
                if "xai_techniques" in cfg_dict:
                    params["xai_techniques"] = ",".join(cfg_dict["xai_techniques"]) if isinstance(cfg_dict["xai_techniques"], list) else str(cfg_dict["xai_techniques"])

            if hasattr(config, "get_configuration_hash"):
                params["configuration_hash"] = config.get_configuration_hash()

            # A2 Reproducibility parameters
            if hasattr(result, "reproducibility_metadata") and isinstance(result.reproducibility_metadata, dict):
                repro = result.reproducibility_metadata
                for k in ["python_version", "torch_version", "cuda_available", "cuda_version", "device_name"]:
                    if k in repro and repro[k] is not None:
                        params[f"repro_{k}"] = str(repro[k])

            # Batch log parameters (MLflow limits parameter value length)
            cleaned_params = {k: str(v)[:250] for k, v in params.items()}
            mlflow.log_params(cleaned_params)

            # 2. Log Categorical Tags
            tags: Dict[str, str] = {
                "project": "AdverScan",
                "experiment_name": str(getattr(config, "experiment_name", "Experiment")),
                "model_name": str(getattr(config, "model_name", "TargetModel")),
                "dataset_name": str(getattr(config, "dataset_name", "Dataset")),
                "device": str(getattr(config, "device", "cpu")),
                "status": str(getattr(result, "status", "UNKNOWN")),
                "hardening_enabled": str(getattr(config, "enable_hardening", False)),
                "defense": str(getattr(config, "defense", "none")),
                "xai_enabled": str(getattr(config, "enable_xai", False)),
                "deterministic": str(getattr(config, "deterministic", False)),
            }
            if hasattr(config, "attacks") and isinstance(config.attacks, list):
                tags["attack_count"] = str(len(config.attacks))

            failure_recs = getattr(result, "failure_records", []) or []
            tags["failure_status"] = "NO_FAILURES" if len(failure_recs) == 0 else "HAS_FAILURES"

            mlflow.set_tags(tags)

            # 3. Log Concise Scalar Metrics
            metrics: Dict[str, float] = {}

            # Execution time & status metrics
            metrics["execution_time_seconds"] = float(getattr(result, "execution_time_seconds", 0.0))
            metrics["failure_count"] = float(len(failure_recs))

            # A4 Resource metrics
            if hasattr(result, "resource_summary") and isinstance(result.resource_summary, dict):
                res_sum = result.resource_summary
                for metric_key, val in [
                    ("cpu_percent", res_sum.get("cpu_percent")),
                    ("ram_used_gb", res_sum.get("ram_used_gb")),
                    ("process_ram_used_gb", res_sum.get("process_ram_used_gb")),
                    ("gpu_memory_used_mb", res_sum.get("gpu_memory_used_mb")),
                    ("gpu_peak_memory_mb", res_sum.get("gpu_peak_memory_mb")),
                ]:
                    if val is not None and isinstance(val, (int, float)):
                        metrics[metric_key] = float(val)

            # M2 Baseline Evaluation metrics
            if hasattr(result, "baseline_evaluation") and isinstance(result.baseline_evaluation, dict):
                base_eval = result.baseline_evaluation
                for m_name in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "average_confidence", "average_entropy"]:
                    if m_name in base_eval and isinstance(base_eval[m_name], (int, float)):
                        metrics[f"baseline_{m_name}"] = float(base_eval[m_name])

            # M3 & M2 Adversarial Evaluation metrics
            if hasattr(result, "adversarial_evaluations") and isinstance(result.adversarial_evaluations, dict):
                for atk_name, adv_eval in result.adversarial_evaluations.items():
                    if isinstance(adv_eval, dict):
                        for m_name in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
                            if m_name in adv_eval and isinstance(adv_eval[m_name], (int, float)):
                                metrics[f"adv_{atk_name}_{m_name}"] = float(adv_eval[m_name])

            # M5 Vulnerability Analysis metrics
            if hasattr(result, "vulnerability_analysis") and isinstance(result.vulnerability_analysis, dict):
                for atk_name, vuln_data in result.vulnerability_analysis.items():
                    if isinstance(vuln_data, dict):
                        scoring = vuln_data.get("scoring", {})
                        if isinstance(scoring, dict) and "vulnerability_score" in scoring:
                            v_score = scoring["vulnerability_score"]
                            if isinstance(v_score, (int, float)):
                                metrics[f"vuln_score_{atk_name}"] = float(v_score)

            # M8 Retest metrics
            if hasattr(result, "retest_results") and isinstance(result.retest_results, dict):
                ret_data = result.retest_results
                if "robustness_improvement" in ret_data and isinstance(ret_data["robustness_improvement"], (int, float)):
                    metrics["robustness_improvement"] = float(ret_data["robustness_improvement"])

            mlflow.log_metrics(metrics)

            # 4. Log Structured Artifacts (JSON & Reports)
            target_dir = Path(output_dir) if output_dir else Path("results/orchestration")
            target_dir.mkdir(parents=True, exist_ok=True)

            # Helper to safely save and log a JSON artifact
            def _log_json_artifact(filename: str, data: Any) -> None:
                if data is not None:
                    filepath = target_dir / filename
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    mlflow.log_artifact(str(filepath))

            if hasattr(config, "to_dict"):
                _log_json_artifact("pipeline_config.json", config.to_dict())

            if hasattr(result, "reproducibility_metadata"):
                _log_json_artifact("reproducibility_metadata.json", result.reproducibility_metadata)

            if hasattr(result, "resource_summary"):
                _log_json_artifact("resource_summary.json", result.resource_summary)

            if hasattr(result, "failure_records"):
                _log_json_artifact("failure_records.json", result.failure_records)

            if hasattr(result, "to_dict"):
                _log_json_artifact("orchestration_result.json", result.to_dict())

            # Log generated report files if present in output_dir
            if output_dir and os.path.exists(output_dir):
                for report_file in ["security_report.json", "security_report.txt", "security_report.md"]:
                    rp = Path(output_dir) / report_file
                    if rp.exists() and rp.is_file():
                        mlflow.log_artifact(str(rp))

            return True
        except Exception as exc:
            logger.warning(f"MLflow logging error: {exc}")
            return False

    def end_run(self, status: str = "FINISHED") -> bool:
        """
        Safely conclude the current active MLflow run.

        Returns:
            True if run was ended cleanly, False otherwise.
        """
        if not is_mlflow_available() or not self._active_run_id:
            self._active_run_id = None
            return False

        try:
            import mlflow

            mlflow.end_run(status=status)
            self._active_run_id = None
            return True
        except Exception as exc:
            logger.warning(f"Failed to end MLflow run: {exc}")
            self._active_run_id = None
            return False
