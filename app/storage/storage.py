"""
Centralized Storage Interface and Artifact Manager for AdverScan.

Provides:
1. Pure dictionary-based CRUD interface (prevents raw ORM session leaks).
2. ArtifactManager for safely offloading large tensors, adversarial samples,
   and XAI attribution maps to the filesystem (preventing SQLite database bloat).
3. Thread-safe execution using scoped SQLite sessions and WAL mode.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np

from app.storage.db import get_db_path, get_db_session, init_db, DEFAULT_STORAGE_DIR
from app.storage.models import (
    AssessmentRecord,
    AttackExecutionRecord,
    DatasetRecord,
    DefenseExecutionRecord,
    DefenseRecommendationRecord,
    ModelRecord,
    ReportRecord,
    VulnerabilityFindingRecord,
    XAIResultRecord,
)

# Base artifact storage directory
DEFAULT_ARTIFACT_DIR = DEFAULT_STORAGE_DIR / "artifacts"


# ==============================================================================
# ARTIFACT MANAGER (Filesystem Offloading for Large Tensors & Visualizations)
# ==============================================================================

class ArtifactManager:
    """
    Manages filesystem serialization and retrieval of large evaluation artifacts
    (adversarial tensors, heatmaps, detailed XAI attributions, model checkpoints).
    Keeps the SQLite database lightweight and responsive.
    """

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        self.base_dir = Path(base_dir).resolve() if base_dir else DEFAULT_ARTIFACT_DIR.resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("attacks", "defenses", "xai", "reports"):
            (self.base_dir / sub).mkdir(parents=True, exist_ok=True)

    def save_artifact(
        self,
        data: Any,
        scan_id: str,
        category: str,
        filename_prefix: str,
    ) -> Optional[str]:
        """
        Serialize and save a data artifact to the filesystem.

        Args:
            data: NumPy ndarray, PyTorch Tensor, dictionary, or bytes.
            scan_id: Assessment scan identifier.
            category: 'attacks', 'defenses', 'xai', or 'reports'.
            filename_prefix: Base name for the artifact.

        Returns:
            Relative path string to the saved file (relative to self.base_dir),
            or None if data is None.
        """
        if data is None:
            return None

        clean_cat = category.strip().lower()
        cat_dir = self.base_dir / clean_cat / scan_id
        cat_dir.mkdir(parents=True, exist_ok=True)

        # 1. PyTorch Tensor / NumPy Array
        if hasattr(data, "cpu") and hasattr(data, "numpy"):
            data_np = data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            data_np = data
        else:
            data_np = None

        if data_np is not None:
            file_name = f"{filename_prefix}.npz"
            file_path = cat_dir / file_name
            np.savez_compressed(file_path, data=data_np)
            return str(file_path.relative_to(self.base_dir))

        # 2. Dictionary / List
        if isinstance(data, (dict, list)):
            file_name = f"{filename_prefix}.json"
            file_path = cat_dir / file_name
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)
                return str(file_path.relative_to(self.base_dir))
            except Exception:
                pass

        # 3. String / Text / Markdown
        if isinstance(data, str):
            file_name = f"{filename_prefix}.txt"
            file_path = cat_dir / file_name
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(data)
            return str(file_path.relative_to(self.base_dir))

        # 4. Raw Bytes
        if isinstance(data, (bytes, bytearray)):
            file_name = f"{filename_prefix}.bin"
            file_path = cat_dir / file_name
            with open(file_path, "wb") as f:
                f.write(data)
            return str(file_path.relative_to(self.base_dir))

        return None

    def load_artifact(self, relative_path: str) -> Any:
        """
        Load a stored artifact from its relative path.
        """
        if not relative_path:
            return None
        full_path = (self.base_dir / relative_path).resolve()
        if not full_path.is_file():
            # Try absolute path fallback
            if Path(relative_path).is_file():
                full_path = Path(relative_path).resolve()
            else:
                return None

        if full_path.suffix == ".npz":
            with np.load(full_path) as loaded:
                return loaded["data"] if "data" in loaded else dict(loaded)
        elif full_path.suffix == ".json":
            with open(full_path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif full_path.suffix in (".txt", ".md", ".html"):
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        elif full_path.suffix in (".bin", ".pdf"):
            with open(full_path, "rb") as f:
                return f.read()

        return None


# ==============================================================================
# CENTRALIZED STORAGE REPOSITORY
# ==============================================================================

class StorageRepository:
    """
    Central database repository for AdverScan assessments, model registries,
    attack/defense telemetry, XAI explanations, and reports.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        artifact_dir: Optional[Union[str, Path]] = None,
        auto_init: bool = True,
    ):
        self.db_path = get_db_path(db_path)
        self.artifacts = ArtifactManager(base_dir=artifact_dir)
        if auto_init:
            init_db(self.db_path)

    # --------------------------------------------------------------------------
    # 1. Assessment / Scan Lifecycle
    # --------------------------------------------------------------------------

    def create_assessment(
        self,
        scan_id: Optional[str] = None,
        assessment_name: Optional[str] = None,
        model_info: Optional[Dict[str, Any]] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
        configuration: Optional[Dict[str, Any]] = None,
        parent_scan_id: Optional[str] = None,
        baseline_scan_id: Optional[str] = None,
        modality: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Initialize and persist a new assessment run in 'running' state.
        Generates a unique scan_id (SCAN-XXXXXXXX) if omitted.
        """
        final_scan_id = scan_id or f"SCAN-{uuid.uuid4().hex[:8].upper()}"
        resolved_name = (
            assessment_name
            or f"Assessment {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )

        with get_db_session(self.db_path) as session:
            # 1. Upsert model record if provided
            m_id = None
            m_name = None
            if model_info:
                m_id = str(model_info.get("model_id") or model_info.get("model_identifier") or "custom-model")
                m_name = str(model_info.get("model_name") or m_id)
                existing_m = session.query(ModelRecord).filter_by(model_id=m_id).first()
                if not existing_m:
                    m_rec = ModelRecord(
                        model_id=m_id,
                        model_name=m_name,
                        architecture=model_info.get("architecture") or model_info.get("model_architecture"),
                        framework=model_info.get("framework"),
                        modality=model_info.get("modality") or model_info.get("domain") or modality,
                        model_version=str(model_info.get("model_version") or "1.0"),
                        model_path=model_info.get("model_path"),
                        total_parameters=model_info.get("total_parameters"),
                        trainable_parameters=model_info.get("trainable_parameters"),
                        num_classes=model_info.get("num_classes"),
                        input_shape=str(model_info.get("input_shape")),
                        metadata_json=model_info.get("metadata") or model_info,
                    )
                    session.add(m_rec)
                else:
                    if model_info.get("architecture") or model_info.get("model_architecture"):
                        existing_m.architecture = model_info.get("architecture") or model_info.get("model_architecture")
                    if model_info.get("framework"):
                        existing_m.framework = model_info.get("framework")
                    if model_info.get("total_parameters") is not None:
                        existing_m.total_parameters = model_info.get("total_parameters")
                    if model_info.get("trainable_parameters") is not None:
                        existing_m.trainable_parameters = model_info.get("trainable_parameters")
                    if model_info.get("num_classes") is not None:
                        try:
                            existing_m.num_classes = int(model_info.get("num_classes"))
                        except (ValueError, TypeError):
                            pass
                    if model_info.get("input_shape"):
                        existing_m.input_shape = str(model_info.get("input_shape"))

            # 2. Upsert dataset record if provided
            d_id = None
            d_name = None
            if dataset_info:
                d_id = str(dataset_info.get("dataset_id") or dataset_info.get("dataset_name") or "eval-dataset")
                d_name = str(dataset_info.get("dataset_name") or d_id)
                existing_d = session.query(DatasetRecord).filter_by(dataset_id=d_id).first()
                if not existing_d:
                    d_rec = DatasetRecord(
                        dataset_id=d_id,
                        dataset_name=d_name,
                        dataset_version=str(dataset_info.get("dataset_version") or "1.0"),
                        modality=dataset_info.get("modality") or dataset_info.get("domain") or modality,
                        number_of_samples=dataset_info.get("number_of_samples") or dataset_info.get("total_samples"),
                        number_of_classes=dataset_info.get("number_of_classes"),
                        dataset_path=dataset_info.get("dataset_path"),
                        data_split=dataset_info.get("data_split") or dataset_info.get("split", "test"),
                        metadata_json=dataset_info,
                    )
                    session.add(d_rec)
                else:
                    if dataset_info.get("number_of_samples") or dataset_info.get("total_samples"):
                        existing_d.number_of_samples = dataset_info.get("number_of_samples") or dataset_info.get("total_samples")
                    if dataset_info.get("data_split") or dataset_info.get("split"):
                        existing_d.data_split = dataset_info.get("data_split") or dataset_info.get("split")
                    if dataset_info.get("number_of_classes"):
                        try:
                            existing_d.number_of_classes = int(dataset_info.get("number_of_classes"))
                        except (ValueError, TypeError):
                            pass

            # 3. Create AssessmentRecord
            res_modality = modality or (model_info.get("domain") if model_info else None) or "image"
            clean_modality = str(res_modality).lower().replace("-", "_")

            scan = AssessmentRecord(
                scan_id=final_scan_id,
                assessment_name=resolved_name,
                created_at=datetime.now(timezone.utc),
                status="running",
                model_id=m_id,
                model_name=m_name,
                dataset_id=d_id,
                dataset_name=d_name,
                modality=clean_modality,
                configuration=configuration or {},
                parent_scan_id=parent_scan_id,
                baseline_scan_id=baseline_scan_id,
            )
            session.add(scan)
            session.flush()
            return scan.to_dict()

    def update_assessment_status(
        self,
        scan_id: str,
        status: str,  # 'running', 'completed', 'partially_completed', 'failed'
        error_message: Optional[str] = None,
        overall_summary: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update final or mid-pipeline assessment status."""
        with get_db_session(self.db_path) as session:
            scan = session.query(AssessmentRecord).filter_by(scan_id=scan_id).first()
            if not scan:
                return False
            scan.status = status
            if status in ("completed", "partially_completed", "failed"):
                scan.completed_at = datetime.now(timezone.utc)
            if error_message:
                scan.error_message = str(error_message)
            if overall_summary:
                scan.overall_summary = overall_summary
            return True

    # --------------------------------------------------------------------------
    # 2. Attack Execution Persistence
    # --------------------------------------------------------------------------

    def record_attack_execution(
        self,
        scan_id: str,
        attack_result: Dict[str, Any],
        attack_params: Optional[Dict[str, Any]] = None,
        adversarial_examples: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Record attack metrics and offload large tensor samples to the artifact store.
        """
        run_id = f"ATK-{uuid.uuid4().hex[:8].upper()}"
        atk_name = str(attack_result.get("attack_name", "Unknown_Attack")).upper()

        # Offload large tensor/samples to filesystem
        samples_path = None
        adv_obj = adversarial_examples
        if adv_obj is None and "_attack_result_obj" in attack_result:
            adv_obj = getattr(attack_result["_attack_result_obj"], "adversarial_examples", None)
        if adv_obj is None:
            adv_obj = attack_result.get("adversarial_examples")

        if adv_obj is not None:
            samples_path = self.artifacts.save_artifact(
                data=adv_obj,
                scan_id=scan_id,
                category="attacks",
                filename_prefix=f"{atk_name.lower()}_adv_samples",
            )

        # Resolve full attack parameters dictionary
        resolved_params: Dict[str, Any] = {}
        for candidate in (
            attack_result.get("parameters"),
            attack_result.get("attack_parameters"),
            attack_result.get("hyperparameters"),
            attack_result.get("config"),
            attack_params,
        ):
            if isinstance(candidate, dict):
                resolved_params.update(candidate)

        # Ensure explicit epsilon, bounds, and norms are in parameters dictionary
        eps_val = attack_result.get("epsilon_used") or attack_result.get("epsilon") or resolved_params.get("epsilon")
        if eps_val is not None:
            resolved_params["epsilon"] = float(eps_val)
        if "clip_min" not in resolved_params and attack_result.get("clip_min") is not None:
            resolved_params["clip_min"] = attack_result.get("clip_min")
        if "clip_max" not in resolved_params and attack_result.get("clip_max") is not None:
            resolved_params["clip_max"] = attack_result.get("clip_max")
        if "norm" not in resolved_params and attack_result.get("norm"):
            resolved_params["norm"] = attack_result.get("norm")

        # Resolve execution time in seconds
        exec_time = float(
            attack_result.get("execution_time")
            or attack_result.get("execution_time_seconds")
            or attack_result.get("duration")
            or attack_result.get("time_taken")
            or 0.0
        )

        # Resolve total evaluated samples
        tot_samples = int(
            attack_result.get("total_samples")
            or attack_result.get("sample_count")
            or attack_result.get("num_samples")
            or attack_result.get("evaluated_samples")
            or (len(adv_obj) if hasattr(adv_obj, "__len__") and not isinstance(adv_obj, (dict, str)) else 0)
        )

        # Resolve perturbation metrics
        pert_metrics = attack_result.get("perturbation_metrics") or {}
        if not isinstance(pert_metrics, dict):
            pert_metrics = {}
        if "ms_per_sample" not in pert_metrics and attack_result.get("ms_per_sample"):
            pert_metrics["ms_per_sample"] = float(attack_result["ms_per_sample"])
        elif "ms_per_sample" not in pert_metrics and tot_samples > 0 and exec_time > 0:
            pert_metrics["ms_per_sample"] = round((exec_time * 1000.0) / tot_samples, 2)

        with get_db_session(self.db_path) as session:
            rec = AttackExecutionRecord(
                attack_run_id=run_id,
                scan_id=scan_id,
                attack_name=atk_name,
                attack_type=str(attack_result.get("attack_type", "white_box")),
                attack_objective="targeted" if attack_result.get("is_targeted") else "untargeted",
                target_label=attack_result.get("target_label"),
                epsilon=float(eps_val or 0.0),
                attack_parameters=resolved_params,
                execution_status="completed" if attack_result.get("clean_accuracy") is not None else "failed",
                execution_time_seconds=exec_time,
                total_samples=tot_samples,
                clean_accuracy=float(attack_result.get("clean_accuracy", 0.0)),
                adversarial_accuracy=float(attack_result.get("adversarial_accuracy", 0.0)),
                accuracy_drop=float(attack_result.get("accuracy_drop", 0.0)),
                attack_success_rate=float(attack_result.get("asr_percentage", 0.0)),
                successful_attacks_count=int(attack_result.get("successful_count", 0)),
                clean_confidence_avg=attack_result.get("clean_confidence_avg") or attack_result.get("clean_confidence"),
                adv_confidence_avg=attack_result.get("adv_confidence_avg") or attack_result.get("adversarial_confidence"),
                confidence_drop=attack_result.get("confidence_drop"),
                l_inf_norm=attack_result.get("linf") or attack_result.get("l_inf"),
                l2_norm=attack_result.get("l2"),
                l0_norm=attack_result.get("l0"),
                mae=attack_result.get("mae"),
                perturbation_metrics=pert_metrics,
                prediction_changed=bool(attack_result.get("accuracy_drop", 0.0) > 0.0),
                attack_success=bool(attack_result.get("asr_percentage", 0.0) > 0.0),
                failure_reason=attack_result.get("error_message"),
                adversarial_samples_path=samples_path,
                created_at=datetime.now(timezone.utc),
            )
            session.add(rec)
            session.flush()
            return rec.to_dict()

    # --------------------------------------------------------------------------
    # 3. Defense Execution Persistence
    # --------------------------------------------------------------------------

    def record_defense_execution(
        self,
        scan_id: str,
        defense_result: Dict[str, Any],
        attack_run_id: Optional[str] = None,
        hardened_samples: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Record applied defense execution and comparative metrics."""
        run_id = f"DEF-{uuid.uuid4().hex[:8].upper()}"
        def_name = str(defense_result.get("defense_name") or defense_result.get("selected_defense", "defense"))

        samples_path = None
        if hardened_samples is not None:
            samples_path = self.artifacts.save_artifact(
                data=hardened_samples,
                scan_id=scan_id,
                category="defenses",
                filename_prefix=f"{def_name.lower()}_hardened_samples",
            )

        with get_db_session(self.db_path) as session:
            rec = DefenseExecutionRecord(
                defense_run_id=run_id,
                scan_id=scan_id,
                attack_run_id=attack_run_id,
                defense_name=def_name,
                defense_family=defense_result.get("defense_family") or defense_result.get("defense_type") or "Preprocessing",
                defense_stage=defense_result.get("defense_stage", "preprocessing"),
                defense_configuration=defense_result.get("parameters") or defense_result.get("defense_configuration") or defense_result.get("custom_params") or {},
                execution_status="completed" if defense_result.get("success", True) else "failed",
                execution_time_seconds=float(defense_result.get("execution_time_seconds", 0.0)),
                clean_accuracy_before=float(defense_result.get("before_clean_accuracy", 0.0)),
                clean_accuracy_after=float(defense_result.get("after_clean_accuracy", 0.0)),
                clean_accuracy_drop=float(defense_result.get("clean_accuracy_drop", 0.0)),
                adv_accuracy_before=float(defense_result.get("before_adv_accuracy", 0.0)),
                adv_accuracy_after=float(defense_result.get("after_adv_accuracy", 0.0)),
                asr_before=float(defense_result.get("before_asr", 0.0)),
                asr_after=float(defense_result.get("after_asr", 0.0)),
                asr_reduction=float(defense_result.get("asr_reduction_percentage", 0.0)),
                accuracy_recovery=float(defense_result.get("accuracy_recovery", 0.0)),
                robustness_status=defense_result.get("robustness_status") or defense_result.get("defense_effectiveness"),
                is_improved=bool(defense_result.get("is_improved", False)),
                validation_status=str(defense_result.get("status", "accepted")),
                before_metrics=defense_result.get("metrics_before") or {},
                after_metrics=defense_result.get("metrics_after") or {},
                hardened_samples_path=samples_path,
                created_at=datetime.now(timezone.utc),
            )
            session.add(rec)
            session.flush()
            return rec.to_dict()

    # --------------------------------------------------------------------------
    # 4. XAI Results Persistence
    # --------------------------------------------------------------------------

    def record_xai_result(
        self,
        scan_id: str,
        xai_result: Dict[str, Any],
        attack_run_id: Optional[str] = None,
        raw_attributions: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Record explainability summaries and store full multi-dimensional attributions to disk.
        """
        xai_id = f"XAI-{uuid.uuid4().hex[:8].upper()}"
        tech_name = str(xai_result.get("technique", "saliency")).lower()

        attr_obj = raw_attributions if raw_attributions is not None else xai_result.get("attribution")
        artifact_path = None
        if attr_obj is not None:
            artifact_path = self.artifacts.save_artifact(
                data=attr_obj,
                scan_id=scan_id,
                category="xai",
                filename_prefix=f"{tech_name}_attribution_map",
            )

        with get_db_session(self.db_path) as session:
            local_exp = xai_result.get("local_explanation", {})
            feat_imp = xai_result.get("feature_importance", {})

            rec = XAIResultRecord(
                xai_id=xai_id,
                scan_id=scan_id,
                attack_run_id=attack_run_id,
                explainer_technique=tech_name,
                modality=xai_result.get("modality", "image"),
                target_class=xai_result.get("target_class"),
                clean_prediction=xai_result.get("clean_prediction"),
                adversarial_prediction=xai_result.get("adversarial_prediction"),
                clean_confidence=xai_result.get("clean_confidence"),
                adversarial_confidence=xai_result.get("adversarial_confidence"),
                prediction_changed=xai_result.get("prediction_changed"),
                attribution_drift=local_exp.get("attribution_drift"),
                feature_salience_distortion=xai_result.get("feature_salience_distortion"),
                top_features_clean=local_exp.get("clean_top_features") or feat_imp.get("clean_top_features", []),
                top_features_adversarial=local_exp.get("adversarial_top_features") or feat_imp.get("adversarial_top_features", []),
                root_cause_narrative=xai_result.get("message"),
                execution_status="completed" if xai_result.get("executed", True) else "failed",
                samples_explained=int(xai_result.get("samples_explained", 1)),
                attribution_artifact_path=artifact_path,
                created_at=datetime.now(timezone.utc),
            )
            session.add(rec)
            session.flush()
            return rec.to_dict()

    # --------------------------------------------------------------------------
    # 5. Vulnerability Findings & Recommendations
    # --------------------------------------------------------------------------

    def record_vulnerabilities(
        self,
        scan_id: str,
        vuln_result: Dict[str, Any],
        attack_run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Record diagnostic findings produced by Module 5."""
        out = []
        findings = vuln_result.get("vulnerabilities") or vuln_result.get("findings") or []
        if not findings and ("vulnerability_score" in vuln_result or "score" in vuln_result or "risk_level" in vuln_result or "sub_scores" in vuln_result):
            score_val = float(vuln_result.get("vulnerability_score") if vuln_result.get("vulnerability_score") is not None else vuln_result.get("score", 0.0))
            risk_val = str(vuln_result.get("risk_level") or vuln_result.get("severity", "Medium")).upper()
            v_type = vuln_result.get("vulnerability_type") or "Adversarial Perturbation Sensitivity"

            findings = [{
                "name": f"{v_type} [{risk_val}]",
                "severity": risk_val.capitalize(),
                "score": score_val,
                "description": vuln_result.get("description") or f"Model exhibits {risk_val} adversarial sensitivity with composite vulnerability score of {score_val:.1f}/100.",
                "evidence": vuln_result.get("metric_explanations") or {"score": score_val, "risk_level": risk_val},
                "affected_region": "Input feature space / Decision boundary",
            }]

            sub_scores = vuln_result.get("sub_scores") or {}
            if sub_scores.get("asr_score") is not None:
                findings.append({
                    "name": "Attack Success Susceptibility (ASR)",
                    "severity": "High" if float(sub_scores["asr_score"]) >= 50.0 else ("Medium" if float(sub_scores["asr_score"]) >= 20.0 else "Low"),
                    "score": float(sub_scores["asr_score"]),
                    "description": "Susceptibility to misclassification when inputs are subjected to targeted or untargeted perturbations.",
                    "evidence": {"sub_score": sub_scores["asr_score"], "weight": 0.50},
                    "affected_region": "Prediction logits",
                })
            if sub_scores.get("degradation_score") is not None:
                findings.append({
                    "name": "Confidence & Margin Degradation",
                    "severity": "High" if float(sub_scores["degradation_score"]) >= 40.0 else ("Medium" if float(sub_scores["degradation_score"]) >= 15.0 else "Low"),
                    "score": float(sub_scores["degradation_score"]),
                    "description": "Significant loss of prediction margin and softmax probability degradation under adversarial attack.",
                    "evidence": {"sub_score": sub_scores["degradation_score"], "weight": 0.35},
                    "affected_region": "Softmax margin",
                })
            if sub_scores.get("stealth_score") is not None:
                findings.append({
                    "name": "Perturbation Stealth & Distortion Vulnerability",
                    "severity": "High" if float(sub_scores["stealth_score"]) >= 50.0 else ("Medium" if float(sub_scores["stealth_score"]) >= 20.0 else "Low"),
                    "score": float(sub_scores["stealth_score"]),
                    "description": "Perturbation magnitude remains within allowable human imperceptibility bounds.",
                    "evidence": {"sub_score": sub_scores["stealth_score"], "weight": 0.15},
                    "affected_region": "High-frequency features",
                })

        with get_db_session(self.db_path) as session:
            for item in findings:
                f_id = f"VULN-{uuid.uuid4().hex[:8].upper()}"
                rec = VulnerabilityFindingRecord(
                    finding_id=f_id,
                    scan_id=scan_id,
                    attack_run_id=attack_run_id,
                    vulnerability_name=item.get("name", "Unknown Vulnerability"),
                    affected_feature_or_region=item.get("affected_region"),
                    severity=item.get("severity", "Medium"),
                    vulnerability_score=float(item.get("score", 0.0)),
                    description=item.get("description"),
                    evidence=item.get("evidence") or item,
                    finding_timestamp=datetime.now(timezone.utc),
                )
                session.add(rec)
                session.flush()
                out.append(rec.to_dict())
        return out

    def record_recommendations(
        self,
        scan_id: str,
        recommendations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Record actionable hardening recommendations."""
        out = []
        with get_db_session(self.db_path) as session:
            for item in recommendations:
                r_id = f"REC-{uuid.uuid4().hex[:8].upper()}"
                rec = DefenseRecommendationRecord(
                    recommendation_id=r_id,
                    scan_id=scan_id,
                    finding_id=item.get("finding_id"),
                    recommended_technique=item.get("technique") or item.get("recommended_technique", "auto"),
                    defense_family=item.get("defense_family"),
                    defense_stage=item.get("defense_stage", "preprocessing"),
                    confidence=float(item.get("confidence", 0.8)),
                    evidence=item.get("evidence") or {},
                    rationale=item.get("rationale") or item.get("description"),
                    validation_status=item.get("status", "recommended"),
                    created_at=datetime.now(timezone.utc),
                )
                session.add(rec)
                session.flush()
                out.append(rec.to_dict())
        return out

    # --------------------------------------------------------------------------
    # 6. Report Persistence
    # --------------------------------------------------------------------------

    def record_report(
        self,
        scan_id: str,
        report_path: str,
        report_name: Optional[str] = None,
        report_format: str = "pdf",
        summary_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record generated PDF or JSON report path linked to the assessment."""
        rep_id = f"REP-{uuid.uuid4().hex[:8].upper()}"
        res_name = report_name or f"Security_Report_{scan_id}.{report_format.lower()}"

        with get_db_session(self.db_path) as session:
            rec = ReportRecord(
                report_id=rep_id,
                scan_id=scan_id,
                report_name=res_name,
                report_path=str(report_path),
                report_format=report_format.lower(),
                generation_status="completed",
                summary_metrics=summary_metrics or {},
                generated_at=datetime.now(timezone.utc),
            )
            session.add(rec)
            session.flush()
            return rec.to_dict()

    # --------------------------------------------------------------------------
    # 7. Querying, Historical Retrieval & Diagnostics
    # --------------------------------------------------------------------------

    def get_assessment(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full hierarchical assessment details by scan_id."""
        with get_db_session(self.db_path) as session:
            scan = session.query(AssessmentRecord).filter_by(scan_id=scan_id).first()
            if not scan:
                return None
            return scan.to_dict(include_children=True)

    def list_assessments(
        self,
        status: Optional[str] = None,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        modality: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Retrieve historical assessment records matching query filters."""
        with get_db_session(self.db_path) as session:
            q = session.query(AssessmentRecord)
            if status:
                q = q.filter(AssessmentRecord.status == status)
            if model_name:
                q = q.filter(AssessmentRecord.model_name.ilike(f"%{model_name}%"))
            if dataset_name:
                q = q.filter(AssessmentRecord.dataset_name.ilike(f"%{dataset_name}%"))
            if modality:
                q = q.filter(AssessmentRecord.modality == modality.lower().replace("-", "_"))

            q = q.order_by(AssessmentRecord.created_at.desc())
            results = q.slice(offset, offset + limit).all()
            return [s.to_dict(include_children=True) for s in results]

    def delete_assessment(self, scan_id: str) -> bool:
        """Delete an assessment and its cascading children."""
        with get_db_session(self.db_path) as session:
            scan = session.query(AssessmentRecord).filter_by(scan_id=scan_id).first()
            if not scan:
                return False
            session.delete(scan)
            return True

    def get_storage_stats(self) -> Dict[str, Any]:
        """Diagnostic summary of database size, records, and connection status."""
        p = Path(self.db_path)
        exists = p.is_file()
        size_bytes = p.stat().st_size if exists else 0

        with get_db_session(self.db_path) as session:
            total_scans = session.query(AssessmentRecord).count()
            completed_scans = session.query(AssessmentRecord).filter_by(status="completed").count()
            failed_scans = session.query(AssessmentRecord).filter_by(status="failed").count()
            running_scans = session.query(AssessmentRecord).filter_by(status="running").count()
            total_attacks = session.query(AttackExecutionRecord).count()
            total_defenses = session.query(DefenseExecutionRecord).count()
            total_reports = session.query(ReportRecord).count()

        return {
            "database_path": str(p),
            "database_exists": exists,
            "database_size_mb": round(size_bytes / (1024 * 1024), 2),
            "total_assessments": total_scans,
            "completed_assessments": completed_scans,
            "failed_assessments": failed_scans,
            "running_assessments": running_scans,
            "total_attacks_recorded": total_attacks,
            "total_defenses_recorded": total_defenses,
            "total_reports_recorded": total_reports,
            "wal_mode_enabled": True,
        }
