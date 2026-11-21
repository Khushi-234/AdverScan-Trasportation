"""
SQLAlchemy ORM models for the AdverScan SQLite database.

Defines the normalized relational schema for tracking:
1. Assessments (Scans) & lineage (baseline/parent relationships)
2. Models registry
3. Datasets registry
4. Attack executions & perturbation metrics
5. Defense executions & hardening validations
6. XAI root-cause results & attribution drift
7. Vulnerability findings & risk scoring
8. Defense recommendations
9. Generated security reports
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

def _utc_now():
    return datetime.now(timezone.utc)

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from app.storage.db import Base


# ==============================================================================
# 1. MODEL REGISTRY
# ==============================================================================

class ModelRecord(Base):
    """Stores ingested model architecture, framework, and parameter metadata."""
    __tablename__ = "models"

    model_id = Column(String(255), primary_key=True)
    model_name = Column(String(255), nullable=False)
    architecture = Column(String(255), nullable=True)
    framework = Column(String(64), nullable=True)  # pytorch, huggingface, sklearn
    modality = Column(String(64), nullable=True, index=True)  # image, tabular, text, time_series, audio
    model_version = Column(String(64), nullable=True)
    model_path = Column(String(512), nullable=True)
    total_parameters = Column(BigInteger, nullable=True)
    trainable_parameters = Column(BigInteger, nullable=True)
    num_classes = Column(Integer, nullable=True)
    input_shape = Column(String(128), nullable=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=_utc_now, nullable=False)

    assessments = relationship("AssessmentRecord", back_populates="model")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "architecture": self.architecture,
            "framework": self.framework,
            "modality": self.modality,
            "model_version": self.model_version,
            "model_path": self.model_path,
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "num_classes": self.num_classes,
            "input_shape": self.input_shape,
            "metadata_json": self.metadata_json or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ==============================================================================
# 2. DATASET REGISTRY
# ==============================================================================

class DatasetRecord(Base):
    """Stores dataset metadata, split, sample counts, and feature dimensions."""
    __tablename__ = "datasets"

    dataset_id = Column(String(255), primary_key=True)
    dataset_name = Column(String(255), nullable=False)
    dataset_version = Column(String(64), nullable=True)
    modality = Column(String(64), nullable=True, index=True)
    number_of_samples = Column(Integer, nullable=True)
    number_of_classes = Column(Integer, nullable=True)
    dataset_path = Column(String(512), nullable=True)
    data_split = Column(String(64), nullable=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=_utc_now, nullable=False)

    assessments = relationship("AssessmentRecord", back_populates="dataset")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "modality": self.modality,
            "number_of_samples": self.number_of_samples,
            "number_of_classes": self.number_of_classes,
            "dataset_path": self.dataset_path,
            "data_split": self.data_split,
            "metadata_json": self.metadata_json or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ==============================================================================
# 3. ASSESSMENT / SCAN RECORD
# ==============================================================================

class AssessmentRecord(Base):
    """
    Primary container representing a single completed or in-progress assessment.
    Correlates models, datasets, attacks, defenses, XAI, findings, and reports.
    """
    __tablename__ = "assessments"

    scan_id = Column(String(64), primary_key=True)
    assessment_name = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=_utc_now, nullable=False, index=True)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(32), default="pending", nullable=False, index=True)  # pending, running, completed, partially_completed, failed

    model_id = Column(String(255), ForeignKey("models.model_id", ondelete="SET NULL"), nullable=True, index=True)
    model_name = Column(String(255), nullable=True)
    dataset_id = Column(String(255), ForeignKey("datasets.dataset_id", ondelete="SET NULL"), nullable=True, index=True)
    dataset_name = Column(String(255), nullable=True)
    modality = Column(String(64), nullable=True, index=True)

    configuration = Column(JSON, nullable=True)
    overall_summary = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)

    # Lineage and cross-scan correlation
    parent_scan_id = Column(String(64), ForeignKey("assessments.scan_id", ondelete="SET NULL"), nullable=True, index=True)
    baseline_scan_id = Column(String(64), ForeignKey("assessments.scan_id", ondelete="SET NULL"), nullable=True, index=True)

    # Relationships
    model = relationship("ModelRecord", back_populates="assessments")
    dataset = relationship("DatasetRecord", back_populates="assessments")

    attacks = relationship("AttackExecutionRecord", back_populates="assessment", cascade="all, delete-orphan")
    defenses = relationship("DefenseExecutionRecord", back_populates="assessment", cascade="all, delete-orphan")
    xai_results = relationship("XAIResultRecord", back_populates="assessment", cascade="all, delete-orphan")
    vulnerabilities = relationship("VulnerabilityFindingRecord", back_populates="assessment", cascade="all, delete-orphan")
    recommendations = relationship("DefenseRecommendationRecord", back_populates="assessment", cascade="all, delete-orphan")
    reports = relationship("ReportRecord", back_populates="assessment", cascade="all, delete-orphan")

    def to_dict(self, include_children: bool = False) -> Dict[str, Any]:
        d = {
            "scan_id": self.scan_id,
            "assessment_name": self.assessment_name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "modality": self.modality,
            "configuration": self.configuration or {},
            "overall_summary": self.overall_summary or {},
            "error_message": self.error_message,
            "parent_scan_id": self.parent_scan_id,
            "baseline_scan_id": self.baseline_scan_id,
        }
        if include_children:
            if self.model:
                d["model"] = self.model.to_dict()
            if self.dataset:
                d["dataset"] = self.dataset.to_dict()
            d["attacks"] = [a.to_dict() for a in self.attacks]
            d["defenses"] = [df.to_dict() for df in self.defenses]
            d["xai_results"] = [x.to_dict() for x in self.xai_results]
            d["vulnerabilities"] = [v.to_dict() for v in self.vulnerabilities]
            d["recommendations"] = [r.to_dict() for r in self.recommendations]
            d["reports"] = [rp.to_dict() for rp in self.reports]
        return d


# ==============================================================================
# 4. ATTACK EXECUTIONS
# ==============================================================================

class AttackExecutionRecord(Base):
    """Records an executed adversarial attack, telemetry, and perturbation statistics."""
    __tablename__ = "attack_executions"

    attack_run_id = Column(String(64), primary_key=True)
    scan_id = Column(String(64), ForeignKey("assessments.scan_id", ondelete="CASCADE"), nullable=False, index=True)
    attack_name = Column(String(128), nullable=False, index=True)
    attack_type = Column(String(64), default="white_box")  # white_box, black_box, gray_box
    attack_objective = Column(String(64), default="untargeted")  # untargeted, targeted
    target_label = Column(Integer, nullable=True)
    epsilon = Column(Float, nullable=True)
    attack_parameters = Column(JSON, nullable=True)
    execution_status = Column(String(32), default="completed")  # completed, failed
    execution_time_seconds = Column(Float, nullable=True)

    # Core accuracy and vulnerability metrics
    total_samples = Column(Integer, default=0)
    clean_accuracy = Column(Float, nullable=True)  # 0.0 - 100.0
    adversarial_accuracy = Column(Float, nullable=True)  # 0.0 - 100.0
    accuracy_drop = Column(Float, nullable=True)  # clean - adv
    attack_success_rate = Column(Float, nullable=True)  # ASR percentage (0.0 - 100.0)
    successful_attacks_count = Column(Integer, default=0)

    # Confidence metrics
    clean_confidence_avg = Column(Float, nullable=True)  # 0.0 - 100.0
    adv_confidence_avg = Column(Float, nullable=True)
    confidence_drop = Column(Float, nullable=True)

    # Distortion norms
    l_inf_norm = Column(Float, nullable=True)
    l2_norm = Column(Float, nullable=True)
    l0_norm = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    perturbation_metrics = Column(JSON, nullable=True)

    # Status flags & filesystem references
    prediction_changed = Column(Boolean, nullable=True)
    attack_success = Column(Boolean, nullable=True)
    failure_reason = Column(Text, nullable=True)
    adversarial_samples_path = Column(String(512), nullable=True)  # Relative path to .npz or .pt artifact
    created_at = Column(DateTime, default=_utc_now, nullable=False)

    assessment = relationship("AssessmentRecord", back_populates="attacks")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attack_run_id": self.attack_run_id,
            "scan_id": self.scan_id,
            "attack_name": self.attack_name,
            "attack_type": self.attack_type,
            "attack_objective": self.attack_objective,
            "target_label": self.target_label,
            "epsilon": self.epsilon,
            "attack_parameters": self.attack_parameters or {},
            "execution_status": self.execution_status,
            "execution_time_seconds": self.execution_time_seconds,
            "total_samples": self.total_samples,
            "clean_accuracy": self.clean_accuracy,
            "adversarial_accuracy": self.adversarial_accuracy,
            "accuracy_drop": self.accuracy_drop,
            "attack_success_rate": self.attack_success_rate,
            "successful_attacks_count": self.successful_attacks_count,
            "clean_confidence_avg": self.clean_confidence_avg,
            "adv_confidence_avg": self.adv_confidence_avg,
            "confidence_drop": self.confidence_drop,
            "l_inf_norm": self.l_inf_norm,
            "l2_norm": self.l2_norm,
            "l0_norm": self.l0_norm,
            "mae": self.mae,
            "perturbation_metrics": self.perturbation_metrics or {},
            "prediction_changed": self.prediction_changed,
            "attack_success": self.attack_success,
            "failure_reason": self.failure_reason,
            "adversarial_samples_path": self.adversarial_samples_path,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ==============================================================================
# 5. DEFENSE EXECUTIONS
# ==============================================================================

class DefenseExecutionRecord(Base):
    """Records an applied defense, hardening stage, before/after metrics, and validation status."""
    __tablename__ = "defense_executions"

    defense_run_id = Column(String(64), primary_key=True)
    scan_id = Column(String(64), ForeignKey("assessments.scan_id", ondelete="CASCADE"), nullable=False, index=True)
    attack_run_id = Column(String(64), ForeignKey("attack_executions.attack_run_id", ondelete="SET NULL"), nullable=True)

    defense_name = Column(String(128), nullable=False, index=True)
    defense_family = Column(String(64), nullable=True)  # preprocessing, smoothing, detection, adversarial_training
    defense_stage = Column(String(64), nullable=True)  # preprocessing, model_hardening, postprocessing
    defense_configuration = Column(JSON, nullable=True)
    execution_status = Column(String(32), default="completed")
    execution_time_seconds = Column(Float, nullable=True)

    # Comparative before/after metrics
    clean_accuracy_before = Column(Float, nullable=True)
    clean_accuracy_after = Column(Float, nullable=True)
    clean_accuracy_drop = Column(Float, nullable=True)
    adv_accuracy_before = Column(Float, nullable=True)
    adv_accuracy_after = Column(Float, nullable=True)
    asr_before = Column(Float, nullable=True)
    asr_after = Column(Float, nullable=True)
    asr_reduction = Column(Float, nullable=True)  # asr_before - asr_after
    accuracy_recovery = Column(Float, nullable=True)

    robustness_status = Column(String(64), nullable=True)  # Robust, Partial Robust, Vulnerable
    is_improved = Column(Boolean, nullable=True)
    validation_status = Column(String(64), default="evaluated")  # recommended, configured, evaluated, accepted

    before_metrics = Column(JSON, nullable=True)
    after_metrics = Column(JSON, nullable=True)
    hardened_model_path = Column(String(512), nullable=True)
    hardened_samples_path = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=_utc_now, nullable=False)

    assessment = relationship("AssessmentRecord", back_populates="defenses")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "defense_run_id": self.defense_run_id,
            "scan_id": self.scan_id,
            "attack_run_id": self.attack_run_id,
            "defense_name": self.defense_name,
            "defense_family": self.defense_family,
            "defense_stage": self.defense_stage,
            "defense_configuration": self.defense_configuration or {},
            "execution_status": self.execution_status,
            "execution_time_seconds": self.execution_time_seconds,
            "clean_accuracy_before": self.clean_accuracy_before,
            "clean_accuracy_after": self.clean_accuracy_after,
            "clean_accuracy_drop": self.clean_accuracy_drop,
            "adv_accuracy_before": self.adv_accuracy_before,
            "adv_accuracy_after": self.adv_accuracy_after,
            "asr_before": self.asr_before,
            "asr_after": self.asr_after,
            "asr_reduction": self.asr_reduction,
            "accuracy_recovery": self.accuracy_recovery,
            "robustness_status": self.robustness_status,
            "is_improved": self.is_improved,
            "validation_status": self.validation_status,
            "before_metrics": self.before_metrics or {},
            "after_metrics": self.after_metrics or {},
            "hardened_model_path": self.hardened_model_path,
            "hardened_samples_path": self.hardened_samples_path,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ==============================================================================
# 6. XAI EXPLANATION RESULTS
# ==============================================================================

class XAIResultRecord(Base):
    """Records explainability findings, attribution drift, and top features."""
    __tablename__ = "xai_results"

    xai_id = Column(String(64), primary_key=True)
    scan_id = Column(String(64), ForeignKey("assessments.scan_id", ondelete="CASCADE"), nullable=False, index=True)
    attack_run_id = Column(String(64), ForeignKey("attack_executions.attack_run_id", ondelete="SET NULL"), nullable=True)

    explainer_technique = Column(String(64), nullable=False, index=True)  # shap, saliency, integrated_gradients, gradcam
    modality = Column(String(64), nullable=True)
    target_class = Column(Integer, nullable=True)
    clean_prediction = Column(Integer, nullable=True)
    adversarial_prediction = Column(Integer, nullable=True)
    clean_confidence = Column(Float, nullable=True)
    adversarial_confidence = Column(Float, nullable=True)
    prediction_changed = Column(Boolean, nullable=True)

    # Attribution metrics and feature importance summaries
    attribution_drift = Column(Float, nullable=True)
    feature_salience_distortion = Column(Float, nullable=True)
    top_features_clean = Column(JSON, nullable=True)
    top_features_adversarial = Column(JSON, nullable=True)
    root_cause_narrative = Column(Text, nullable=True)

    execution_status = Column(String(32), default="completed")
    execution_time_seconds = Column(Float, nullable=True)
    samples_explained = Column(Integer, default=1)
    attribution_artifact_path = Column(String(512), nullable=True)  # Path to full tensor/heatmap on disk
    created_at = Column(DateTime, default=_utc_now, nullable=False)

    assessment = relationship("AssessmentRecord", back_populates="xai_results")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "xai_id": self.xai_id,
            "scan_id": self.scan_id,
            "attack_run_id": self.attack_run_id,
            "explainer_technique": self.explainer_technique,
            "modality": self.modality,
            "target_class": self.target_class,
            "clean_prediction": self.clean_prediction,
            "adversarial_prediction": self.adversarial_prediction,
            "clean_confidence": self.clean_confidence,
            "adversarial_confidence": self.adversarial_confidence,
            "prediction_changed": self.prediction_changed,
            "attribution_drift": self.attribution_drift,
            "feature_salience_distortion": self.feature_salience_distortion,
            "top_features_clean": self.top_features_clean or [],
            "top_features_adversarial": self.top_features_adversarial or [],
            "root_cause_narrative": self.root_cause_narrative,
            "execution_status": self.execution_status,
            "execution_time_seconds": self.execution_time_seconds,
            "samples_explained": self.samples_explained,
            "attribution_artifact_path": self.attribution_artifact_path,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ==============================================================================
# 7. VULNERABILITY FINDINGS
# ==============================================================================

class VulnerabilityFindingRecord(Base):
    """Records vulnerability diagnosis, severity score, and affected model regions."""
    __tablename__ = "vulnerability_findings"

    finding_id = Column(String(64), primary_key=True)
    scan_id = Column(String(64), ForeignKey("assessments.scan_id", ondelete="CASCADE"), nullable=False, index=True)
    attack_run_id = Column(String(64), ForeignKey("attack_executions.attack_run_id", ondelete="SET NULL"), nullable=True)

    vulnerability_name = Column(String(128), nullable=False)
    affected_feature_or_region = Column(String(255), nullable=True)
    severity = Column(String(32), default="Medium")  # Critical, High, Medium, Low
    vulnerability_score = Column(Float, nullable=True)  # 0.0 - 100.0
    description = Column(Text, nullable=True)
    evidence = Column(JSON, nullable=True)
    finding_timestamp = Column(DateTime, default=_utc_now, nullable=False)

    assessment = relationship("AssessmentRecord", back_populates="vulnerabilities")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "scan_id": self.scan_id,
            "attack_run_id": self.attack_run_id,
            "vulnerability_name": self.vulnerability_name,
            "affected_feature_or_region": self.affected_feature_or_region,
            "severity": self.severity,
            "vulnerability_score": self.vulnerability_score,
            "description": self.description,
            "evidence": self.evidence or {},
            "finding_timestamp": self.finding_timestamp.isoformat() if self.finding_timestamp else None,
        }


# ==============================================================================
# 8. DEFENSE RECOMMENDATIONS
# ==============================================================================

class DefenseRecommendationRecord(Base):
    """Records recommended hardening actions produced by M5/M7."""
    __tablename__ = "defense_recommendations"

    recommendation_id = Column(String(64), primary_key=True)
    scan_id = Column(String(64), ForeignKey("assessments.scan_id", ondelete="CASCADE"), nullable=False, index=True)
    finding_id = Column(String(64), ForeignKey("vulnerability_findings.finding_id", ondelete="SET NULL"), nullable=True)

    recommended_technique = Column(String(128), nullable=False)
    defense_family = Column(String(64), nullable=True)
    defense_stage = Column(String(64), nullable=True)
    confidence = Column(Float, nullable=True)
    evidence = Column(JSON, nullable=True)
    rationale = Column(Text, nullable=True)
    validation_status = Column(String(64), default="recommended")  # recommended, applied, bypassed
    created_at = Column(DateTime, default=_utc_now, nullable=False)

    assessment = relationship("AssessmentRecord", back_populates="recommendations")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "scan_id": self.scan_id,
            "finding_id": self.finding_id,
            "recommended_technique": self.recommended_technique,
            "defense_family": self.defense_family,
            "defense_stage": self.defense_stage,
            "confidence": self.confidence,
            "evidence": self.evidence or {},
            "rationale": self.rationale,
            "validation_status": self.validation_status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ==============================================================================
# 9. ASSESSMENT REPORTS
# ==============================================================================

class ReportRecord(Base):
    """Records generated PDF/JSON assessment reports linked to their scan ID."""
    __tablename__ = "assessment_reports"

    report_id = Column(String(64), primary_key=True)
    scan_id = Column(String(64), ForeignKey("assessments.scan_id", ondelete="CASCADE"), nullable=False, index=True)
    report_name = Column(String(255), nullable=False)
    report_path = Column(String(512), nullable=False)  # Path to PDF/JSON on disk
    report_format = Column(String(32), default="pdf")  # pdf, json, markdown
    generation_status = Column(String(32), default="completed")
    summary_metrics = Column(JSON, nullable=True)
    generated_at = Column(DateTime, default=_utc_now, nullable=False)

    assessment = relationship("AssessmentRecord", back_populates="reports")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "scan_id": self.scan_id,
            "report_name": self.report_name,
            "report_path": self.report_path,
            "report_format": self.report_format,
            "generation_status": self.generation_status,
            "summary_metrics": self.summary_metrics or {},
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
        }
