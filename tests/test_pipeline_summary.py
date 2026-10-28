"""
Tests for pipeline summary display and multi-attack support in main.py.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from app.ingestion.metadata.model_metadata import ModelMetadata
from app.vulnerability_analysis.assessment_result import AssessmentResult
from app.vulnerability_analysis.scoring_result import ScoringResult
from main import parse_args, step6_final_summary, ATTACK_CHOICES


def test_attack_choices_include_all():
    assert "all" in ATTACK_CHOICES
    assert "fgsm" in ATTACK_CHOICES
    assert "pgd" in ATTACK_CHOICES
    assert "deepfool" in ATTACK_CHOICES
    assert "cw" in ATTACK_CHOICES
    assert "fab" in ATTACK_CHOICES


def test_parse_args_all(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "--attack", "all", "--samples", "5"])
    args = parse_args()
    assert args.attack == "all"
    assert args.samples == 5


def test_step6_final_summary_formatting(capsys, tmp_path):
    baseline = MagicMock()
    baseline.accuracy = 1.00
    baseline.average_confidence = 0.9981
    baseline.f1_macro = 1.00

    metadata = ModelMetadata(
        framework="huggingface",
        model_name="GTSRB ViT",
        input_shape=(1, 3, 224, 224),
        output_shape=(1, 43),
        num_classes=43,
        task_type="classification",
        device="cpu",
        total_parameters=85828651,
        trainable_parameters=85828651,
        model_architecture="ViTForImageClassification",
    )

    assessment = AssessmentResult(
        attack_name="fgsm",
        dataset_name="bazyl/GTSRB",
        num_samples=10,
        attack_success_rate=0.80,
        perturbation={"linf_mean": 0.0314, "l2_mean": 1.2541, "l0_mean": 1.0, "is_estimated": False},
        accuracy_drop=0.80,
        f1_drop=0.80,
        confidence_drop=0.0298,
        model_degradation=0.5433,
        clean_accuracy=1.00,
        adversarial_accuracy=0.20,
        clean_f1=1.00,
        adversarial_f1=0.20,
        clean_confidence=0.9981,
        adversarial_confidence=0.9683,
    )

    scoring = ScoringResult(
        attack_name="fgsm",
        vulnerability_score=54.52,
        risk_level="CRITICAL",
        sub_scores={"asr_score": 80.0, "degradation_score": 41.5, "stealth_score": 0.0},
    )

    attack_eval_results = [
        {
            "attack_name": "fgsm",
            "attack_class": "FGSM",
            "execution_time": 2.71,
            "assessment": assessment,
            "scoring": scoring,
            "clean_m": {"accuracy": 1.00},
            "adv_m": {"accuracy": 0.20},
        }
    ]

    pipeline_errors = [
        {"module": "M6_explainability", "error": "TypeError — embedding(): argument 'indices' must be Tensor"}
    ]

    step6_final_summary(
        baseline=baseline,
        attack_eval_results=attack_eval_results,
        attacks_to_run=["fgsm"],
        metadata=metadata,
        device="cpu",
        samples=10,
        batch_size=32,
        pipeline_errors=pipeline_errors,
        total_pipeline_runtime=39.45,
        output_dir=tmp_path,
    )

    captured = capsys.readouterr().out

    assert "ADVERSCAN SECURITY ASSESSMENT RESULTS SUMMARY" in captured
    assert "Overall Status:            PARTIAL_SUCCESS" in captured
    assert "Execution Mode:            FULL" in captured
    assert "Total Pipeline Runtime:    39.45s" in captured
    assert "Target Modality / Domain:  IMAGE" in captured
    assert "Evaluated Model:           GTSRB ViT (bazyl/gtsrb-model)" in captured
    assert "PIPELINE ERRORS & DIAGNOSTICS" in captured
    assert "❌ [M6_explainability]" in captured
    assert "M1 — MODEL SPECIFICATIONS & PARAMETERS" in captured
    assert "ViTForImageClassification" in captured
    assert "85,828,651" in captured
    assert "Estimated Model Size:    ~327.4 MB (FP32)" in captured
    assert "Input Shape:             (1, 3, 224, 224)" in captured
    assert "DATASET & EVALUATION DETAILS" in captured
    assert "M2 — CLEAN BASELINE EVALUATION METRICS" in captured
    assert "100.00%" in captured
    assert "M3 — ADVERSARIAL ATTACK ENGINE RESULTS" in captured
    assert "▶ Attack: [FGSM]" in captured
    assert "M5 — VULNERABILITY ANALYSIS & SECURITY RISK SCORING" in captured
    assert "★ MODEL-LEVEL UNIFIED SECURITY ASSESSMENT:" in captured
    assert "Model Vulnerability Score: 54.52 / 100" in captured
    assert "Model Robustness Score:    45.48 / 100" in captured
    assert "Unified Security Risk:     CRITICAL" in captured
    assert "Primary Driving Threat:    FGSM" in captured
    assert "Score Components:      ASR=80.0, Degradation=41.5, Stealth=0.0" in captured
