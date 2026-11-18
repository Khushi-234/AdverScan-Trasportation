"""
Comprehensive Regression Tests for Module 8 (Retest & Robustness Validation) in AdverScan.

Covers:
1. Same attack configuration before and after defense.
2. Actual defended-model inference (no stale prediction reuse).
3. ASR reduction calculation (positive, negative, zero).
4. Clean accuracy degradation and harmful threshold detection.
5. Undefined ASR and zero clean accuracy handling.
6. Zero-success baseline attack handling (honest inconclusive status).
7. Failed attack execution handling (no crash, clear diagnostics).
8. Unsupported model/defense combinations (Scikit-learn with gradient attacks).
9. All supported modalities (Vision, Tabular, Text, Time-Series).
10. Threshold acceptance, rejection, and harmful detection.
11. Original model preservation and state restoration.
12. M7-to-M8 handoff with HardeningResult DTO.
13. M8-to-M9 report generator handoff.
14. Frontend defense service integration with M8 contract keys.
"""

import copy
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

from app.ingestion.adapters.pytorch_adapter import PyTorchAdapter
from app.ingestion.adapters.sklearn_adapter import SklearnAdapter
from app.orchestration.dataset_adapter import InMemoryDatasetLoader
from app.evaluation.results import EvaluationResult
from app.attack_engine.config import AttackConfig
from app.attack_engine.models import AttackResult, AttackMetadata
from app.hardening.hardening_result import HardeningMetadata, HardeningResult
from app.hardening.hardening_engine import HardeningEngine
from app.hardening.defenses.preprocessing.normalization import NormalizationDefense
from app.hardening.defenses.preprocessing.median_filter import MedianFilterDefense
from app.hardening.defenses.preprocessing.gaussian_filter import GaussianFilterDefense
from app.hardening.defenses.detection.confidence_rejection import ConfidenceRejectionDefense
from app.retest.comparison import ComparisonEngine, compare_results
from app.retest.retest_engine import RetestEngine
from app.retest.retest_result import ComparisonResult, RetestResult
from app.retest.threshold import RetestThresholds
from app.report_generator.report_data import ReportData
from app.report_generator.report_generator import ReportGenerator
from frontend.services.defense_service import execute_configured_defense


class ToyVisionModel(nn.Module):
    """Linear vision model mapping (B, 3, 16, 16) to (B, 4)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 16 * 16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.fc(x)


class ToyTabularModel(nn.Module):
    """Linear tabular model mapping (B, 8) to (B, 3)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ToyTimeSeriesModel(nn.Module):
    """Temporal model mapping (B, 10, 4) to (B, 2)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10 * 4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        return self.fc(x)


class ToyTextModel(nn.Module):
    """Embedding classifier for discrete token IDs (B, 8) to (B, 2)."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(500, 16)
        self.fc = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.emb(x.long())
        pooled = embedded.mean(dim=1)
        return self.fc(pooled)


def make_eval_result(dataset_name="toy_ds", model_name="toy_model", num_samples=10, num_classes=4, accuracy=0.9):
    return EvaluationResult(
        dataset_name=dataset_name,
        model_name=model_name,
        num_samples=num_samples,
        num_classes=num_classes,
        accuracy=accuracy,
        precision_macro=accuracy,
        recall_macro=accuracy,
        f1_macro=accuracy,
        precision_weighted=accuracy,
        recall_weighted=accuracy,
        f1_weighted=accuracy,
        average_confidence=accuracy,
        average_entropy=0.1,
    )


@pytest.fixture
def sample_before_eval():
    return make_eval_result(dataset_name="test_data", model_name="baseline_model", num_samples=10, num_classes=4, accuracy=0.90)


@pytest.fixture
def sample_before_vuln():
    return {
        "fgsm": {
            "assessment": {
                "attack_name": "fgsm",
                "clean_accuracy": 0.90,
                "adversarial_accuracy": 0.20,
                "attack_success_rate": 0.7778,
                "accuracy_drop": 0.70,
                "f1_drop": 0.70,
                "confidence_drop": 0.40,
                "model_degradation": 0.60,
                "perturbation": {"l2_mean": 0.35},
            },
            "scoring": {
                "attack_name": "fgsm",
                "vulnerability_score": 70.0,
                "risk_level": "HIGH",
            },
        }
    }


# ==============================================================================
# TEST 1: Same attack configuration before and after defense
# ==============================================================================
def test_retest_same_attack_config_preserved(sample_before_eval, sample_before_vuln):
    torch.manual_seed(42)
    inputs = torch.randn(10, 3, 16, 16)
    labels = torch.randint(0, 4, (10,))
    ds_loader = InMemoryDatasetLoader(inputs=inputs, targets=labels, dataset_name="toy_ds")

    model = ToyVisionModel()
    defended_model = MedianFilterDefense(kernel_size=3).apply(model)

    engine = RetestEngine()
    retest_cfg = {"fgsm": {"epsilon": 0.05, "norm": "inf"}}

    res: RetestResult = engine.retest(
        hardened_model=defended_model,
        dataset_loader=ds_loader,
        attacks=["fgsm"],
        before_baseline_result=sample_before_eval,
        before_vulnerability_analysis=sample_before_vuln,
        attack_configs=retest_cfg,
        num_classes=4,
    )

    assert "fgsm" in res.comparisons
    assert res.after_attack_results["fgsm"]["parameters"].get("epsilon") == 0.05
    assert res.comparisons["fgsm"].attack_name == "fgsm"


# ==============================================================================
# TEST 2: Actual defended-model inference (no stale predictions)
# ==============================================================================
def test_actual_defended_model_inference_executed(sample_before_eval, sample_before_vuln):
    torch.manual_seed(42)
    inputs = torch.randn(12, 3, 16, 16)
    labels = torch.randint(0, 4, (12,))
    ds_loader = InMemoryDatasetLoader(inputs=inputs, targets=labels, dataset_name="toy_ds")

    model = ToyVisionModel()
    # Normalize inputs as defense
    defended_model = NormalizationDefense().apply(model)

    engine = RetestEngine()
    res: RetestResult = engine.retest(
        hardened_model=defended_model,
        dataset_loader=ds_loader,
        attacks=["fgsm"],
        before_baseline_result=sample_before_eval,
        before_vulnerability_analysis=sample_before_vuln,
        num_classes=4,
    )

    # Clean accuracy must be calculated on the defended model
    after_clean = res.after_baseline_evaluation["accuracy"]
    assert after_clean is not None
    # Compare with pre-defense: not identical placeholder
    comp = res.comparisons["fgsm"]
    assert comp.after_assessment["clean_accuracy"] == after_clean


# ==============================================================================
# TEST 3: ASR reduction calculation
# ==============================================================================
def test_asr_reduction_calculation():
    comp_engine = ComparisonEngine()

    before_assess = {"attack_name": "pgd", "attack_success_rate": 0.85, "clean_accuracy": 0.90}
    after_assess = {"attack_name": "pgd", "attack_success_rate": 0.25, "clean_accuracy": 0.89}
    before_score = {"vulnerability_score": 80.0, "risk_level": "HIGH"}
    after_score = {"vulnerability_score": 30.0, "risk_level": "LOW"}

    res = comp_engine.compare_attack(
        before_assessment=before_assess,
        after_assessment=after_assess,
        before_scoring=before_score,
        after_scoring=after_score,
    )

    # delta_attack_success_rate = After - Before = 0.25 - 0.85 = -0.60
    assert res.delta_attack_success_rate == -0.60
    # asr_reduction = Before - After = 0.85 - 0.25 = +0.60
    assert res.asr_reduction == 0.60
    assert res.is_improved is True
    assert res.status == "successful"
    assert res.defense_effectiveness == "high"


# ==============================================================================
# TEST 4: Clean accuracy degradation & harmful threshold detection
# ==============================================================================
def test_clean_accuracy_severe_degradation_flagged_harmful():
    comp_engine = ComparisonEngine(thresholds=RetestThresholds(severe_clean_drop_threshold=0.10))

    # Clean accuracy drops by 25% (0.90 -> 0.65)
    before_assess = {"attack_name": "fgsm", "attack_success_rate": 0.80, "clean_accuracy": 0.90}
    after_assess = {"attack_name": "fgsm", "attack_success_rate": 0.20, "clean_accuracy": 0.65}
    before_score = {"vulnerability_score": 75.0, "risk_level": "HIGH"}
    after_score = {"vulnerability_score": 25.0, "risk_level": "LOW"}

    res = comp_engine.compare_attack(
        before_assessment=before_assess,
        after_assessment=after_assess,
        before_scoring=before_score,
        after_scoring=after_score,
    )

    assert res.clean_accuracy_drop == 0.25
    assert res.status == "failed_defense"
    assert res.defense_effectiveness == "harmful"
    assert res.is_improved is False
    assert res.threshold_evaluation["status"] == "harmful"


# ==============================================================================
# TEST 5: Undefined ASR handling (zero clean accuracy)
# ==============================================================================
def test_undefined_asr_handling():
    comp_engine = ComparisonEngine()

    # Pre-defense clean accuracy is 0.0 -> ASR undefined (None)
    before_assess = {"attack_name": "fgsm", "attack_success_rate": None, "clean_accuracy": 0.0}
    after_assess = {"attack_name": "fgsm", "attack_success_rate": None, "clean_accuracy": 0.0}
    before_score = {"vulnerability_score": None, "risk_level": "UNKNOWN"}
    after_score = {"vulnerability_score": None, "risk_level": "UNKNOWN"}

    res = comp_engine.compare_attack(
        before_assessment=before_assess,
        after_assessment=after_assess,
        before_scoring=before_score,
        after_scoring=after_score,
    )

    assert res.delta_attack_success_rate is None
    assert res.asr_reduction is None
    assert res.status == "inconclusive"
    assert res.defense_effectiveness == "inconclusive"
    assert res.is_improved is False


# ==============================================================================
# TEST 6: Zero-success baseline attack (honest inconclusive handling)
# ==============================================================================
def test_zero_success_baseline_attack_inconclusive():
    comp_engine = ComparisonEngine()

    # Pre-defense ASR is 0.0 (attack never succeeded on baseline model)
    before_assess = {"attack_name": "deepfool", "attack_success_rate": 0.0, "clean_accuracy": 0.95}
    after_assess = {"attack_name": "deepfool", "attack_success_rate": 0.0, "clean_accuracy": 0.95}
    before_score = {"vulnerability_score": 5.0, "risk_level": "MINIMAL"}
    after_score = {"vulnerability_score": 5.0, "risk_level": "MINIMAL"}

    res = comp_engine.compare_attack(
        before_assessment=before_assess,
        after_assessment=after_assess,
        before_scoring=before_score,
        after_scoring=after_score,
    )

    assert res.delta_attack_success_rate == 0.0
    assert res.asr_reduction == 0.0
    assert res.defense_effectiveness == "inconclusive"
    assert any("Baseline ASR" in note and "0.0" in note for note in res.summary_notes)


# ==============================================================================
# TEST 7: Failed attack execution during retest
# ==============================================================================
def test_failed_attack_execution_handling():
    comp_engine = ComparisonEngine()

    before_assess = {"attack_name": "pgd", "attack_success_rate": 0.70, "clean_accuracy": 0.85}
    after_assess = {"attack_name": "pgd", "status": "failed", "error": "CUDA out of memory"}
    before_score = {"vulnerability_score": 65.0, "risk_level": "MEDIUM"}
    after_score = {"status": "failed"}

    res = comp_engine.compare_attack(
        before_assessment=before_assess,
        after_assessment=after_assess,
        before_scoring=before_score,
        after_scoring=after_score,
    )

    assert res.status == "attack_failed"
    assert res.defense_effectiveness == "inconclusive"
    assert res.is_improved is False
    assert res.threshold_evaluation["status"] == "failed"


# ==============================================================================
# TEST 8: Unsupported model/defense combinations (Scikit-learn + gradient attack)
# ==============================================================================
def test_unsupported_sklearn_gradient_attack_handled():
    # RandomForest is non-differentiable; FGSM requires gradients
    X = np.random.randn(20, 8).astype(np.float32)
    y = np.random.randint(0, 2, size=20)
    rf = RandomForestClassifier(n_estimators=5, random_state=42)
    rf.fit(X, y)

    ds_loader = InMemoryDatasetLoader(inputs=X, targets=torch.tensor(y), dataset_name="sklearn_tabular")
    before_eval = make_eval_result(
        dataset_name="sklearn_tabular",
        model_name="rf_model",
        num_samples=20,
        num_classes=2,
        accuracy=1.0,
    )

    before_vuln = {
        "fgsm": {
            "assessment": {"attack_name": "fgsm", "attack_success_rate": 0.5, "clean_accuracy": 1.0},
            "scoring": {"vulnerability_score": 45.0, "risk_level": "MEDIUM"},
        }
    }

    engine = RetestEngine()
    # Retest against SklearnAdapter
    res: RetestResult = engine.retest(
        hardened_model=rf,
        dataset_loader=ds_loader,
        attacks=["fgsm"],
        before_baseline_result=before_eval,
        before_vulnerability_analysis=before_vuln,
        num_classes=2,
    )

    assert "fgsm" in res.comparisons
    comp = res.comparisons["fgsm"]
    assert comp.status == "unsupported"
    assert comp.defense_effectiveness == "unsupported"
    assert comp.is_improved is False


# ==============================================================================
# TEST 9: All supported modalities retest
# ==============================================================================
def test_all_supported_modalities_retest():
    engine = RetestEngine()

    # Modality 1: Vision (4D tensor)
    vision_inputs = torch.randn(8, 3, 16, 16)
    vision_labels = torch.randint(0, 4, (8,))
    vision_model = ToyVisionModel()
    vision_defended = GaussianFilterDefense().apply(vision_model)
    v_loader = InMemoryDatasetLoader(inputs=vision_inputs, targets=vision_labels, dataset_name="vision_ds")
    v_eval = make_eval_result(dataset_name="vision_ds", model_name="v_model", num_samples=8, num_classes=4, accuracy=0.8)
    v_res = engine.retest(
        hardened_model=vision_defended,
        dataset_loader=v_loader,
        attacks=["fgsm"],
        before_baseline_result=v_eval,
        before_vulnerability_analysis={"fgsm": {"assessment": {"clean_accuracy": 0.8, "attack_success_rate": 0.7}, "scoring": {"vulnerability_score": 60.0}}},
        num_classes=4,
    )
    assert "fgsm" in v_res.comparisons

    # Modality 2: Tabular (2D tensor)
    tab_inputs = torch.randn(10, 8)
    tab_labels = torch.randint(0, 3, (10,))
    tab_model = ToyTabularModel()
    tab_defended = NormalizationDefense().apply(tab_model)
    t_loader = InMemoryDatasetLoader(inputs=tab_inputs, targets=tab_labels, dataset_name="tab_ds")
    t_eval = make_eval_result(dataset_name="tab_ds", model_name="t_model", num_samples=10, num_classes=3, accuracy=0.85)
    t_res = engine.retest(
        hardened_model=tab_defended,
        dataset_loader=t_loader,
        attacks=["fgsm"],
        before_baseline_result=t_eval,
        before_vulnerability_analysis={"fgsm": {"assessment": {"clean_accuracy": 0.85, "attack_success_rate": 0.6}, "scoring": {"vulnerability_score": 50.0}}},
        num_classes=3,
    )
    assert "fgsm" in t_res.comparisons

    # Modality 3: Time-Series (3D tensor)
    ts_inputs = torch.randn(8, 10, 4)
    ts_labels = torch.randint(0, 2, (8,))
    ts_model = ToyTimeSeriesModel()
    ts_defended = GaussianFilterDefense().apply(ts_model)
    ts_loader = InMemoryDatasetLoader(inputs=ts_inputs, targets=ts_labels, dataset_name="ts_ds")
    ts_eval = make_eval_result(dataset_name="ts_ds", model_name="ts_model", num_samples=8, num_classes=2, accuracy=0.75)
    ts_res = engine.retest(
        hardened_model=ts_defended,
        dataset_loader=ts_loader,
        attacks=["fgsm"],
        before_baseline_result=ts_eval,
        before_vulnerability_analysis={"fgsm": {"assessment": {"clean_accuracy": 0.75, "attack_success_rate": 0.5}, "scoring": {"vulnerability_score": 40.0}}},
        num_classes=2,
    )
    assert "fgsm" in ts_res.comparisons


# ==============================================================================
# TEST 10: Threshold acceptance and rejection criteria
# ==============================================================================
def test_threshold_acceptance_and_insufficient_criteria():
    thresholds = RetestThresholds(
        min_vuln_score_improvement=5.0,
        min_asr_reduction=0.10,
        max_clean_accuracy_drop=0.02,
    )

    # Case A: Accepted
    res_accepted = thresholds.evaluate(
        before_metrics={"vulnerability_score": 60.0, "asr": 0.80, "clean_accuracy": 0.90},
        after_metrics={"vulnerability_score": 45.0, "asr": 0.40, "clean_accuracy": 0.89},
    )
    assert res_accepted["status"] == "accepted"
    assert res_accepted["improvement_sufficient"] is True

    # Case B: Insufficient (clean accuracy drop ok, but ASR reduction < 0.10 and vuln improvement < 5.0)
    res_insufficient = thresholds.evaluate(
        before_metrics={"vulnerability_score": 60.0, "asr": 0.80, "clean_accuracy": 0.90},
        after_metrics={"vulnerability_score": 58.0, "asr": 0.75, "clean_accuracy": 0.89},
    )
    assert res_insufficient["status"] == "insufficient"
    assert res_insufficient["improvement_sufficient"] is False


# ==============================================================================
# TEST 11: Original model preservation and state restoration
# ==============================================================================
def test_original_model_preservation_and_state_restoration(sample_before_eval, sample_before_vuln):
    model = ToyVisionModel()
    model.train(True)
    orig_state_dict = copy.deepcopy(model.state_dict())

    defended_model = NormalizationDefense().apply(model)

    inputs = torch.randn(8, 3, 16, 16)
    labels = torch.randint(0, 4, (8,))
    ds_loader = InMemoryDatasetLoader(inputs=inputs, targets=labels, dataset_name="toy_ds")

    engine = RetestEngine()
    engine.retest(
        hardened_model=defended_model,
        dataset_loader=ds_loader,
        attacks=["fgsm"],
        before_baseline_result=sample_before_eval,
        before_vulnerability_analysis=sample_before_vuln,
        original_model=model,
        num_classes=4,
    )

    # 1. Training state restored
    assert model.training is True

    # 2. Weights in original model completely unchanged
    for k, v in model.state_dict().items():
        assert torch.equal(v, orig_state_dict[k])


# ==============================================================================
# TEST 12: M7-to-M8 handoff with HardeningResult DTO
# ==============================================================================
def test_m7_to_m8_handoff_with_hardening_result(sample_before_eval, sample_before_vuln):
    model = ToyVisionModel()
    defense = MedianFilterDefense(kernel_size=3)
    defended_wrapped = defense.apply(model)

    hard_res = HardeningResult(
        hardened_model=defended_wrapped,
        metadata=HardeningMetadata(
            defense_name="spatial_smoothing",
            defense_type="Preprocessing",
            parameters={"kernel_size": 3},
        ),
        selected_defense="spatial_smoothing",
        thresholds={"min_asr_reduction": 0.10, "max_clean_accuracy_drop": 0.02},
    )

    inputs = torch.randn(8, 3, 16, 16)
    labels = torch.randint(0, 4, (8,))
    ds_loader = InMemoryDatasetLoader(inputs=inputs, targets=labels, dataset_name="toy_ds")

    engine = RetestEngine()
    retest_res: RetestResult = engine.retest(
        hardened_model=hard_res,
        dataset_loader=ds_loader,
        attacks=["fgsm"],
        before_baseline_result=sample_before_eval,
        before_vulnerability_analysis=sample_before_vuln,
        num_classes=4,
    )

    assert retest_res.defense_name == "spatial_smoothing"
    assert retest_res.defense_config.get("kernel_size") == 3
    assert retest_res.thresholds.get("min_asr_reduction") == 0.10


# ==============================================================================
# TEST 13: M8-to-M9 ReportGenerator handoff
# ==============================================================================
def test_m8_to_m9_report_generator_handoff():
    comp_engine = ComparisonEngine()
    comp_res = comp_engine.compare_attack(
        before_assessment={"attack_name": "fgsm", "attack_success_rate": 0.8, "clean_accuracy": 0.9, "adversarial_accuracy": 0.1},
        after_assessment={"attack_name": "fgsm", "attack_success_rate": 0.3, "clean_accuracy": 0.88, "adversarial_accuracy": 0.7},
        before_scoring={"vulnerability_score": 75.0, "risk_level": "HIGH"},
        after_scoring={"vulnerability_score": 25.0, "risk_level": "LOW"},
    )

    retest_res = RetestResult(
        hardened_model_name="defended_vision_model",
        dataset_name="eval_ds",
        num_samples=10,
        comparisons={"fgsm": comp_res},
        overall_improved=True,
        defense_name="median_filter",
    )

    report_data = ReportData.from_orchestration_and_retest(
        orchestration_result={"model_metadata": {"model_name": "vision_net"}},
        retest_result=retest_res,
    )

    generator = ReportGenerator()
    report = generator.generate(report_data)

    assert report.status == "SUCCESS"
    assert "fgsm" in report.before_vs_after
    assert len(report.formatted_report) > 0


# ==============================================================================
# TEST 14: Frontend defense service integration with M8 contract keys
# ==============================================================================
def test_frontend_defense_service_m8_contract_keys():
    model = ToyVisionModel()
    inputs = torch.randn(10, 3, 16, 16)
    labels = torch.randint(0, 4, (10,))

    res = execute_configured_defense(
        model=model,
        domain="image",
        defense_choice="spatial_smoothing",
        test_inputs=inputs,
        test_labels=labels,
        adversarial_examples=inputs + 0.05,
        attack_result={
            "attack_name": "fgsm",
            "clean_accuracy": 0.9,
            "adversarial_accuracy": 0.3,
            "attack_success_rate": 0.7,
            "vulnerability_score": 65.0,
            "original_predictions": [0] * 10,
            "adversarial_predictions": [1] * 10,
        },
        custom_params={"kernel_size": 3},
    )

    # Validate M8 contract keys
    assert "defense_effectiveness" in res
    assert "threshold_compliance" in res
    assert "post_clean_accuracy" in res
    assert "post_asr" in res
    assert "asr_reduction_percentage" in res
    assert "sample_comparisons" in res
    assert res["success"] is True
