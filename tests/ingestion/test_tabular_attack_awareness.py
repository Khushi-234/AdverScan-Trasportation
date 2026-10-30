"""
Acceptance tests for Attack-Aware Tabular Model Ingestion in AdverScan (Prompt 2).
Tests:
  Test 1 — Numerical + categorical feature types (age, income, gender)
  Test 2 — Binary feature handling (has_card in {0, 1})
  Test 3 — Immutable feature preservation (freezes immutable columns during attack projection)
  Test 4 — Encoded data (categorical encoding not treated as continuous)
  Test 5 — Scaling awareness (model-space perturbations translated to raw space)
  Test 6 — Gradient capabilities (PyTorch -> gradients available, RandomForest -> unavailable)
  Test 7 — Attack compatibility (gradient attacks rejected on non-gradient models)
"""

from typing import Any
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from app.ingestion import (
    AdapterCapabilities,
    AttackCompatibility,
    BaseModelAdapter,
    IngestionResult,
    ModelMetadata,
    PyTorchAdapter,
    TabularMetadata,
    TabularProcessor,
    ingest_model,
)


class NonGradientAdapter(BaseModelAdapter):
    """Standardized adapter wrapping non-differentiable / tree-based models (e.g. sklearn RandomForest)."""

    def __init__(self, model: Any):
        self._model = model

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            supports_gradients=False,
            supports_predict=True,
            supports_batch_predict=True,
            supports_logits=True,
        )

    def predict(self, inputs: Any, **kwargs: Any) -> Any:
        if hasattr(inputs, "detach"):
            inputs = inputs.detach().cpu().numpy()
        return self._model.predict_proba(inputs) if hasattr(self._model, "predict_proba") else self._model.predict(inputs)

    def get_model(self) -> Any:
        return self._model

    def to(self, device: Any) -> "NonGradientAdapter":
        return self

    def eval(self) -> "NonGradientAdapter":
        return self

    def train(self, mode: bool = True) -> "NonGradientAdapter":
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.predict(*args, **kwargs)


class DummyTabularNet(nn.Module):
    """PyTorch MLP for tabular inference."""
    def __init__(self, in_features: int = 3, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ---------------------------------------------------------------------------
# Test 1 — Numerical + categorical feature types (Requirement 2 & Acceptance Test 1)
# ---------------------------------------------------------------------------

def test_numerical_and_categorical_feature_types():
    """
    Test 1: Verify correct inference and distinction of numerical and categorical features
    (e.g., age, income, gender).
    """
    df = pd.DataFrame({
        "age": [25, 40, 65, 30, 50] * 5,
        "income": [45000.0, 72000.0, 115000.0, 52000.0, 88000.0] * 5,
        "gender": ["M", "F", "M", "F", "Other"] * 5,
    })

    proc = TabularProcessor(sample_data=df)

    assert "age" in proc.numerical_features or proc.feature_types["age"] == "numerical"
    assert "income" in proc.numerical_features or proc.feature_types["income"] == "numerical"
    assert "gender" in proc.categorical_features or proc.feature_types["gender"] == "categorical"

    # Verify attack space reflects correct continuity
    assert proc.attack_space["income"]["is_continuous"] is True
    assert proc.attack_space["gender"]["is_continuous"] is False
    assert proc.attack_space["gender"]["perturbation_mode"] == "category_swap"


# ---------------------------------------------------------------------------
# Test 2 — Binary feature handling (Requirement 2 & Acceptance Test 2)
# ---------------------------------------------------------------------------

def test_binary_feature_handling():
    """
    Test 2: Verify binary feature has_card in {0, 1} is identified as binary
    and constrained to discrete values rather than continuous floats.
    """
    df = pd.DataFrame({
        "has_card": [0, 1, 0, 1] * 10,
        "balance": [150.0, 2400.0, 0.0, 560.0] * 10,
    })

    proc = TabularProcessor(sample_data=df)

    assert "has_card" in proc.binary_features
    assert proc.feature_types["has_card"] == "binary"
    assert proc.attack_space["has_card"]["is_continuous"] is False
    assert proc.attack_space["has_card"]["perturbation_mode"] == "flip"

    # Verify binary projection prevents float perturbation like 0.37
    perturbed_df = pd.DataFrame([{"has_card": 0.37, "balance": 200.0}])
    projected = proc.project_to_constraints(perturbed_df)
    assert projected["has_card"].iloc[0] == 0.0  # Snapped to 0 rather than float 0.37

    perturbed_df_high = pd.DataFrame([{"has_card": 0.85, "balance": 200.0}])
    projected_high = proc.project_to_constraints(perturbed_df_high)
    assert projected_high["has_card"].iloc[0] == 1.0


# ---------------------------------------------------------------------------
# Test 3 — Immutable feature preservation (Requirement 3 & Acceptance Test 3)
# ---------------------------------------------------------------------------

def test_immutable_feature_preservation():
    """
    Test 3: Configure one feature as immutable (e.g. account_id) and verify the constraint
    is preserved and enforced.
    """
    proc = TabularProcessor(
        feature_names=["account_id", "age", "balance"],
        feature_types={"account_id": "numerical", "age": "numerical", "balance": "numerical"},
        immutable_features=["account_id"],
    )

    assert proc.is_mutable("account_id") is False
    assert proc.is_mutable("age") is True
    assert "account_id" in proc.immutable_features
    assert proc.attack_space["account_id"]["mutable"] is False

    original = pd.DataFrame([{"account_id": 998877, "age": 32, "balance": 5000.0}])
    perturbed = pd.DataFrame([{"account_id": 123456, "age": 35, "balance": 4500.0}])

    # 1. Validation detects immutable violation
    is_valid, violations = proc.validate_constraints(perturbed, reference_sample=original)
    assert is_valid is False
    assert any("account_id" in v and "Immutable" in v for v in violations)

    # 2. Projection freezes immutable feature to original value
    projected = proc.project_to_constraints(perturbed, reference_sample=original)
    assert projected["account_id"].iloc[0] == 998877
    assert projected["age"].iloc[0] == 35


# ---------------------------------------------------------------------------
# Test 4 — Encoded categorical data awareness (Requirement 6 & Acceptance Test 4)
# ---------------------------------------------------------------------------

def test_encoded_data_not_treated_as_continuous():
    """
    Test 4: Verify categorical encoding (e.g. integer-encoded colors or label-encoded gender)
    is NOT incorrectly treated as continuous numerical values in the attack space.
    """
    proc = TabularProcessor(
        feature_names=["color_code", "income"],
        feature_types={"color_code": "categorical", "income": "numerical"},
        categorical_encoding_type="label",
        feature_constraints={
            "color_code": {
                "allowed_values": [0, 1, 2],  # red=0, green=1, blue=2
                "is_continuous": False,
            }
        },
    )

    # color_code must NOT be treated as continuous
    assert proc.attack_space["color_code"]["is_continuous"] is False
    assert proc.attack_space["color_code"]["type"] == "categorical"
    assert proc.attack_space["income"]["is_continuous"] is True


# ---------------------------------------------------------------------------
# Test 5 — Scaling and perturbation interpretation (Requirement 5 & Acceptance Test 5)
# ---------------------------------------------------------------------------

def test_scaling_and_perturbation_interpretation():
    """
    Test 5: Verify model-space perturbations (e.g. +0.1 in scaled space) can be
    correctly interpreted through the stored transformation back to original feature space.
    """
    # Suppose raw dataset has mean=50000, scale=20000 for income, and mean=35, scale=15 for age
    scaler = StandardScaler()
    scaler.mean_ = np.array([35.0, 50000.0], dtype=np.float32)
    scaler.scale_ = np.array([15.0, 20000.0], dtype=np.float32)
    scaler.var_ = scaler.scale_ ** 2

    proc = TabularProcessor(
        feature_names=["age", "income"],
        feature_types={"age": "numerical", "income": "numerical"},
        preprocessor=scaler,
    )

    raw_sample = pd.DataFrame([{"age": 35.0, "income": 50000.0}])

    # Adversarial perturbation in model space: [0.0, +0.1]
    # In model space: age=0.0 (raw=35.0), income=0.1 (raw = 50000 + 0.1 * 20000 = 52000)
    model_perturbed = np.array([[0.0, 0.1]], dtype=np.float32)

    report = proc.interpret_perturbation(raw_original=raw_sample, model_perturbed=model_perturbed)

    assert report["representation_space"] == "scaled"
    assert report["has_inverse_transform"] is True

    # Raw delta must be 2000.0 in dollars, NOT 0.1!
    income_delta = report["raw_deltas"]["income"]["delta"]
    assert abs(income_delta - 2000.0) < 1e-3
    assert abs(report["raw_perturbed"]["income"] - 52000.0) < 1e-3
    assert abs(report["raw_deltas"]["age"]["delta"]) < 1e-3


# ---------------------------------------------------------------------------
# Test 6 — Gradient capability detection (Requirement 8 & Acceptance Test 6)
# ---------------------------------------------------------------------------

def test_gradient_capability_detection():
    """
    Test 6: Verify:
      PyTorch -> gradients available (supports_gradients=True)
      RandomForest / non-gradient model -> gradients unavailable (supports_gradients=False)
    """
    pt_model = DummyTabularNet(in_features=3, num_classes=2)
    pt_adapter = PyTorchAdapter(pt_model)
    assert pt_adapter.capabilities.supports_gradients is True
    assert pt_adapter.capabilities.supports_predict is True

    rf_model = RandomForestClassifier(n_estimators=5, random_state=42)
    rf_adapter = NonGradientAdapter(rf_model)
    assert rf_adapter.capabilities.supports_gradients is False
    assert rf_adapter.capabilities.supports_predict is True


# ---------------------------------------------------------------------------
# Test 7 — Attack compatibility verification (Requirement 9 & Acceptance Test 7)
# ---------------------------------------------------------------------------

def test_attack_compatibility_verification():
    """
    Test 7: Verify an attack requiring gradients (e.g. PGD, FGSM, C&W) is NOT reported
    as compatible with a non-gradient model.
    """
    proc = TabularProcessor(
        feature_names=["f1", "f2", "f3"],
        feature_types={"f1": "numerical", "f2": "numerical", "f3": "numerical"},
    )

    pt_model = DummyTabularNet(in_features=3, num_classes=2)
    pt_adapter = PyTorchAdapter(pt_model)

    rf_model = RandomForestClassifier(n_estimators=5, random_state=42)
    rf_adapter = NonGradientAdapter(rf_model)

    # 1. PGD on PyTorch -> Compatible
    pt_pgd = proc.check_attack_compatibility("PGD", adapter=pt_adapter)
    assert pt_pgd.compatible is True
    assert pt_pgd.attack_name == "PGD"

    # 2. PGD on RandomForest -> Incompatible due to lack of gradients
    rf_pgd = proc.check_attack_compatibility("PGD", adapter=rf_adapter)
    assert rf_pgd.compatible is False
    assert "requires gradient" in rf_pgd.reason.lower()

    # 3. C&W on RandomForest -> Incompatible
    rf_cw = proc.check_attack_compatibility("C&W", adapter=rf_adapter)
    assert rf_cw.compatible is False

    # 4. HopSkipJump (black-box query attack) on RandomForest -> Compatible
    rf_hsj = proc.check_attack_compatibility("HopSkipJump", adapter=rf_adapter)
    assert rf_hsj.compatible is True


# ---------------------------------------------------------------------------
# End-to-End Pipeline Integration with Attack-Aware Metadata
# ---------------------------------------------------------------------------

def test_pipeline_attack_aware_metadata_integration():
    """Verify ingest_model correctly passes constraints and mutability into IngestionResult."""
    model = DummyTabularNet(in_features=3, num_classes=2)
    sample = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0], "f3": [5.0, 6.0]})

    result = ingest_model(
        model_path=model,
        sample_input=sample,
        data_domain="tabular",
        task_type="classification",
        feature_names=["f1", "f2", "f3"],
        immutable_features=["f1"],
        feature_constraints={
            "f2": {"type": "numerical", "min": 0.0, "max": 10.0, "mutable": True},
        },
        device="cpu",
    )

    assert isinstance(result, IngestionResult)
    adapter, meta = result
    assert isinstance(result.processor, TabularProcessor)
    proc = result.processor

    # Constraints preserved on processor and metadata
    assert proc.is_mutable("f1") is False
    assert proc.is_mutable("f2") is True
    assert meta.tabular_metadata.get_feature_constraint("f2")["max"] == 10.0
    assert "f1" in meta.tabular_metadata.immutable_features

    # Check attack compatibility through processor
    compat = proc.check_attack_compatibility("CAP", adapter=adapter)
    assert compat.compatible is True
