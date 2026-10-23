"""
End-to-end integration, security boundary, and edge-case tests for tabular model ingestion in AdverScan.
Tests real scikit-learn models (LogisticRegression, RandomForestClassifier) and PyTorch tabular MLP,
verifying complete flow from loading through framework detection, adaptation, metadata, constraints,
baseline prediction, security controls, and attack compatibility.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn

from app.ingestion.adapters.pytorch_adapter import PyTorchAdapter
from app.ingestion.adapters.sklearn_adapter import SklearnAdapter
from app.ingestion.exceptions import (
    ModelLoadError,
    ModelValidationError,
    UnsafeModelError,
)
from app.ingestion.loader.pytorch_loader import PyTorchLoader
from app.ingestion.loader.sklearn_loader import SklearnLoader
from app.ingestion.metadata.tabular_metadata import TabularMetadata
from app.ingestion.pipeline import IngestionResult, ingest_model
from app.ingestion.processors.tabular_processor import TabularProcessor


class PyTorchTabularMLP(nn.Module):
    """Real PyTorch tabular neural network for gradient-based attack testing."""

    def __init__(self, in_features: int = 4, hidden_dim: int = 16, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Section 10 — Real End-to-End Tests
# ---------------------------------------------------------------------------

def test_end_to_end_real_sklearn_logistic_regression():
    """
    Test 10.1: Complete path for a real scikit-learn LogisticRegression classifier.
    Verify: load -> framework detection -> SklearnAdapter -> TabularProcessor ->
            metadata -> constraints -> baseline prediction -> attack compatibility.
    """
    # 1. Prepare and fit real scikit-learn model
    np.random.seed(42)
    X = np.random.randn(50, 4).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    clf = LogisticRegression()
    clf.fit(X, y)

    sample_df = pd.DataFrame(X[:5], columns=["f1", "f2", "f3", "f4"])

    # 2. Ingest through standard AdverScan pipeline
    result = ingest_model(
        model_path=clf,
        sample_input=sample_df,
        data_domain="tabular",
        feature_names=["f1", "f2", "f3", "f4"],
        immutable_features=["f1"],
        feature_constraints={
            "f2": {"type": "numerical", "min": -5.0, "max": 5.0, "mutable": True},
        },
        device="cpu",
    )

    assert isinstance(result, IngestionResult)
    adapter, meta = result
    proc = result.processor

    # 3. Verify Framework Detection and Adapter
    assert isinstance(adapter, SklearnAdapter)
    assert meta.framework == "sklearn"
    assert meta.data_domain == "tabular"
    assert meta.num_classes == 2

    # 4. Verify Capabilities
    caps = adapter.capabilities
    assert caps.supports_gradients is False
    assert caps.supports_predict is True
    assert caps.supports_batch_predict is True
    assert caps.supports_logits is True

    # 5. Verify Processor & Metadata
    assert isinstance(proc, TabularProcessor)
    assert proc.is_mutable("f1") is False
    assert proc.is_mutable("f2") is True
    assert "f1" in proc.get_immutable_features()

    # 6. Verify Baseline Prediction
    # Standard forward pass (returns logits/log-probabilities tensor for evaluator compatibility)
    logits = adapter.predict(sample_df)
    assert logits.shape == (5, 2)

    # Calibrated probabilities
    probs = adapter.predict_proba(sample_df)
    assert probs.shape == (5, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)

    # Discrete classes
    classes = adapter.predict_classes(sample_df)
    assert len(classes) == 5
    assert set(classes).issubset({0, 1})

    # Batch prediction
    batch_preds = adapter.predict_batch(sample_df)
    assert batch_preds.shape == (5, 2)

    # 7. Verify Attack Compatibility
    # Gradient attacks (PGD, JSMA, C&W) must be incompatible
    pgd_check = proc.check_attack_compatibility(adapter, "PGD")
    assert pgd_check.compatible is False
    assert "does not support gradients" in pgd_check.reason

    # Black-box attacks (HopSkipJump, Boundary) must be compatible
    hsj_check = proc.check_attack_compatibility(adapter, "HopSkipJump")
    assert hsj_check.compatible is True


def test_end_to_end_real_sklearn_random_forest():
    """
    Test 10.2: Complete path for a real scikit-learn RandomForestClassifier.
    Verify non-gradient capability, decision probabilities, and constraint handling.
    """
    np.random.seed(42)
    X = np.random.randn(60, 3).astype(np.float32)
    y = np.random.choice([0, 1, 2], size=60)

    rf = RandomForestClassifier(n_estimators=5, random_state=42)
    rf.fit(X, y)

    sample_df = pd.DataFrame(X[:4], columns=["age", "income", "credit_score"])

    result = ingest_model(
        model_path=rf,
        sample_input=sample_df,
        data_domain="tabular",
        device="cpu",
    )

    adapter, meta = result
    proc = result.processor
    assert isinstance(proc, TabularProcessor)

    assert isinstance(adapter, SklearnAdapter)
    assert meta.framework == "sklearn"
    assert meta.num_classes == 3

    # Baseline predictions
    preds = adapter.predict(sample_df)
    assert preds.shape == (4, 3)

    classes = adapter.predict_classes(sample_df)
    assert len(classes) == 4

    # Capabilities
    assert adapter.capabilities.supports_gradients is False

    # Attack compatibility
    cw_check = proc.check_attack_compatibility(adapter, "C&W")
    assert cw_check.compatible is False


def test_end_to_end_real_pytorch_tabular_mlp():
    """
    Test 10.3: Complete path for a real PyTorch tabular MLP classifier.
    Verify gradient support, backward compatibility, and attack readiness.
    """
    model = PyTorchTabularMLP(in_features=4, hidden_dim=8, num_classes=2)
    sample_df = pd.DataFrame(np.random.randn(3, 4).astype(np.float32), columns=["c1", "c2", "c3", "c4"])

    result = ingest_model(
        model_path=model,
        sample_input=sample_df,
        data_domain="tabular",
        device="cpu",
    )

    adapter, meta = result
    proc = result.processor
    assert isinstance(proc, TabularProcessor)

    assert isinstance(adapter, PyTorchAdapter)
    assert meta.framework == "pytorch"
    assert meta.num_classes == 2
    assert adapter.capabilities.supports_gradients is True

    # Baseline prediction
    preds = adapter.predict(sample_df)
    assert preds.shape == (3, 2)

    # Attack compatibility: gradient attack should be runnable
    pgd_check = proc.check_attack_compatibility(adapter, "PGD")
    assert pgd_check.compatible is True


# ---------------------------------------------------------------------------
# Section 11 — Required Edge-Case Tests
# ---------------------------------------------------------------------------

def test_edge_case_a_feature_mismatch_raises_clear_validation_error():
    """
    Test 11.A: Model expects 10 features, input provides 8 features.
    Must raise clear validation error without silently truncating or reshaping.
    """
    # Train model on 10 features
    X_train = np.random.randn(20, 10)
    y_train = np.random.choice([0, 1], size=20)
    clf = LogisticRegression().fit(X_train, y_train)

    # Provide only 8 features in sample input
    sample_8 = pd.DataFrame(np.random.randn(2, 8), columns=[f"feat_{i}" for i in range(8)])

    with pytest.raises(ModelValidationError, match=r"(?i)model expects 10 features.*8"):
        ingest_model(
            model_path=clf,
            sample_input=sample_8,
            data_domain="tabular",
            device="cpu",
        )


def test_edge_case_a_feature_ordering_mismatch_raises_error():
    """
    Test 11.A (bis): DataFrame columns in different order than declared feature_names.
    Must raise ModelValidationError instead of silently reordering columns.
    """
    model = PyTorchTabularMLP(in_features=3, num_classes=2)
    sample = pd.DataFrame({"f2": [1.0], "f1": [2.0], "f3": [3.0]})

    with pytest.raises(ModelValidationError, match=r"(?i)feature ordering mismatch"):
        ingest_model(
            model_path=model,
            sample_input=sample,
            data_domain="tabular",
            feature_names=["f1", "f2", "f3"],
            device="cpu",
        )


def test_edge_case_b_unknown_preprocessing_does_not_invent_transforms():
    """
    Test 11.B: Verify the system does not invent a scaler or encoder when none exists.
    """
    df = pd.DataFrame({
        "age": [30.0, 45.0, 60.0],
        "income": [50000.0, 80000.0, 120000.0],
    })
    proc = TabularProcessor(sample_data=df)

    assert proc.preprocessing_info["status"] == "none"
    assert proc.representation_space == "raw"
    assert proc.categorical_encoding_type == "none"

    # Perturbation in raw space equals raw delta
    delta = proc.interpret_perturbation(
        raw_original={"age": 30.0, "income": 50000.0},
        model_adversarial=np.array([[31.0, 52000.0]]),
    )
    assert delta["delta_numerical"]["age"] == 1.0
    assert delta["delta_numerical"]["income"] == 2000.0


def test_edge_case_c_categorical_variables_not_treated_as_continuous():
    """
    Test 11.C: Categorical features must never be treated as continuous numerical features.
    """
    df = pd.DataFrame({
        "gender": ["M", "F", "Other", "M"] * 10,
        "education": [0, 1, 2, 0] * 10,  # Integer-encoded categories
        "income": [40000.0, 70000.0, 95000.0, 52000.0] * 10,
    })

    proc = TabularProcessor(
        sample_data=df,
        categorical_features=["gender", "education"],
        numerical_features=["income"],
    )

    # Both nominal string and integer-encoded categoricals must have is_continuous = False
    assert proc.attack_space["gender"]["is_continuous"] is False
    assert proc.attack_space["gender"]["perturbation_mode"] == "category_swap"

    assert proc.attack_space["education"]["is_continuous"] is False
    assert proc.attack_space["education"]["perturbation_mode"] == "category_swap"

    assert proc.attack_space["income"]["is_continuous"] is True
    assert proc.attack_space["income"]["perturbation_mode"] == "continuous"


def test_edge_case_d_immutable_feature_constraints_exposed_and_enforced():
    """
    Test 11.D: Attack constraints must expose immutable features and reject / revert mutations.
    """
    proc = TabularProcessor(
        feature_names=["id", "age", "balance"],
        feature_types={"id": "numerical", "age": "numerical", "balance": "numerical"},
        immutable_features=["id"],
        feature_constraints={
            "age": {"min": 18.0, "max": 100.0, "mutable": True},
        },
    )

    assert proc.is_mutable("id") is False
    assert proc.is_mutable("age") is True
    assert proc.get_immutable_features() == ["id"]

    # Validate constraint detection
    orig = {"id": 101, "age": 30.0, "balance": 1000.0}
    perturbed = {"id": 102, "age": 31.0, "balance": 1100.0}  # id modified!

    is_valid, violations = proc.validate_constraints(perturbed, reference_sample=orig)
    assert is_valid is False
    assert any("Immutable feature 'id' was modified" in v for v in violations)

    # Project to constraints should reset immutable feature
    projected = proc.project_to_constraints(pd.DataFrame([perturbed]), reference_sample=pd.DataFrame([orig]))
    assert projected["id"].iloc[0] == 101


def test_edge_case_e_non_gradient_model_marks_gradient_attacks_incompatible():
    """
    Test 11.E: Non-gradient model (e.g. RandomForest) must mark gradient attacks incompatible.
    """
    rf = RandomForestClassifier(n_estimators=3).fit([[1, 2], [3, 4]], [0, 1])
    adapter = SklearnAdapter(rf)
    proc = TabularProcessor(num_features=2, num_classes=2)

    # Gradient attacks must be incompatible
    for attack in ["PGD", "FGSM", "JSMA", "C&W", "BIM"]:
        res = proc.check_attack_compatibility(adapter, attack)
        assert res.compatible is False
        assert "does not support gradients" in res.reason

    # Non-gradient attacks must be compatible
    for attack in ["HopSkipJump", "Boundary", "RandomSearch"]:
        res = proc.check_attack_compatibility(adapter, attack)
        assert res.compatible is True


def test_edge_case_f_unsafe_model_file_deserialization_protection():
    """
    Test 11.F: Unsafe deserialization of .joblib / .pkl files is refused without explicit consent.
    """
    clf = LogisticRegression().fit([[1, 2], [3, 4]], [0, 1])

    with tempfile.TemporaryDirectory() as tmpdir:
        import joblib
        model_path = Path(tmpdir) / "model.joblib"
        joblib.dump(clf, model_path)

        loader = SklearnLoader()

        # 1. Default load without explicit consent must raise UnsafeModelError
        with pytest.raises(UnsafeModelError, match=r"(?i)Security risk: Refusing to deserialize"):
            loader.load(model_path, unsafe_allow_pickle=False)

        # 2. Pipeline without explicit consent must also raise UnsafeModelError
        with pytest.raises(UnsafeModelError, match=r"(?i)Security risk"):
            ingest_model(
                model_path=model_path,
                data_domain="tabular",
                unsafe_allow_pickle=False,
                device="cpu",
            )

        # 3. Explicit consent allows loading safely
        loaded = loader.load(model_path, unsafe_allow_pickle=True)
        assert hasattr(loaded, "predict")


def test_edge_case_g_unknown_architecture_state_dict_fails_cleanly():
    """
    Test 11.G: System fails clearly rather than returning a broken wrapper when
    architecture cannot be determined from a state_dict.
    """
    model = PyTorchTabularMLP()
    with tempfile.TemporaryDirectory() as tmpdir:
        weights_path = Path(tmpdir) / "weights.pt"
        torch.save(model.state_dict(), weights_path)

        loader = PyTorchLoader()

        with pytest.raises(ModelLoadError, match=r"(?i)State dictionary detected, but the model architecture could not be determined"):
            loader.load(weights_path, model_class=None)


def test_error_and_evidence_reporting_in_metadata():
    """
    Test 7: Verify detection failures and successes preserve structured audit evidence.
    """
    clf = LogisticRegression().fit([[1, 2, 3], [4, 5, 6]], [0, 1])
    sample = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["a", "b", "c"])

    result = ingest_model(
        model_path=clf,
        sample_input=sample,
        data_domain="tabular",
        device="cpu",
    )

    metadata = result.metadata
    assert "evidence" in metadata.extra_info
    evidence = metadata.extra_info["evidence"]

    # Check evidence details
    evidence_str = " ".join(evidence)
    assert "scikit-learn estimator" in evidence_str
    assert "predict()" in evidence_str
    assert "tabular" in evidence_str
    assert "capabilities" in metadata.extra_info
    assert metadata.extra_info["capabilities"]["supports_gradients"] is False


def test_registered_tabular_model_presets_and_attack_compatibility():
    """
    Verify both tabular model presets in MODEL_PRESETS can be ingested, validated,
    and checked for attack compatibility.
    """
    from app.ingestion.model_registry import ModelPresetRegistry

    assert ModelPresetRegistry.has_preset("tabular-mlp-adult-census")
    assert ModelPresetRegistry.has_preset("tabular-mlp-credit-fraud")

    # 1. Ingest Adult Census Tabular MLP
    res_adult = ingest_model("tabular-mlp-adult-census")
    assert res_adult.metadata.data_domain == "tabular"
    assert res_adult.metadata.framework == "pytorch"
    assert isinstance(res_adult.processor, TabularProcessor)
    assert res_adult.processor.num_features == 14
    compat_adult = res_adult.processor.check_attack_compatibility("PGD", res_adult.adapter)
    assert compat_adult.compatible is True

    # 2. Ingest Credit Card Fraud Tabular MLP
    res_credit = ingest_model("tabular-mlp-credit-fraud")
    assert res_credit.metadata.data_domain == "tabular"
    assert res_credit.metadata.framework == "pytorch"
    assert isinstance(res_credit.processor, TabularProcessor)
    assert res_credit.processor.num_features == 30
    compat_credit = res_credit.processor.check_attack_compatibility("PGD", res_credit.adapter)
    assert compat_credit.compatible is True

