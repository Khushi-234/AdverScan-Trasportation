"""
Unit and integration tests for Tabular model and dataset ingestion in AdverScan.
Verifies tabular classification model ingestion, TabularProcessor functionality,
feature-type categorization (numerical, categorical, binary, ordinal), target handling,
class mapping preservation, preprocessing awareness, raw <-> model space mappings,
input validation, and LoadResult integration.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from app.ingestion import (
    IngestionResult,
    LoadResult,
    ModelMetadata,
    ModelValidationError,
    PyTorchAdapter,
    PyTorchLoader,
    TabularMetadata,
    TabularProcessor,
    UnsupportedDomainError,
    ingest_model,
)
from app.ingestion.model_registry import detect_model_domain
from app.ingestion.validation import ModelValidator


# ---------------------------------------------------------------------------
# Test Tabular Neural Network Architecture (PyTorch MLP)
# ---------------------------------------------------------------------------

class TabularMLPClassifier(nn.Module):
    """
    Standard multi-layer perceptron (MLP) for tabular classification tasks.
    Maps [batch_size, num_features] -> [batch_size, num_classes].
    """

    def __init__(self, num_features: int = 4, num_classes: int = 3, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 1. Tabular Metadata and Feature Type Inference Tests
# ---------------------------------------------------------------------------

def test_tabular_metadata_structure_and_serialization():
    """Verify TabularMetadata preserves required fields and serialization."""
    meta = TabularMetadata(
        feature_names=["age", "income", "gender", "education_level"],
        feature_types={
            "age": "numerical",
            "income": "numerical",
            "gender": "binary",
            "education_level": "ordinal",
        },
        target_column="churn",
        num_classes=2,
        class_mapping={"no_churn": 0, "churn": 1},
    )

    assert meta.num_features == 4
    assert meta.numerical_features == ["age", "income"]
    assert meta.binary_features == ["gender"]
    assert meta.ordinal_features == ["education_level"]
    assert meta.index_to_class == {0: "no_churn", 1: "churn"}

    d = meta.to_dict()
    assert d["num_features"] == 4
    assert d["target_column"] == "churn"

    restored = TabularMetadata.from_dict(d)
    assert restored.feature_names == meta.feature_names
    assert restored.feature_types == meta.feature_types


def test_feature_type_inference_heuristics_and_overrides():
    """
    Verify feature-type inference distinguishes numerical, binary, categorical, ordinal,
    does NOT assume integers are numerical, and respects explicit overrides.
    """
    df = pd.DataFrame({
        "continuous_float": [1.2, 3.4, 5.6, 7.8, 9.0] * 10,
        "binary_flag": [0, 1, 0, 1, 1] * 10,
        "string_category": ["cat", "dog", "bird", "fish", "cat"] * 10,
        "integer_low_cardinality": [1, 2, 3, 2, 1] * 10,  # Should NOT automatically be numerical
        "ordinal_rating": [1, 2, 3, 4, 5] * 10,
    })

    # Auto inference with explicit override on ordinal_rating
    inferred = TabularMetadata.infer_feature_types(
        df,
        ordinal_overrides=["ordinal_rating"],
        cardinality_threshold=5,
    )

    assert inferred["feature_types"]["continuous_float"] == "numerical"
    assert inferred["feature_types"]["binary_flag"] == "binary"
    assert inferred["feature_types"]["string_category"] == "categorical"
    assert inferred["feature_types"]["integer_low_cardinality"] == "categorical"
    assert inferred["feature_types"]["ordinal_rating"] == "ordinal"

    assert "continuous_float" in inferred["numerical_features"]
    assert "binary_flag" in inferred["binary_features"]
    assert "string_category" in inferred["categorical_features"]
    assert "ordinal_rating" in inferred["ordinal_features"]


# ---------------------------------------------------------------------------
# 2. Dataset and Target Handling Tests
# ---------------------------------------------------------------------------

def test_split_features_and_target_with_explicit_target():
    """Verify target separation with explicitly configured target_column."""
    df = pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0],
        "feat_b": [4.0, 5.0, 6.0],
        "label": ["A", "B", "A"],
    })

    proc = TabularProcessor(target_column="label", feature_names=["feat_a", "feat_b"])
    X, y = proc.split_features_and_target(df)

    assert list(X.columns) == ["feat_a", "feat_b"]
    assert y is not None
    assert list(y.values) == ["A", "B", "A"]
    assert proc.tabular_metadata.target_available is True


def test_split_features_and_target_without_target_column():
    """Verify that when target_column is not provided, the last column is NOT assumed to be target."""
    df = pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0],
        "feat_b": [4.0, 5.0, 6.0],
        "feat_c": [7.0, 8.0, 9.0],
    })

    proc = TabularProcessor(target_column=None)
    X, y = proc.split_features_and_target(df)

    assert y is None
    assert list(X.columns) == ["feat_a", "feat_b", "feat_c"]
    assert proc.tabular_metadata.target_available is False
    assert proc.tabular_metadata.target_status is not None
    assert "unavailable" in proc.tabular_metadata.target_status.lower()


def test_split_features_and_target_missing_column_raises():
    """Verify error when configured target_column does not exist."""
    df = pd.DataFrame({"feat_a": [1.0, 2.0]})
    proc = TabularProcessor(target_column="non_existent")

    with pytest.raises(ModelValidationError, match="not found in dataset columns"):
        proc.split_features_and_target(df)


# ---------------------------------------------------------------------------
# 3. Class Mapping Preservation Tests
# ---------------------------------------------------------------------------

def test_class_mapping_preservation_non_alphabetical():
    """Verify non-alphabetical class mapping is preserved faithfully."""
    # Suppose label 0 is "fraud" and 1 is "normal" (non-alphabetical)
    mapping = {"fraud": 0, "normal": 1}
    proc = TabularProcessor(
        feature_names=["f1", "f2"],
        class_mapping=mapping,
        num_classes=2,
    )

    assert proc.tabular_metadata.class_mapping == {"fraud": 0, "normal": 1}
    assert proc.tabular_metadata.index_to_class == {0: "fraud", 1: "normal"}


# ---------------------------------------------------------------------------
# 4. Preprocessing and Raw <-> Model Space Tests
# ---------------------------------------------------------------------------

def test_tabular_processor_raw_and_model_space_transformations():
    """
    Verify bidirectional mapping between raw space and model input space
    using a fitted StandardScaler.
    """
    raw_data = np.array([
        [10.0, 100.0],
        [20.0, 200.0],
        [30.0, 300.0],
    ], dtype=np.float32)

    scaler = StandardScaler()
    scaler.fit(raw_data)

    proc = TabularProcessor(
        feature_names=["feat_1", "feat_2"],
        preprocessor=scaler,
    )

    # 1. Preprocessing metadata inspection
    assert proc.preprocessing_info["status"] == "present"
    assert proc.preprocessing_info["type"] == "StandardScaler"
    assert "mean" in proc.preprocessing_info
    assert "scale" in proc.preprocessing_info

    # 2. Raw space -> Model space
    model_space = proc.to_model_space(raw_data)
    assert isinstance(model_space, np.ndarray)
    np.testing.assert_allclose(model_space.mean(axis=0), [0.0, 0.0], atol=1e-5)

    # 3. Model space -> Raw space
    reconstructed_raw = proc.to_raw_space(model_space)
    if isinstance(reconstructed_raw, pd.DataFrame):
        reconstructed_vals = reconstructed_raw.values
    else:
        reconstructed_vals = reconstructed_raw

    np.testing.assert_allclose(np.asarray(reconstructed_vals, dtype=np.float32), raw_data, atol=1e-5)


def test_tabular_processor_without_preprocessor_reports_cleanly():
    """Verify that when no preprocessor is configured, it reports clearly instead of fabricating one."""
    proc = TabularProcessor(feature_names=["a", "b"])
    assert proc.preprocessing_info["status"] == "none"
    assert "No external preprocessing pipeline configured" in proc.preprocessing_info["description"]


# ---------------------------------------------------------------------------
# 5. Model Input Validation Tests
# ---------------------------------------------------------------------------

def test_validate_model_input_feature_mismatch_raises():
    """Verify ModelValidationError when sample input has feature count mismatch."""
    model = TabularMLPClassifier(num_features=4, num_classes=3)
    adapter = PyTorchAdapter(model)
    proc = TabularProcessor(num_features=4, num_classes=3)

    # 3 features instead of 4
    bad_sample = torch.randn(2, 3)
    with pytest.raises(ModelValidationError, match="Feature count mismatch"):
        proc.validate_model_input(adapter, sample_input=bad_sample)


def test_validate_model_input_class_count_mismatch_raises():
    """Verify ModelValidationError when model output class dimension does not match metadata."""
    model = TabularMLPClassifier(num_features=4, num_classes=2)  # returns 2 classes
    adapter = PyTorchAdapter(model)
    proc = TabularProcessor(num_features=4, num_classes=3)  # expects 3 classes

    sample = torch.randn(2, 4)
    with pytest.raises(ModelValidationError, match="Class count mismatch"):
        proc.validate_model_input(adapter, sample_input=sample)


# ---------------------------------------------------------------------------
# 6. LoadResult Integration Tests
# ---------------------------------------------------------------------------

def test_load_result_dataclass():
    """Verify LoadResult provides standardized loading and inspection representation."""
    model = TabularMLPClassifier(num_features=4, num_classes=3)
    result = LoadResult(
        model=model,
        source="local",
        framework="pytorch",
        evidence=[
            "Detected in-memory PyTorch nn.Module",
            "Model architecture TabularMLPClassifier",
        ],
        domain_hint="tabular",
        task_hint="classification",
    )

    assert result.model is model
    assert result.source == "local"
    assert result.framework == "pytorch"
    assert result.domain_hint == "tabular"
    assert result.task_hint == "classification"
    assert len(result.evidence) == 2


# ---------------------------------------------------------------------------
# 7. End-to-End Pipeline Ingestion Acceptance Test (Requirement 11 & 12)
# ---------------------------------------------------------------------------

def test_end_to_end_tabular_ingestion_acceptance():
    """
    Acceptance test for tabular model ingestion:
    1. Model loads successfully.
    2. Framework is detected correctly (pytorch / pytorch_local).
    3. Domain is 'tabular'.
    4. Task is 'classification'.
    5. PyTorchAdapter is used.
    6. TabularProcessor is actually invoked.
    7. Feature metadata is generated.
    8. Target metadata is correct.
    9. Class mapping is preserved.
    10. Model input shape is validated.
    11. IngestionResult container exposes adapter, metadata, and processor.
    """
    # Real dataset: Iris (4 features, 3 classes)
    iris_obj: Any = load_iris()
    X = pd.DataFrame(iris_obj.data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    y = pd.Series(iris_obj.target, name="species")
    df = pd.concat([X, y], axis=1)

    # Real Tabular MLP classification model
    model = TabularMLPClassifier(num_features=4, num_classes=3, hidden_dim=8)
    model.eval()

    # Pre-fitted scaler for realistic tabular pipeline
    scaler = MinMaxScaler()
    scaler.fit(X.values)

    class_map = {"setosa": 0, "versicolor": 1, "virginica": 2}

    sample_input = X.iloc[:5]

    # Run full ingestion pipeline
    result = ingest_model(
        model_path=model,
        sample_input=sample_input,
        data_domain="tabular",
        task_type="classification",
        framework="pytorch",
        feature_names=list(X.columns),
        numerical_features=list(X.columns),
        target_column="species",
        num_classes=3,
        class_mapping=class_map,
        preprocessor=scaler,
        device="cpu",
    )

    # 1. Return type validation
    assert isinstance(result, IngestionResult)
    adapter, meta = result  # Verify backward compatibility tuple unpacking

    # 2. Adapter validation
    assert isinstance(adapter, PyTorchAdapter)
    assert isinstance(result.adapter, PyTorchAdapter)

    # 3. Domain & Task validation
    assert meta.data_domain == "tabular"
    assert meta.task_type == "classification"
    assert meta.framework == "pytorch"

    # 4. Processor validation
    assert isinstance(result.processor, TabularProcessor)
    proc = result.processor
    assert proc.data_domain == "tabular"

    # 5. Metadata validation
    assert meta.num_classes == 3
    assert meta.input_shape == (5, 4) or meta.input_shape == (1, 4)
    assert meta.tabular_metadata is not None
    tab_meta = meta.tabular_metadata
    assert isinstance(tab_meta, TabularMetadata)
    assert tab_meta.feature_names == ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    assert tab_meta.num_features == 4
    assert tab_meta.num_classes == 3
    assert tab_meta.class_mapping == class_map
    assert tab_meta.target_column == "species"

    # 6. Preprocessing awareness
    assert tab_meta.preprocessing_info is not None
    assert tab_meta.preprocessing_info["status"] == "present"
    assert tab_meta.preprocessing_info["type"] == "MinMaxScaler"

    # 7. Predictions through adapter with processor
    preds = adapter.predict(sample_input)
    assert preds.shape == (5, 3)


def test_ingest_tabular_saved_checkpoint():
    """Verify ingestion of a saved tabular model checkpoint from disk."""
    model = TabularMLPClassifier(num_features=4, num_classes=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "tabular_mlp.pt"
        torch.save(model, model_path)

        sample = pd.DataFrame({
            "f1": [0.1, 0.2],
            "f2": [0.3, 0.4],
            "f3": [0.5, 0.6],
            "f4": [0.7, 0.8],
        })

        result = ingest_model(
            model_path=model_path,
            sample_input=sample,
            data_domain="tabular",
            task_type="classification",
            device="cpu",
        )

        adapter, meta = result
        assert isinstance(adapter, PyTorchAdapter)
        assert meta.data_domain == "tabular"
        assert meta.framework == "pytorch_local"
        assert meta.num_classes == 2
        assert isinstance(result.processor, TabularProcessor)
        assert result.processor.tabular_metadata.num_features == 4
