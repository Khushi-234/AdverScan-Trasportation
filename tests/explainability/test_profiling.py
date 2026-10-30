"""
Unit tests for XAI profiling and query counting (Module 5 Step 1).
Validates explain_seconds timing, num_model_queries count, and ExplanationResult metadata storage.
"""

from typing import Any
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeClassifier

from app.explainability.explainer import XAIExplainer
from app.explainability.explanation_result import ExplanationResult
from app.explainability.modality import Modality
from app.explainability.profiling import (
    ProfileContext,
    ProfileStats,
    QueryCounter,
    profile_explainer_call,
)
from app.explainability.techniques.gradient_explainer import GradientExplainer
from app.ingestion.adapters.pytorch_adapter import PyTorchAdapter


class SimpleClassifier(nn.Module):
    """Simple linear neural network for testing."""

    def __init__(self, in_features: int = 4, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MockExplainer:
    """Mock explainer that invokes model a specified number of times."""

    def __init__(self, num_queries_to_make: int = 3):
        self.num_queries_to_make = num_queries_to_make

    def explain(self, model: nn.Module, inputs: torch.Tensor, **kwargs: Any) -> dict:
        for _ in range(self.num_queries_to_make):
            _ = model(inputs)
        return {
            "status": "success",
            "executed": True,
            "technique": "mock",
            "attribution": inputs.detach().cpu().numpy(),
        }


def test_profile_stats_dataclass():
    """Validate ProfileStats initialization and dict serialization."""
    stats = ProfileStats(explain_seconds=0.123, num_model_queries=5, num_samples=20)
    d = stats.to_dict()
    assert d["explain_seconds"] == 0.123
    assert d["num_model_queries"] == 5
    assert d["num_samples"] == 20


def test_query_counter_pytorch():
    """Verify QueryCounter tracks forward passes and sample counts on PyTorch models."""
    model = SimpleClassifier(in_features=4, num_classes=2)

    with QueryCounter(model) as counter:
        assert counter.query_count == 0
        x1 = torch.randn(2, 4)
        _ = model(x1)
        assert counter.query_count == 1
        assert counter.sample_count == 2

        x2 = torch.randn(5, 4)
        _ = model(x2)
        assert counter.query_count == 2
        assert counter.sample_count == 7

    # Ensure forward method is fully restored after context exit
    _ = model(torch.randn(1, 4))
    assert counter.query_count == 2


def test_query_counter_sklearn():
    """Verify QueryCounter tracks predict_proba and predict calls on sklearn models."""
    X = np.random.randn(25, 4)
    y = np.random.randint(0, 2, size=25)
    model = DecisionTreeClassifier(random_state=42).fit(X, y)

    with QueryCounter(model) as counter:
        assert counter.query_count == 0
        _ = model.predict_proba(X[:3])
        assert counter.query_count == 1
        assert counter.sample_count == 3

        _ = model.predict(X[:2])
        # Reentrancy guard should prevent inner predict_proba from double counting
        assert counter.query_count == 2
        assert counter.sample_count == 5


def test_query_counter_pytorch_adapter():
    """Verify QueryCounter instruments PyTorchAdapter and its wrapped nn.Module."""
    raw_model = SimpleClassifier(in_features=4, num_classes=2)
    adapter = PyTorchAdapter(raw_model)

    with QueryCounter(adapter) as counter:
        _ = adapter.predict(np.random.randn(2, 4).astype(np.float32))
        assert counter.query_count == 1

        _ = adapter(torch.randn(3, 4))
        assert counter.query_count == 2


def test_profile_explainer_call():
    """Verify profile_explainer_call wraps explain() and captures elapsed time and queries."""
    model = SimpleClassifier(in_features=4, num_classes=2)
    mock = MockExplainer(num_queries_to_make=4)
    inp = torch.randn(1, 4)

    res, stats = profile_explainer_call(mock, model, inp)
    assert res["status"] == "success"
    assert stats.num_model_queries == 4
    assert stats.explain_seconds >= 0.0
    assert res["explain_seconds"] == stats.explain_seconds
    assert res["num_model_queries"] == 4


def test_xai_explainer_metadata_timing_and_query_count():
    """
    Test Step 1 requirement:
    Wrap each explainer.explain(...) call, and store explain_seconds and num_model_queries in ExplanationResult.metadata.
    """
    model = SimpleClassifier(in_features=4, num_classes=2)
    clean_sample = torch.randn(1, 4)
    adv_sample = clean_sample + 0.05 * torch.randn_like(clean_sample)

    explainer = XAIExplainer(default_technique="saliency")
    res = explainer.explain(
        model=model,
        clean_input=clean_sample,
        adversarial_input=adv_sample,
        technique="saliency",
        modality=Modality.TABULAR,
    )

    assert isinstance(res, ExplanationResult)
    # Check that explain_seconds and num_model_queries exist in metadata
    assert "explain_seconds" in res.metadata, "explain_seconds must be in ExplanationResult.metadata"
    assert "num_model_queries" in res.metadata, "num_model_queries must be in ExplanationResult.metadata"

    assert isinstance(res.metadata["explain_seconds"], (float, int))
    assert isinstance(res.metadata["num_model_queries"], int)
    assert res.metadata["explain_seconds"] >= 0.0
    # Clean explain + Adversarial explain saliency each query the model at least once
    assert res.metadata["num_model_queries"] >= 2

    # Verify per-input breakdown metadata
    assert "clean_explain_seconds" in res.metadata
    assert "clean_num_model_queries" in res.metadata
    assert "adversarial_explain_seconds" in res.metadata
    assert "adversarial_num_model_queries" in res.metadata

    assert res.metadata["clean_num_model_queries"] >= 1
    assert res.metadata["adversarial_num_model_queries"] >= 1


def test_explanation_result_save_json_preserves_profiling(tmp_path: Path):
    """Verify that ExplanationResult.save_json serializes profiling metadata."""
    model = SimpleClassifier(in_features=4, num_classes=2)
    clean_sample = torch.randn(1, 4)
    adv_sample = clean_sample + 0.01

    explainer = XAIExplainer(default_technique="saliency")
    res = explainer.explain(
        model=model,
        clean_input=clean_sample,
        adversarial_input=adv_sample,
        technique="saliency",
        modality=Modality.TABULAR,
    )

    out_json = tmp_path / "explanation_profiling.json"
    res.save_json(out_json, store_npz=False)

    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    assert "explain_seconds" in meta
    assert "num_model_queries" in meta
    assert meta["num_model_queries"] >= 2
