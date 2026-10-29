"""
Unit and integration tests for Step 2 (XAIConfig) and Step 3 (predict_batched).
Verifies:
1. XAIConfig initialization, presets (fast, standard, thorough), serialization (to_dict, from_dict).
2. XAIExplainer wires XAIConfig into metadata, defaults, and explainer techniques.
3. predict_batched evaluates large inputs in mini-batch chunks using torch.inference_mode().
4. predict_batched supports text truncation with max_text_length and tokenizer.
5. predict_proba_np delegates to predict_batched with chunking.
"""

from typing import List
import numpy as np
import pytest
import torch
import torch.nn as nn

from app.explainability.config import XAIConfig
from app.explainability.explainer import XAIExplainer
from app.explainability.utils import configure_cpu_threads, predict_batched, predict_proba_np
from app.explainability.modality import Modality
from app.explainability.techniques.gradient_explainer import GradientExplainer
from app.explainability.techniques.lime_explainer import LIMEExplainer


# -------------------------------------------------------------------------
# Test Models
# -------------------------------------------------------------------------

class BatchRecordingModel(nn.Module):
    """Simple linear model that records batch sizes of each forward pass."""

    def __init__(self, in_features: int = 4, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.recorded_batch_sizes: List[int] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.recorded_batch_sizes.append(x.shape[0])
        return self.fc(x)


class DummyTokenizer:
    """Mock tokenizer to verify padding and truncation arguments."""

    def __init__(self):
        self.last_call_kwargs = {}

    def __call__(self, texts, **kwargs):
        self.last_call_kwargs = kwargs
        # Return mock token tensors of size (len(texts), max_length)
        max_len = kwargs.get("max_length", 128)
        n = len(texts) if isinstance(texts, list) else 1
        return {
            "input_ids": torch.ones((n, min(max_len, 10)), dtype=torch.long),
            "attention_mask": torch.ones((n, min(max_len, 10)), dtype=torch.long),
        }


# -------------------------------------------------------------------------
# 1. Tests for Step 2: XAIConfig
# -------------------------------------------------------------------------

def test_xai_config_defaults():
    cfg = XAIConfig()
    assert cfg.mode == "standard"
    assert cfg.technique is None
    assert cfg.budget == 100
    assert cfg.failed_attack_samples == 10
    assert cfg.per_sample_timeout_s == 30.0
    assert cfg.ig_steps == 25
    assert cfg.shap_max_evals == 100
    assert cfg.lime_num_samples == 200
    assert cfg.batch_size == 64
    assert cfg.max_text_length == 128
    assert cfg.seed == 0


def test_xai_config_presets():
    fast = XAIConfig.fast()
    assert fast.mode == "fast"
    assert fast.ig_steps == 10
    assert fast.shap_max_evals == 50
    assert fast.lime_num_samples == 50
    assert fast.batch_size == 64

    custom_fast = XAIConfig.fast(batch_size=128)
    assert custom_fast.batch_size == 128

    thorough = XAIConfig.thorough()
    assert thorough.mode == "thorough"
    assert thorough.ig_steps == 50
    assert thorough.shap_max_evals == 250
    assert thorough.lime_num_samples == 500
    assert thorough.batch_size == 64


def test_xai_config_serialization():
    cfg = XAIConfig(
        mode="fast",
        budget=40,
        ig_steps=12,
        batch_size=32,
    )
    d = cfg.to_dict()
    assert isinstance(d, dict)
    assert d["mode"] == "fast"
    assert d["budget"] == 40
    assert d["ig_steps"] == 12
    assert d["batch_size"] == 32

    restored = XAIConfig.from_dict(d)
    assert restored.mode == "fast"
    assert restored.budget == 40
    assert restored.ig_steps == 12
    assert restored.batch_size == 32


def test_xai_explainer_wires_config():
    model = nn.Sequential(nn.Linear(4, 2))
    custom_cfg = XAIConfig(
        mode="fast",
        ig_steps=11,
        lime_num_samples=33,
        shap_max_evals=44,
        batch_size=16,
    )
    explainer = XAIExplainer(config=custom_cfg)
    assert explainer.config.ig_steps == 11

    clean = torch.randn(1, 4)
    adv = torch.randn(1, 4)

    res = explainer.explain(
        model=model,
        clean_input=clean,
        adversarial_input=adv,
        technique="saliency",
        modality=Modality.TABULAR,
    )
    assert res.metadata["config"]["ig_steps"] == 11
    assert res.metadata["config"]["batch_size"] == 16
    assert res.metadata["config"]["mode"] == "fast"


def test_xai_explainer_config_technique_override():
    model = nn.Sequential(nn.Linear(4, 2))
    cfg = XAIConfig(technique="saliency")
    explainer = XAIExplainer(config=cfg)
    clean = torch.randn(1, 4)
    adv = torch.randn(1, 4)
    res = explainer.explain(
        model=model,
        clean_input=clean,
        adversarial_input=adv,
        modality=Modality.TABULAR,
    )
    assert res.technique == "saliency"


# -------------------------------------------------------------------------
# 2. Tests for Step 3: Batching Prediction Wrapper (predict_batched)
# -------------------------------------------------------------------------

def test_predict_batched_chunks_inputs():
    torch.manual_seed(42)
    model = BatchRecordingModel(in_features=4, num_classes=2)
    x = torch.randn(150, 4)

    # 150 items with batch_size=64 should result in 3 chunks: [64, 64, 22]
    out = predict_batched(model, x, batch_size=64)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (150, 2)
    assert model.recorded_batch_sizes == [64, 64, 22]


def test_predict_batched_numpy_input():
    torch.manual_seed(42)
    model = BatchRecordingModel(in_features=4, num_classes=2)
    x_np = np.random.randn(70, 4).astype(np.float32)

    out_np = predict_batched(model, x_np, batch_size=32)
    assert isinstance(out_np, np.ndarray)
    assert out_np.shape == (70, 2)
    assert model.recorded_batch_sizes == [32, 32, 6]


def test_predict_batched_text_truncation_and_tokenizer():
    class TextModel(nn.Module):
        def forward(self, input_ids=None, attention_mask=None):
            # output logits of shape (batch, 2)
            n = input_ids.shape[0]
            return torch.zeros((n, 2))

    model = TextModel()
    tokenizer = DummyTokenizer()
    texts = ["Sample sentence number " + str(i) for i in range(25)]

    out = predict_batched(
        model,
        texts,
        batch_size=10,
        max_text_length=64,
        tokenizer=tokenizer,
    )
    assert out.shape == (25, 2)
    assert tokenizer.last_call_kwargs.get("truncation") is True
    assert tokenizer.last_call_kwargs.get("max_length") == 64


def test_predict_proba_np_delegates_to_batching():
    model = BatchRecordingModel(in_features=4, num_classes=2)
    x_np = np.random.randn(80, 4).astype(np.float32)

    probs = predict_proba_np(model, x_np, output_mode="classification", batch_size=25)
    assert probs.shape == (80, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert model.recorded_batch_sizes == [25, 25, 25, 5]


def test_configure_cpu_threads():
    # Verify configure_cpu_threads executes safely without exception
    configure_cpu_threads(2)
    configure_cpu_threads(None)


# -------------------------------------------------------------------------
# 3. Integration Tests for Techniques with Config Parameters
# -------------------------------------------------------------------------

def test_gradient_explainer_ig_steps_from_kwargs():
    model = nn.Sequential(nn.Linear(4, 2))
    ig = GradientExplainer(method="integrated_gradients")
    x = torch.randn(1, 4)

    res = ig.explain(
        model,
        x,
        target_class=0,
        ig_steps=12,
        batch_size=5,
    )
    assert res["status"] == "success"
    assert res["executed"] is True
    assert np.array(res["attribution"]).shape == (1, 4)


def test_lime_explainer_reads_lime_num_samples_and_batch_size():
    model = nn.Sequential(nn.Linear(4, 2))
    lime = LIMEExplainer()
    x = np.random.randn(1, 4).astype(np.float32)

    res = lime.explain(
        model,
        x,
        target_class=1,
        lime_num_samples=30,
        batch_size=10,
        modality="tabular",
    )
    assert res["status"] == "success"
    assert res["executed"] is True
    assert len(res["attribution"]) == 4
