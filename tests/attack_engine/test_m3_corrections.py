"""
Comprehensive regression tests for M3 (Attack Engine) corrections.

Verifies:
1. UniversalTrigger execution through execute_attack() produces AttackResult without crashing.
2. UniversalTrigger execution through AttackEngine.run_attack().
3. Shape-changing attacks produce None / explicit shape mismatch in perturbation metrics.
4. Shape-preserving attacks continue to compute valid L0, L2, Linf norms.
5. Parameter Registry accepts initial_trigger for UniversalTrigger (both string and token list).
6. Parameter Registry accepts targets and target_labels as aliases for FAB target_label.
7. Parameter Registry correctly rejects genuinely unknown parameters.
8. validate_inputs rejects tensors containing NaN values with AttackExecutionError.
9. validate_inputs rejects tensors containing Inf values with AttackExecutionError.
10. validate_inputs accepts valid tensors (both float and integer token tensors).
11. Prediction inference failure in attack executor logs warning, records error,
    and returns AttackResult with success=None rather than crashing.
"""

import logging
from typing import List, Optional
import pytest
import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification

from app.attack_engine.attack_engine import AttackEngine
from app.attack_engine.attack_executor import execute_attack
from app.attack_engine.attacks.base_attack import BaseAttack
from app.attack_engine.attacks.text.universal_trigger import UniversalTrigger
from app.attack_engine.config import AttackConfig
from app.attack_engine.exceptions import (
    AttackConfigurationError,
    AttackExecutionError,
)
from app.attack_engine.models import AttackMetadata, AttackResult
from app.attack_engine.parameter_registry import (
    prepare_attack_config,
    validate_attack_params,
)
from app.attack_engine.utils.perturbation import (
    compute_l0_norm,
    compute_l2_norm,
    compute_linf_norm,
    compute_perturbation,
    compute_perturbation_metrics,
)
from app.attack_engine.utils.validation import validate_inputs


# =============================================================================
# Fixtures
# =============================================================================


class DummyTokenizer:
    """Lightweight deterministic tokenizer fixture with no external network calls."""

    def __init__(self, vocab_size: int = 120):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2
        self.unk_token_id = 3
        self.all_special_ids = [0, 1, 2, 3]

        self.id_to_word = {0: "[PAD]", 1: "[CLS]", 2: "[SEP]", 3: "[UNK]"}
        for i in range(4, vocab_size):
            self.id_to_word[i] = f"tok{i}"
        self.word_to_id = {v: k for k, v in self.id_to_word.items()}

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        words = text.split()
        ids = [self.word_to_id.get(w, self.unk_token_id) for w in words]
        if add_special_tokens:
            return [self.cls_token_id] + ids + [self.sep_token_id]
        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        words = []
        for tid in token_ids:
            tid_int = int(tid)
            if skip_special_tokens and tid_int in self.all_special_ids:
                continue
            words.append(self.id_to_word.get(tid_int, "[UNK]"))
        return " ".join(words)

    def batch_decode(self, sequences, skip_special_tokens: bool = False) -> List[str]:
        return [
            self.decode(s, skip_special_tokens=skip_special_tokens) for s in sequences
        ]

    def __call__(
        self,
        texts,
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 128,
    ):
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self.encode(t) for t in texts]
        max_len = max(len(s) for s in encoded)
        batch = torch.full((len(texts), max_len), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(texts), max_len), dtype=torch.long)
        for i, s in enumerate(encoded):
            batch[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1
        return {"input_ids": batch, "attention_mask": mask}


@pytest.fixture
def tokenizer():
    return DummyTokenizer(vocab_size=120)


@pytest.fixture
def bert_model(tokenizer):
    torch.manual_seed(42)
    config = BertConfig(
        vocab_size=120,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
    )
    model = BertForSequenceClassification(config)
    model.tokenizer = tokenizer
    model.eval()
    return model


# =============================================================================
# 1 & 2: UniversalTrigger Execution Compatibility
# =============================================================================


def test_universal_trigger_execute_attack_produces_result_without_crashing(bert_model, tokenizer):
    """Verify execute_attack() runs UniversalTrigger and handles sequence length expansion gracefully."""
    input_ids = torch.tensor([[1, 10, 11, 2], [1, 20, 21, 2]])
    labels = torch.tensor([0, 1])

    config = AttackConfig(
        params={
            "tokenizer": tokenizer,
            "trigger_length": 2,
            "num_iterations": 2,
            "candidate_vocab_size": 10,
            "placement": "prefix",
        },
    )

    result = execute_attack(
        model=bert_model,
        attack_cls=UniversalTrigger,
        inputs=input_ids,
        labels=labels,
        config=config,
    )

    assert isinstance(result, AttackResult)
    assert result.attack_name in ("universal_trigger", "UniversalTrigger")
    assert result.adversarial_examples is not None
    assert isinstance(result.adversarial_examples, torch.Tensor)
    assert result.adversarial_examples.shape == (2, 6)
    assert result.execution_time >= 0

    # Perturbation metrics handle shape mismatch cleanly
    assert result.perturbation_metrics.get("shape_mismatch") is True
    assert result.perturbation_metrics.get("l0") is None
    assert result.perturbation_metrics.get("l2") is None
    assert result.perturbation_metrics.get("linf") is None
    assert result.perturbation_metrics.get("orig_shape") == [2, 4]
    assert result.perturbation_metrics.get("adv_shape") == [2, 6]


def test_universal_trigger_attack_engine_run_attack(bert_model, tokenizer):
    """Verify AttackEngine.run_attack() executes UniversalTrigger cleanly."""
    input_ids = torch.tensor([[1, 10, 11, 2], [1, 20, 21, 2]])
    labels = torch.tensor([0, 1])

    config = AttackConfig(
        params={
            "tokenizer": tokenizer,
            "trigger_length": 2,
            "num_iterations": 2,
            "candidate_vocab_size": 10,
            "placement": "suffix",
        },
    )

    engine = AttackEngine(bert_model)
    result = engine.run_attack(
        attack_name="universal_trigger",
        inputs=input_ids,
        labels=labels,
        config=config,
    )

    assert isinstance(result, AttackResult)
    assert result.attack_name in ("universal_trigger", "UniversalTrigger")
    assert result.adversarial_examples.shape == (2, 6)
    assert result.perturbation_metrics["shape_mismatch"] is True
    assert result.perturbation_metrics["l2"] is None


# =============================================================================
# 3 & 4: Perturbation Metrics Shape Mismatch vs Shape Preserving
# =============================================================================


def test_shape_changing_metrics_return_explicit_mismatch():
    """Verify compute_perturbation_metrics returns None norms with shape mismatch info."""
    x_orig = torch.zeros(2, 4)
    x_adv = torch.zeros(2, 6)

    metrics = compute_perturbation_metrics(x_orig, x_adv)
    assert metrics["shape_mismatch"] is True
    assert metrics["orig_shape"] == [2, 4]
    assert metrics["adv_shape"] == [2, 6]
    assert metrics["l0"] is None
    assert metrics["l2"] is None
    assert metrics["linf"] is None
    assert metrics["mean_abs_error"] is None

    # Individual norm functions should raise ValueError on mismatched shapes
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_l0_norm(x_orig, x_adv)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_l2_norm(x_orig, x_adv)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_linf_norm(x_orig, x_adv)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_perturbation(x_orig, x_adv)


def test_shape_preserving_metrics_compute_valid_norms():
    """Verify compute_perturbation_metrics correctly computes valid norms when shapes match."""
    x_orig = torch.zeros(2, 4)
    x_adv = torch.ones(2, 4) * 0.5

    metrics = compute_perturbation_metrics(x_orig, x_adv)
    assert "shape_mismatch" not in metrics
    assert isinstance(metrics["l0"], float)
    assert isinstance(metrics["l2"], float)
    assert isinstance(metrics["linf"], float)
    assert isinstance(metrics["mean_abs_error"], float)

    assert metrics["l0"] > 0
    assert metrics["l2"] > 0
    assert abs(metrics["linf"] - 0.5) < 1e-5


# =============================================================================
# 5, 6, 7: Parameter Registry Consistency & Alias Resolution
# =============================================================================


def test_parameter_registry_accepts_initial_trigger_string_and_token_list():
    """Verify initial_trigger is accepted both as string and as token ID list."""
    # String format
    validated_str = validate_attack_params(
        "text", "universal_trigger", {"initial_trigger": "sample initial trigger"}
    )
    assert validated_str["initial_trigger"] == "sample initial trigger"

    # Token list format
    validated_list = validate_attack_params(
        "text", "universal_trigger", {"initial_trigger": [10, 11, 12]}
    )
    assert validated_list["initial_trigger"] == [10, 11, 12]

    # Integration through prepare_attack_config
    cfg = AttackConfig(params={"initial_trigger": [10, 11, 12]})
    norm_cfg = prepare_attack_config("text", "universal_trigger", cfg)
    assert norm_cfg.params["initial_trigger"] == [10, 11, 12]

    # Invalid type rejection
    with pytest.raises(AttackConfigurationError, match="initial_trigger"):
        validate_attack_params("text", "universal_trigger", {"initial_trigger": 12345})

    with pytest.raises(AttackConfigurationError, match="initial_trigger"):
        validate_attack_params("text", "universal_trigger", {"initial_trigger": ["invalid_token_id"]})


def test_parameter_registry_fab_target_label_aliases():
    """Verify FAB accepts 'targets' and 'target_labels' as aliases for target_label."""
    # 'targets' alias
    val_targets = validate_attack_params(
        "image", "fab", {"targeted": True, "targets": 5}
    )
    assert val_targets["target_label"] == 5
    assert "targets" not in val_targets

    # 'target_labels' alias
    val_labels = validate_attack_params(
        "image", "fab", {"targeted": True, "target_labels": 3}
    )
    assert val_labels["target_label"] == 3
    assert "target_labels" not in val_labels

    # Integration through prepare_attack_config
    cfg = AttackConfig(params={"targeted": True, "targets": 7})
    norm_cfg = prepare_attack_config("image", "fab", cfg)
    assert norm_cfg.params["target_label"] == 7


def test_parameter_registry_rejects_genuinely_unknown_parameters():
    """Verify unsupported parameters are rejected across domains."""
    with pytest.raises(AttackConfigurationError, match="Unknown parameter 'bogus_param'"):
        validate_attack_params("image", "fab", {"bogus_param": 1.0})

    with pytest.raises(AttackConfigurationError, match="Unknown parameter 'unsupported_key'"):
        validate_attack_params("text", "universal_trigger", {"unsupported_key": "val"})

    with pytest.raises(AttackConfigurationError, match="Unknown parameter 'step_size'"):
        validate_attack_params("image", "fgsm", {"step_size": 0.1})


# =============================================================================
# 8, 9, 10: NaN and Inf Input Validation
# =============================================================================


def test_validate_inputs_rejects_nan():
    """Verify validate_inputs raises AttackExecutionError when inputs contain NaN."""
    nan_tensor = torch.tensor([[1.0, float("nan")], [0.5, 0.2]])
    labels = torch.tensor([0, 1])

    with pytest.raises(AttackExecutionError, match="contain NaN values"):
        validate_inputs(nan_tensor, labels, attack_name="TestAttack")


def test_validate_inputs_rejects_inf():
    """Verify validate_inputs raises AttackExecutionError when inputs contain positive or negative Inf."""
    pos_inf_tensor = torch.tensor([[1.0, float("inf")], [0.5, 0.2]])
    neg_inf_tensor = torch.tensor([[1.0, float("-inf")], [0.5, 0.2]])
    labels = torch.tensor([0, 1])

    with pytest.raises(AttackExecutionError, match="contain Inf values"):
        validate_inputs(pos_inf_tensor, labels, attack_name="TestAttack")

    with pytest.raises(AttackExecutionError, match="contain Inf values"):
        validate_inputs(neg_inf_tensor, labels, attack_name="TestAttack")


def test_validate_inputs_accepts_valid_float_and_integer_tensors():
    """Verify validate_inputs cleanly accepts valid float and integer tensors without dtype errors."""
    # Float image/tabular tensor
    float_tensor = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])
    validate_inputs(float_tensor, labels, attack_name="FGSM")

    # Integer token tensor (text attacks)
    int_tensor = torch.tensor([[1, 10, 11, 2], [1, 20, 21, 2]], dtype=torch.long)
    int_labels = torch.tensor([0, 1])
    validate_inputs(int_tensor, int_labels, attack_name="UniversalTrigger")


# =============================================================================
# 11: Prediction Inference Exception Handling
# =============================================================================


def test_prediction_inference_failure_logged_and_recorded(caplog):
    """Verify prediction failure during execute_attack logs warning and sets success=None without crashing."""
    class FailingEvalModel(nn.Module):
        """Model that executes normally during attack generation but fails during prediction evaluation."""

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 2)
            self._fail_predictions = False

        def forward(self, x):
            if self._fail_predictions:
                raise RuntimeError("Simulated inference out-of-memory or device disconnection")
            return self.linear(x)

    model = FailingEvalModel()
    inputs = torch.randn(2, 4)
    labels = torch.tensor([0, 1])

    class CustomPassAttack(BaseAttack):
        def __init__(self, m):
            super().__init__(m)
            self.metadata = AttackMetadata(name="CustomPassAttack")

        def generate(self, x, y, config=None):
            # Arm model to fail when execute_attack attempts post-attack prediction
            self.model._fail_predictions = True
            return x + 0.01

        def validate_config(self, config):
            pass

    with caplog.at_level(logging.WARNING):
        result = execute_attack(
            model=model,
            attack_cls=CustomPassAttack,
            inputs=inputs,
            labels=labels,
        )

    assert isinstance(result, AttackResult)
    assert result.success is None
    assert result.original_predictions is None
    assert result.adversarial_predictions is None
    assert "prediction_eval_error" in result.metadata.parameters
    assert "Simulated inference out-of-memory" in result.metadata.parameters["prediction_eval_error"]

    # Verify log output was recorded
    assert any("Post-attack prediction inference failed" in record.message for record in caplog.records)
