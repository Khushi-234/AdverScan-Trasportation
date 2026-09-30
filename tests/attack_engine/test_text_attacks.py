"""
Unit and integration tests for white-box text adversarial attacks in AdverScan:
- HotFlip
- EmbeddingPGD
- TextBuggerWB
- UniversalTrigger
- GradientSubstitution
"""

from typing import List, Optional

import pytest
import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification

from app.attack_engine.attack_discovery import discover_attacks
from app.attack_engine.attack_engine import AttackEngine
from app.attack_engine.attack_executor import execute_attack
from app.attack_engine.attack_registry import get_attack, list_attacks
from app.attack_engine.attacks.text.embedding_pgd import EmbeddingPGD
from app.attack_engine.attacks.text.gradient_substitution import GradientSubstitution
from app.attack_engine.attacks.text.hotflip import HotFlip
from app.attack_engine.attacks.text.textbugger_wb import TextBuggerWB
from app.attack_engine.attacks.text.universal_trigger import UniversalTrigger
from app.attack_engine.config import AttackConfig
from app.attack_engine.exceptions import (
    AttackConfigurationError,
    AttackExecutionError,
    UnsupportedModelError,
)


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


class SimpleCustomTextClassifier(nn.Module):
    """Custom PyTorch text classifier with custom forward signature."""

    def __init__(
        self, vocab_size: int = 120, embed_dim: int = 16, num_classes: int = 2
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ):
        emb = self.embedding(input_ids)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            emb = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            emb = emb.mean(dim=1)
        return self.fc(emb)


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


@pytest.fixture
def custom_model(tokenizer):
    torch.manual_seed(42)
    model = SimpleCustomTextClassifier(vocab_size=120, embed_dim=16, num_classes=2)
    model.tokenizer = tokenizer
    model.eval()
    return model


# =========================================================================
# Discovery and Registry Tests
# =========================================================================


def test_text_attacks_registered():
    discover_attacks()
    attacks = list_attacks()
    for expected in [
        "hotflip",
        "embedding_pgd",
        "textbugger_wb",
        "textbugger",
        "universal_trigger",
        "gradient_substitution",
    ]:
        assert expected in attacks
        cls = get_attack(expected)
        assert cls is not None
        assert cls.metadata.domain == "text"
        assert cls.metadata.attack_type == "white_box"


# =========================================================================
# 1. HotFlip Tests
# =========================================================================


def test_hotflip_untargeted_strings(bert_model, tokenizer):
    attack = HotFlip(bert_model, tokenizer=tokenizer)
    inputs = ["tok10 tok11 tok12 tok13", "tok20 tok21 tok22 tok23"]
    labels = torch.tensor([0, 1])
    config = AttackConfig(params={"max_flips": 2, "candidate_vocab_size": 40})

    adv = attack.generate(inputs, labels, config)
    assert isinstance(adv, list)
    assert len(adv) == 2
    for orig, res in zip(inputs, adv):
        assert isinstance(res, str)
        # Tokenizer can process adversarial text
        re_encoded = tokenizer(res)
        assert re_encoded["input_ids"].shape[1] >= 2


def test_hotflip_tensor_input_and_budget(bert_model, tokenizer):
    attack = HotFlip(bert_model, tokenizer=tokenizer)
    input_ids = torch.tensor([[1, 10, 11, 12, 2], [1, 20, 21, 22, 2]])
    labels = torch.tensor([0, 1])
    config = AttackConfig(params={"max_flips": 1})

    adv = attack.generate(input_ids, labels, config)
    assert isinstance(adv, torch.Tensor)
    assert adv.shape == input_ids.shape
    # Special tokens must not be modified: [CLS]=1 at idx 0, [SEP]=2 at idx 4
    assert adv[0, 0].item() == 1
    assert adv[0, 4].item() == 2
    assert adv[1, 0].item() == 1
    assert adv[1, 4].item() == 2
    # At most 1 token modified per sample
    diff = (adv != input_ids).sum(dim=1)
    assert (diff <= 1).all()


def test_hotflip_targeted(bert_model, tokenizer):
    attack = HotFlip(bert_model, tokenizer=tokenizer)
    inputs = "tok10 tok11 tok12"
    labels = torch.tensor([0])
    config = AttackConfig(
        params={
            "targeted": True,
            "target_label": 1,
            "max_flips": 3,
            "candidate_vocab_size": 30,
        }
    )
    adv = attack.generate(inputs, labels, config)
    assert isinstance(adv, str)


def test_hotflip_invalid_config(bert_model):
    attack = HotFlip(bert_model)
    with pytest.raises(AttackConfigurationError):
        attack.validate_config(AttackConfig(params={"max_flips": 0}))

    with pytest.raises(AttackConfigurationError):
        attack.validate_config(AttackConfig(params={"candidate_vocab_size": -5}))


# =========================================================================
# 2. EmbeddingPGD Tests
# =========================================================================


def test_embedding_pgd_linf(bert_model, tokenizer):
    attack = EmbeddingPGD(bert_model, tokenizer=tokenizer)
    inputs = "tok10 tok11 tok12 tok13"
    labels = torch.tensor([0])
    config = AttackConfig(
        epsilon=0.8,
        params={"norm": "Linf", "alpha": 0.2, "num_steps": 3, "k_nearest": 1},
    )

    adv = attack.generate(inputs, labels, config)
    assert isinstance(adv, str)
    assert len(adv) > 0


def test_embedding_pgd_l2_and_knn(bert_model, tokenizer):
    attack = EmbeddingPGD(bert_model, tokenizer=tokenizer)
    input_ids = torch.tensor([[1, 15, 16, 17, 2]])
    labels = torch.tensor([0])
    config = AttackConfig(
        epsilon=1.0,
        params={"norm": "L2", "alpha": 0.3, "num_steps": 3, "k_nearest": 3},
    )

    adv = attack.generate(input_ids, labels, config)
    assert isinstance(adv, torch.Tensor)
    assert adv.shape == input_ids.shape
    # Special tokens [CLS] and [SEP] preserved
    assert adv[0, 0].item() == 1
    assert adv[0, 4].item() == 2


def test_embedding_pgd_targeted(bert_model, tokenizer):
    attack = EmbeddingPGD(bert_model, tokenizer=tokenizer)
    inputs = "tok10 tok11 tok12"
    labels = torch.tensor([0])
    config = AttackConfig(
        epsilon=0.5,
        params={"targeted": True, "target_label": 1, "num_steps": 2},
    )
    adv = attack.generate(inputs, labels, config)
    assert isinstance(adv, str)


def test_embedding_pgd_invalid_config(bert_model):
    attack = EmbeddingPGD(bert_model)
    with pytest.raises(AttackConfigurationError):
        attack.validate_config(AttackConfig(epsilon=-1.0))
    with pytest.raises(AttackConfigurationError):
        attack.validate_config(AttackConfig(params={"step_size": -0.5}))
    with pytest.raises(AttackConfigurationError):
        attack.validate_config(AttackConfig(params={"norm": "L3"}))


# =========================================================================
# 3. TextBuggerWB Tests
# =========================================================================


def test_textbugger_wb_untargeted(bert_model, tokenizer):
    attack = TextBuggerWB(bert_model, tokenizer=tokenizer)
    inputs = "tok10 tok11 tok12 tok13"
    labels = torch.tensor([0])
    config = AttackConfig(
        params={"max_modifications": 2, "bug_types": ["sub_w", "swap"]}
    )

    adv = attack.generate(inputs, labels, config)
    assert isinstance(adv, str)
    assert len(adv) > 0


def test_textbugger_wb_all_bugs(bert_model, tokenizer):
    attack = TextBuggerWB(bert_model, tokenizer=tokenizer)
    input_ids = torch.tensor([[1, 20, 21, 22, 2]])
    labels = torch.tensor([1])
    config = AttackConfig(
        params={
            "max_modifications": 2,
            "bug_types": ["insert", "delete", "swap", "sub_c", "sub_w"],
        }
    )

    adv = attack.generate(input_ids, labels, config)
    assert isinstance(adv, torch.Tensor)
    assert adv.shape == input_ids.shape
    # Special tokens preserved
    assert adv[0, 0].item() == 1
    assert adv[0, 4].item() == 2


def test_textbugger_invalid_config(bert_model):
    attack = TextBuggerWB(bert_model)
    with pytest.raises(AttackConfigurationError):
        attack.validate_config(AttackConfig(params={"max_modifications": 0}))
    with pytest.raises(AttackConfigurationError):
        attack.validate_config(AttackConfig(params={"bug_types": ["non_existent_bug"]}))


# =========================================================================
# 4. UniversalTrigger Tests
# =========================================================================


def test_universal_trigger_prefix(bert_model, tokenizer):
    attack = UniversalTrigger(bert_model, tokenizer=tokenizer)
    inputs = ["tok10 tok11 tok12", "tok20 tok21 tok22"]
    labels = torch.tensor([0, 1])
    config = AttackConfig(
        params={
            "trigger_length": 2,
            "placement": "prefix",
            "num_iterations": 2,
            "candidate_vocab_size": 10,
        }
    )

    adv = attack.generate(inputs, labels, config)
    assert isinstance(adv, list)
    assert len(adv) == 2
    # Verify input_agnostic metadata
    assert attack.metadata.parameters.get("input_agnostic") is True
    assert "trigger_tokens" in config.params
    assert len(config.params["trigger_tokens"]) == 2


def test_universal_trigger_suffix(bert_model, tokenizer):
    attack = UniversalTrigger(bert_model, tokenizer=tokenizer)
    input_ids = torch.tensor([[1, 10, 11, 2], [1, 20, 21, 2]])
    labels = torch.tensor([0, 0])
    config = AttackConfig(
        params={
            "trigger_length": 2,
            "placement": "suffix",
            "num_iterations": 2,
            "candidate_vocab_size": 10,
        }
    )

    adv = attack.generate(input_ids, labels, config)
    assert isinstance(adv, torch.Tensor)
    # Length expands by trigger_length (4 + 2 = 6)
    assert adv.shape == (2, 6)
    # Leading [CLS] preserved
    assert (adv[:, 0] == 1).all()


def test_universal_trigger_targeted(bert_model, tokenizer):
    attack = UniversalTrigger(bert_model, tokenizer=tokenizer)
    inputs = "tok10 tok11"
    labels = torch.tensor([0])
    config = AttackConfig(
        params={
            "targeted": True,
            "target_label": 1,
            "trigger_length": 1,
            "num_iterations": 1,
        }
    )
    adv = attack.generate(inputs, labels, config)
    assert isinstance(adv, str)


def test_universal_trigger_invalid_config(bert_model):
    attack = UniversalTrigger(bert_model)
    with pytest.raises(AttackConfigurationError):
        attack.validate_config(AttackConfig(params={"trigger_length": 0}))
    with pytest.raises(AttackConfigurationError):
        attack.validate_config(AttackConfig(params={"placement": "invalid_pos"}))


# =========================================================================
# 5. GradientSubstitution Tests
# =========================================================================


def test_gradient_substitution_untargeted(bert_model, tokenizer):
    attack = GradientSubstitution(bert_model, tokenizer=tokenizer)
    inputs = "tok10 tok11 tok12 tok13"
    labels = torch.tensor([0])
    config = AttackConfig(
        params={
            "max_modifications": 2,
            "num_candidates": 5,
            "similarity_threshold": 0.2,
        }
    )

    adv = attack.generate(inputs, labels, config)
    assert isinstance(adv, str)


def test_gradient_substitution_tensor_and_budget(bert_model, tokenizer):
    attack = GradientSubstitution(bert_model, tokenizer=tokenizer)
    input_ids = torch.tensor([[1, 30, 31, 32, 2]])
    labels = torch.tensor([1])
    config = AttackConfig(params={"max_modifications": 1, "num_candidates": 5})

    adv = attack.generate(input_ids, labels, config)
    assert isinstance(adv, torch.Tensor)
    assert adv.shape == input_ids.shape
    # Special tokens preserved
    assert adv[0, 0].item() == 1
    assert adv[0, 4].item() == 2
    # At most 1 modification
    diff = (adv != input_ids).sum().item()
    assert diff <= 1


def test_gradient_substitution_targeted(bert_model, tokenizer):
    attack = GradientSubstitution(bert_model, tokenizer=tokenizer)
    inputs = "tok10 tok11"
    labels = torch.tensor([0])
    config = AttackConfig(
        params={"targeted": True, "target_label": 1, "max_modifications": 2}
    )
    adv = attack.generate(inputs, labels, config)
    assert isinstance(adv, str)


def test_gradient_substitution_invalid_config(bert_model):
    attack = GradientSubstitution(bert_model)
    with pytest.raises(AttackConfigurationError):
        attack.validate_config(AttackConfig(params={"max_modifications": -1}))
    with pytest.raises(AttackConfigurationError):
        attack.validate_config(AttackConfig(params={"similarity_threshold": 1.5}))


# =========================================================================
# 6. Common Constraints & Input Formats Tests
# =========================================================================


def test_empty_input_raises_error(bert_model, tokenizer):
    attack = HotFlip(bert_model, tokenizer=tokenizer)
    with pytest.raises(AttackExecutionError):
        attack.generate("", torch.tensor([0]))
    with pytest.raises(AttackExecutionError):
        attack.generate([], torch.tensor([0]))


def test_missing_tokenizer_with_strings_raises_error(bert_model):
    # Model without attached tokenizer and no tokenizer passed in
    model_no_tok = BertForSequenceClassification(
        BertConfig(
            vocab_size=100,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
        )
    )
    attack = HotFlip(model_no_tok)
    with pytest.raises(AttackConfigurationError):
        attack.generate(["some text"], torch.tensor([0]))


def test_dict_input_format(bert_model, tokenizer):
    attack = HotFlip(bert_model, tokenizer=tokenizer)
    encoded = tokenizer("tok10 tok11 tok12", return_tensors="pt")
    labels = torch.tensor([0])
    adv_dict = attack.generate(encoded, labels, AttackConfig(params={"max_flips": 1}))
    assert isinstance(adv_dict, dict)
    assert "input_ids" in adv_dict
    assert adv_dict["input_ids"].shape == encoded["input_ids"].shape


def test_custom_pytorch_model_support(custom_model, tokenizer):
    # Verify attacks execute on custom PyTorch text architectures (not just HF)
    input_ids = torch.tensor([[1, 10, 11, 2]])
    labels = torch.tensor([0])

    hf = HotFlip(custom_model, tokenizer=tokenizer)
    adv_hf = hf.generate(input_ids, labels, AttackConfig(params={"max_flips": 1}))
    assert adv_hf.shape == input_ids.shape

    pgd = EmbeddingPGD(custom_model, tokenizer=tokenizer)
    adv_pgd = pgd.generate(
        input_ids, labels, AttackConfig(epsilon=0.5, params={"num_steps": 2})
    )
    assert adv_pgd.shape == input_ids.shape


# =========================================================================
# 7. Framework Pipeline and Executor Integration Tests
# =========================================================================


def test_execute_attack_integration(bert_model, tokenizer):
    input_ids = torch.tensor([[1, 10, 11, 12, 2]])
    labels = torch.tensor([0])
    res = execute_attack(
        model=bert_model,
        attack_cls=HotFlip,
        inputs=input_ids,
        labels=labels,
        config=AttackConfig(params={"max_flips": 1, "tokenizer": tokenizer}),
    )
    assert res is not None
    assert res.attack_name == "hotflip"
    assert res.adversarial_examples is not None
    assert "l0" in res.perturbation_metrics


def test_attack_engine_orchestration(bert_model, tokenizer):
    engine = AttackEngine(bert_model)
    input_ids = torch.tensor([[1, 20, 21, 2]])
    labels = torch.tensor([0])

    res = engine.run_attack(
        "hotflip",
        input_ids,
        labels,
        AttackConfig(params={"max_flips": 1, "tokenizer": tokenizer}),
    )
    assert res.attack_name == "hotflip"

    res_pgd = engine.run_attack(
        "embedding_pgd",
        input_ids,
        labels,
        AttackConfig(epsilon=0.5, params={"num_steps": 2, "tokenizer": tokenizer}),
    )
    assert res_pgd.attack_name == "embedding_pgd"


def test_unsupported_model_no_embeddings():
    # A model without any nn.Embedding layer should raise UnsupportedModelError
    linear_model = nn.Linear(10, 2)
    attack = HotFlip(linear_model)
    with pytest.raises(UnsupportedModelError):
        attack.generate(torch.tensor([[1, 2, 3]]), torch.tensor([0]))
