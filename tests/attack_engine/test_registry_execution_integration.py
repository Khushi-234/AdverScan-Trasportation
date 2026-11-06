"""
Integration tests for centralized parameter registry with attack execution.

Verifies:
- Correct parameter propagation for Image, Tabular, Time-series, and Text attacks.
- Default parameter preservation.
- Alias normalization.
- Targeted attack configuration.
- Rejection of invalid and unsupported parameters with clear errors.
- Runtime dependency preservation.
- AttackEngine pipeline and execute_attack compatibility.
"""

import pytest
import torch
import torch.nn as nn

from app.attack_engine import (
    AttackConfig,
    AttackEngine,
    execute_attack,
    run_attack_pipeline,
    FGSM,
    PGD,
    CW,
    CapPGD,
    TabularCW,
    HotFlip,
    TextBuggerWB,
    TimeSeriesPGD,
    SmoothPGD,
)
from app.attack_engine.exceptions import AttackConfigurationError


# =============================================================================
# SYNTHETIC TEST MODELS
# =============================================================================

class SimpleImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)
        nn.init.constant_(self.fc.weight[0], 0.3)
        nn.init.constant_(self.fc.weight[1], -0.2)
        nn.init.constant_(self.fc.weight[2], 0.1)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        return self.fc(x)


class SimpleTabularModel(nn.Module):
    def __init__(self, in_features=6, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(x)


class SimpleTimeSeriesModel(nn.Module):
    def __init__(self, in_features=3, hidden_dim=8, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class SimpleTextModel(nn.Module):
    def __init__(self, vocab_size=100, embed_dim=16, num_classes=2):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        pooled = inputs_embeds.mean(dim=1)
        logits = self.classifier(pooled)
        return logits


class SimpleTokenizer:
    def __init__(self):
        self.vocab = {f"word_{i}": i for i in range(100)}
        self.vocab.update({"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3})
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.cls_token_id = 2
        self.sep_token_id = 3

    def encode(self, text, add_special_tokens=True):
        words = text.split()
        return [self.vocab.get(w, self.unk_token_id) for w in words]

    def decode(self, token_ids):
        return " ".join([self.inv_vocab.get(tid, "[UNK]") for tid in token_ids])

    def __call__(self, texts, return_tensors=None, padding=True, truncation=True, max_length=None):
        if isinstance(texts, str):
            texts = [texts]
        ids = [self.encode(t) for t in texts]
        max_l = max(len(row) for row in ids)
        padded = [row + [0] * (max_l - len(row)) for row in ids]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(padded, dtype=torch.long)}
        return {"input_ids": padded}


# =============================================================================
# PARAMETER PROPAGATION ACROSS MODALITIES
# =============================================================================

def test_image_param_propagation():
    model = SimpleImageModel()
    engine = AttackEngine(model)
    inputs = torch.randn(2, 10)
    labels = torch.tensor([0, 1])

    config = AttackConfig(
        epsilon=0.15,
        params={"num_steps": 15, "alpha": 0.02, "norm": "L2"},
    )
    result = engine.run_attack("pgd", inputs, labels, config)
    assert result is not None
    assert result.metadata.name == "pgd"
    assert config.params["num_steps"] == 15
    assert config.params["alpha"] == 0.02
    assert config.params["norm"] == "L2"
    assert config.epsilon == 0.15


def test_tabular_param_propagation():
    model = SimpleTabularModel()
    inputs = torch.randn(3, 6)
    labels = torch.tensor([0, 1, 0])

    config = AttackConfig(
        params={"c_penalty": 2.0, "confidence": 0.5, "max_iterations": 10}
    )
    result = execute_attack(model, TabularCW, inputs, labels, config)
    assert result is not None
    assert result.attack_name == "carlini_wagner_tabular"
    assert config.params["c_penalty"] == 2.0
    assert config.params["confidence"] == 0.5
    assert config.params["max_iterations"] == 10


def test_time_series_param_propagation():
    model = SimpleTimeSeriesModel()
    engine = AttackEngine(model)
    inputs = torch.randn(2, 8, 3)
    labels = torch.tensor([1, 0])

    config = AttackConfig(
        epsilon=0.2,
        params={"num_steps": 5, "alpha": 0.05, "tv_weight": 0.25},
    )
    result = engine.run_attack("smooth_pgd", inputs, labels, config)
    assert result is not None
    assert config.params["num_steps"] == 5
    assert config.params["alpha"] == 0.05
    assert config.params["tv_weight"] == 0.25
    assert config.epsilon == 0.2


def test_text_param_propagation():
    model = SimpleTextModel()
    tokenizer = SimpleTokenizer()
    inputs = torch.tensor([[10, 20, 30, 40]])
    labels = torch.tensor([1])

    config = AttackConfig(
        params={
            "max_modifications": 2,
            "candidate_vocab_size": 25,
            "tokenizer": tokenizer,
        }
    )
    result = execute_attack(model, HotFlip, inputs, labels, config)
    assert result is not None
    assert config.params["max_modifications"] == 2
    assert config.params["candidate_vocab_size"] == 25
    assert config.params["tokenizer"] is tokenizer


# =============================================================================
# DEFAULT PARAMETER PRESERVATION
# =============================================================================

def test_default_parameter_preservation_fgsm():
    model = SimpleImageModel()
    engine = AttackEngine(model)
    inputs = torch.randn(2, 10)
    labels = torch.tensor([0, 1])

    result = engine.run_attack("fgsm", inputs, labels, config=None)
    assert result is not None
    assert result.metadata.epsilon == 0.1
    assert result.metadata.clip_min == 0.0
    assert result.metadata.clip_max == 1.0


def test_default_parameter_preservation_cw():
    model = SimpleImageModel()
    inputs = torch.randn(2, 10)
    labels = torch.tensor([0, 1])

    # Default AttackConfig should not inject epsilon into CW
    config = AttackConfig(params={"max_iterations": 5, "binary_search_steps": 1})
    result = execute_attack(model, CW, inputs, labels, config)
    assert result is not None
    assert config.params["max_iterations"] == 5
    assert "epsilon" not in config.params


# =============================================================================
# ALIAS NORMALIZATION
# =============================================================================

def test_alias_normalization_pgd():
    model = SimpleImageModel()
    engine = AttackEngine(model)
    inputs = torch.randn(2, 10)
    labels = torch.tensor([0, 1])

    # Pass aliases 'steps' and 'step_size'
    config = AttackConfig(params={"steps": 12, "step_size": 0.03})
    result = engine.run_attack("pgd", inputs, labels, config)
    assert result is not None
    assert config.params["num_steps"] == 12
    assert config.params["alpha"] == 0.03
    assert "steps" not in config.params
    assert "step_size" not in config.params


def test_alias_normalization_time_series():
    model = SimpleTimeSeriesModel()
    inputs = torch.randn(2, 6, 3)
    labels = torch.tensor([0, 1])

    # Pass alias 'smoothness_factor' and 'iters'
    config = AttackConfig(params={"iters": 6, "smoothness_factor": 0.2})
    result = execute_attack(model, SmoothPGD, inputs, labels, config)
    assert result is not None
    assert config.params["num_steps"] == 6
    assert config.params["tv_weight"] == 0.2
    assert "iters" not in config.params
    assert "smoothness_factor" not in config.params


def test_alias_normalization_hotflip():
    model = SimpleTextModel()
    tokenizer = SimpleTokenizer()
    inputs = torch.tensor([[5, 12, 18]])
    labels = torch.tensor([0])

    # Pass aliases 'max_flips' and 'top_k_candidates'
    config = AttackConfig(
        params={"max_flips": 1, "top_k_candidates": 30, "tokenizer": tokenizer}
    )
    result = execute_attack(model, HotFlip, inputs, labels, config)
    assert result is not None
    assert config.params["max_modifications"] == 1
    assert config.params["candidate_vocab_size"] == 30
    assert "max_flips" not in config.params
    assert "top_k_candidates" not in config.params


# =============================================================================
# TARGETED ATTACK CONFIGURATION
# =============================================================================

def test_targeted_attack_valid():
    model = SimpleImageModel()
    engine = AttackEngine(model)
    inputs = torch.randn(2, 10)
    labels = torch.tensor([0, 1])

    config = AttackConfig(params={"targeted": True, "target_label": 2})
    result = engine.run_attack("fgsm", inputs, labels, config)
    assert result is not None
    assert config.params["targeted"] is True
    assert config.params["target_label"] == 2


def test_targeted_attack_alias():
    model = SimpleImageModel()
    inputs = torch.randn(2, 10)
    labels = torch.tensor([0, 1])

    # Pass alias 'target' instead of 'target_label'
    config = AttackConfig(params={"targeted": True, "target": 2})
    result = execute_attack(model, FGSM, inputs, labels, config)
    assert result is not None
    assert config.params["target_label"] == 2
    assert "target" not in config.params


def test_target_label_without_targeted_flag_raises():
    model = SimpleImageModel()
    engine = AttackEngine(model)
    inputs = torch.randn(2, 10)
    labels = torch.tensor([0, 1])

    config = AttackConfig(params={"target_label": 2})
    with pytest.raises(AttackConfigurationError) as exc_info:
        engine.run_attack("fgsm", inputs, labels, config)
    assert "Set 'targeted=True' to execute a targeted attack" in str(exc_info.value)


# =============================================================================
# INVALID & UNSUPPORTED PARAMETER REJECTION
# =============================================================================

def test_unsupported_param_rejection():
    model = SimpleImageModel()
    engine = AttackEngine(model)
    inputs = torch.randn(2, 10)
    labels = torch.tensor([0, 1])

    config = AttackConfig(params={"unsupported_fake_param": 123})
    with pytest.raises(AttackConfigurationError) as exc_info:
        engine.run_attack("fgsm", inputs, labels, config)
    assert "Unknown parameter 'unsupported_fake_param' for attack 'fgsm'" in str(exc_info.value)


def test_unsupported_epsilon_for_cw_rejection():
    model = SimpleImageModel()
    inputs = torch.randn(2, 10)
    labels = torch.tensor([0, 1])

    # Passing explicit epsilon into params for CW should be rejected
    config = AttackConfig(params={"epsilon": 0.05})
    with pytest.raises(AttackConfigurationError) as exc_info:
        execute_attack(model, CW, inputs, labels, config)
    assert "Unknown parameter 'epsilon' for attack 'cw'" in str(exc_info.value)


def test_invalid_type_rejection():
    model = SimpleImageModel()
    engine = AttackEngine(model)
    inputs = torch.randn(2, 10)
    labels = torch.tensor([0, 1])

    config = AttackConfig(params={"num_steps": "twenty"})
    with pytest.raises(AttackConfigurationError) as exc_info:
        engine.run_attack("pgd", inputs, labels, config)
    assert "must be an integer" in str(exc_info.value)


def test_invalid_range_rejection():
    model = SimpleImageModel()
    engine = AttackEngine(model)
    inputs = torch.randn(2, 10)
    labels = torch.tensor([0, 1])

    config = AttackConfig(params={"alpha": -0.05})
    with pytest.raises(AttackConfigurationError) as exc_info:
        engine.run_attack("pgd", inputs, labels, config)
    assert "must be strictly greater than 0.0" in str(exc_info.value)


def test_invalid_norm_rejection():
    model = SimpleImageModel()
    engine = AttackEngine(model)
    inputs = torch.randn(2, 10)
    labels = torch.tensor([0, 1])

    config = AttackConfig(params={"norm": "L3"})
    with pytest.raises(AttackConfigurationError) as exc_info:
        engine.run_attack("pgd", inputs, labels, config)
    assert "Unsupported norm 'L3'" in str(exc_info.value)


# =============================================================================
# RUNTIME DEPENDENCY PRESERVATION
# =============================================================================

def test_runtime_dependencies_preserved():
    model = SimpleTabularModel()
    inputs = torch.randn(2, 6)
    labels = torch.tensor([0, 1])

    dummy_meta = {"feat1": "cont", "feat2": "cat"}
    custom_loss = nn.CrossEntropyLoss()
    config = AttackConfig(
        epsilon=0.1,
        params={
            "num_steps": 5,
            "feature_metadata": dummy_meta,
            "loss_fn": custom_loss,
            "device": "cpu",
        },
    )
    result = execute_attack(model, CapPGD, inputs, labels, config)
    assert result is not None
    assert config.params["feature_metadata"] is dummy_meta
    assert config.params["loss_fn"] is custom_loss
    assert config.params["device"] == "cpu"


# =============================================================================
# ATTACK ENGINE PIPELINE COMPATIBILITY
# =============================================================================

def test_pipeline_execution_with_registry():
    model = SimpleImageModel()
    inputs = torch.randn(2, 10)
    labels = torch.tensor([0, 1])

    configs = {
        "fgsm": AttackConfig(epsilon=0.08),
        "pgd": AttackConfig(params={"steps": 5, "step_size": 0.02}),
    }
    results = run_attack_pipeline(model, ["fgsm", "pgd"], inputs, labels, configs=configs)
    assert len(results) == 2
    assert "fgsm" in results
    assert "pgd" in results
    assert results["fgsm"].metadata.epsilon == 0.08
    assert results["pgd"].metadata.parameters["num_steps"] == 5


def test_engine_validate_attack_config_helper():
    model = SimpleImageModel()
    engine = AttackEngine(model)

    config = AttackConfig(params={"steps": 10, "step_size": 0.02})
    validated_config = engine.validate_attack_config("pgd", config)
    assert validated_config.params["num_steps"] == 10
    assert validated_config.params["alpha"] == 0.02
    assert "steps" not in validated_config.params


# =============================================================================
# EXPLICIT VERIFICATION OF REVIEW CRITERIA
# =============================================================================

def test_param_compatibility_deepfool_cw_wavelet_softdtw():
    from app.attack_engine.parameter_registry import get_attack_param_specs

    # 1. DeepFool: max_iter, overshoot, top_k, epsilon, clip_min, clip_max
    df_specs = {s.name for s in get_attack_param_specs("image", "deepfool")}
    assert df_specs == {"max_iter", "overshoot", "top_k", "epsilon", "clip_min", "clip_max"}

    # 2. CW: verify exact parameters; abort_early and epsilon are not valid hyperparameters
    cw_specs = {s.name for s in get_attack_param_specs("image", "cw")}
    assert "abort_early" not in cw_specs
    assert "epsilon" not in cw_specs
    assert cw_specs == {
        "confidence",
        "learning_rate",
        "max_iterations",
        "binary_search_steps",
        "initial_const",
        "targeted",
        "target_label",
        "clip_min",
        "clip_max",
    }

    # 3. WaveletAttack: wavelet and level removed; exact parameters verified
    wv_specs = {s.name for s in get_attack_param_specs("time_series", "wavelet_attack")}
    assert "wavelet" not in wv_specs
    assert "level" not in wv_specs
    assert wv_specs == {
        "epsilon",
        "alpha",
        "num_steps",
        "norm",
        "transform_type",
        "band",
        "clip_min",
        "clip_max",
        "targeted",
        "target_label",
    }

    # 4. SoftDTWAttack: exact parameters including lambda_smooth, downsample_threshold
    dtw_specs = {s.name for s in get_attack_param_specs("time_series", "soft_dtw_attack")}
    assert "lambda_smooth" in dtw_specs
    assert "downsample_threshold" in dtw_specs
    assert dtw_specs == {
        "epsilon",
        "alpha",
        "num_steps",
        "max_warp",
        "gamma",
        "lambda_dtw",
        "lambda_smooth",
        "downsample_threshold",
        "clip_min",
        "clip_max",
        "targeted",
        "target_label",
    }


def test_direct_attack_config_backward_compatibility():
    # Direct instantiation of attack and AttackConfig without AttackEngine
    model = SimpleImageModel()
    fgsm = FGSM(model)
    cw = CW(model)
    inputs = torch.randn(2, 10)
    labels = torch.tensor([0, 1])

    # Direct generate call with standard AttackConfig
    cfg_direct = AttackConfig(epsilon=0.1, clip_min=0.0, clip_max=1.0)
    adv_fgsm = fgsm.generate(inputs, labels, cfg_direct)
    assert adv_fgsm is not None
    assert adv_fgsm.shape == inputs.shape

    # Direct generate call with default AttackConfig on CW (does not fail on default epsilon=0.1)
    cfg_cw = AttackConfig(params={"max_iterations": 2, "binary_search_steps": 1})
    adv_cw = cw.generate(inputs, labels, cfg_cw)
    assert adv_cw is not None
    assert adv_cw.shape == inputs.shape


def test_runtime_context_isolation_and_immutability():
    from app.attack_engine.parameter_registry import (
        get_supported_attacks,
        get_attack_param_specs,
        RUNTIME_CONTEXT_KEYS,
        validate_attack_params,
    )

    # 1. Ensure runtime context keys NEVER appear in user-facing parameter specs
    for domain in ["image", "tabular", "time_series", "text"]:
        for attack_name in get_supported_attacks(domain):
            spec_names = {s.name for s in get_attack_param_specs(domain, attack_name)}
            for rk in RUNTIME_CONTEXT_KEYS:
                assert rk not in spec_names, f"Runtime context key '{rk}' leaked into specs of '{attack_name}'!"

    # 2. Ensure runtime context objects passed in params are never modified or rejected
    dummy_tokenizer = object()
    dummy_loss = object()
    dummy_bounds = [(0.0, 1.0)]
    params = {
        "max_modifications": 2,
        "tokenizer": dummy_tokenizer,
        "loss_fn": dummy_loss,
        "feature_bounds": dummy_bounds,
    }
    validated = validate_attack_params("text", "hotflip", params)
    assert validated["max_modifications"] == 2
    assert validated["tokenizer"] is dummy_tokenizer
    assert validated["loss_fn"] is dummy_loss
    assert validated["feature_bounds"] is dummy_bounds


def test_cw_rejects_epsilon_for_standard_l2_formulation():
    from app.attack_engine.parameter_registry import validate_attack_params

    # Standard CW-L2 is an unbounded optimization formulation, so epsilon is not a valid parameter
    with pytest.raises(AttackConfigurationError) as exc_info:
        validate_attack_params("image", "cw", {"epsilon": 0.1})
    assert "Unknown parameter 'epsilon' for attack 'cw'" in str(exc_info.value)


