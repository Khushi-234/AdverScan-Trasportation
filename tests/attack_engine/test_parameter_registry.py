"""
Comprehensive unit tests for the centralized attack parameter registry.
"""

import pytest
from app.attack_engine.exceptions import AttackConfigurationError
from app.attack_engine.parameter_registry import (
    ParameterSpec,
    SUPPORTED_DOMAINS,
    RUNTIME_CONTEXT_KEYS,
    get_supported_attacks,
    get_attack_param_specs,
    get_attack_param_specs_dict,
    validate_attack_params,
    normalize_domain,
    normalize_attack_name,
)


# =============================================================================
# DOMAIN & ATTACK DISCOVERY TESTS
# =============================================================================

def test_supported_domains():
    assert "image" in SUPPORTED_DOMAINS
    assert "tabular" in SUPPORTED_DOMAINS
    assert "time_series" in SUPPORTED_DOMAINS
    assert "text" in SUPPORTED_DOMAINS


@pytest.mark.parametrize(
    "input_domain,expected",
    [
        ("image", "image"),
        ("IMAGE", "image"),
        ("Image", "image"),
        ("vision", "image"),
        ("tabular", "tabular"),
        ("TABULAR", "tabular"),
        ("tables", "tabular"),
        ("time_series", "time_series"),
        ("time-series", "time_series"),
        ("TIME-SERIES", "time_series"),
        ("timeseries", "time_series"),
        ("ts", "time_series"),
        ("text", "text"),
        ("TEXT", "text"),
        ("nlp", "text"),
    ],
)
def test_normalize_domain_valid(input_domain, expected):
    assert normalize_domain(input_domain) == expected


def test_normalize_domain_invalid():
    with pytest.raises(AttackConfigurationError) as exc_info:
        normalize_domain("audio")
    assert "Unsupported domain 'audio'" in str(exc_info.value)

    with pytest.raises(AttackConfigurationError):
        normalize_domain(123)  # Non-string domain


def test_get_supported_attacks_image():
    attacks = get_supported_attacks("image")
    assert attacks == ["fgsm", "pgd", "deepfool", "cw", "fab"]

    display_names = get_supported_attacks("image", return_display_names=True)
    assert display_names == ["FGSM", "PGD", "DeepFool", "CW", "FAB"]


def test_get_supported_attacks_tabular():
    attacks = get_supported_attacks("tabular")
    assert attacks == ["cap_pgd", "lowprofool", "tabular_cw", "tabular_jsma", "vae_attack"]

    display_names = get_supported_attacks("tabular", return_display_names=True)
    assert display_names == ["CapPGD", "LowProFool", "TabularCW", "TabularJSMA", "VAEAttack"]


def test_get_supported_attacks_time_series():
    attacks = get_supported_attacks("time-series")
    assert attacks == ["time_series_pgd", "smooth_pgd", "wavelet_attack", "soft_dtw_attack", "jsma"]

    display_names = get_supported_attacks("ts", return_display_names=True)
    assert display_names == ["TimeSeriesPGD", "SmoothPGD", "WaveletAttack", "SoftDTWAttack", "JSMA"]


def test_get_supported_attacks_text():
    attacks = get_supported_attacks("text")
    assert attacks == ["hotflip", "embedding_pgd", "textbugger_wb", "universal_trigger", "gradient_substitution"]

    display_names = get_supported_attacks("text", return_display_names=True)
    assert display_names == ["HotFlip", "EmbeddingPGD", "TextBuggerWB", "UniversalTrigger", "GradientSubstitution"]


# =============================================================================
# ATTACK PARAMETER SPECIFICATIONS TESTS
# =============================================================================

@pytest.mark.parametrize(
    "domain,attack_name,expected_param_names",
    [
        # Image
        ("image", "fgsm", ["epsilon", "clip_min", "clip_max", "targeted", "target_label"]),
        ("image", "pgd", ["epsilon", "alpha", "num_steps", "norm", "random_start", "clip_min", "clip_max", "targeted", "target_label"]),
        ("image", "deepfool", ["max_iter", "overshoot", "top_k", "epsilon", "clip_min", "clip_max"]),
        ("image", "cw", ["confidence", "learning_rate", "max_iterations", "binary_search_steps", "initial_const", "targeted", "target_label", "clip_min", "clip_max"]),
        ("image", "fab", ["norm", "n_restarts", "n_iter", "eps", "top_k", "alpha_max", "eta", "beta", "targeted", "target_label", "clip_min", "clip_max"]),
        # Tabular
        ("tabular", "cap_pgd", ["epsilon", "alpha", "num_steps", "norm", "targeted", "target_label"]),
        ("tabular", "lowprofool", ["alpha", "num_steps", "lambda_weight", "budget", "targeted", "target_label"]),
        ("tabular", "tabular_cw", ["c_penalty", "confidence", "learning_rate", "max_iterations", "targeted", "target_label"]),
        ("tabular", "tabular_jsma", ["theta", "max_features", "targeted", "target_label"]),
        ("tabular", "vae_attack", ["eps_latent", "alpha", "num_steps", "recon_weight", "targeted", "target_label"]),
        # Time-series
        ("time_series", "time_series_pgd", ["epsilon", "alpha", "num_steps", "norm", "random_start", "clip_min", "clip_max", "targeted", "target_label"]),
        ("time_series", "smooth_pgd", ["epsilon", "alpha", "num_steps", "norm", "tv_weight", "random_start", "clip_min", "clip_max", "targeted", "target_label"]),
        ("time_series", "wavelet_attack", ["epsilon", "alpha", "num_steps", "norm", "transform_type", "band", "clip_min", "clip_max", "targeted", "target_label"]),
        ("time_series", "soft_dtw_attack", ["epsilon", "alpha", "num_steps", "max_warp", "gamma", "lambda_dtw", "lambda_smooth", "downsample_threshold", "clip_min", "clip_max", "targeted", "target_label"]),
        ("time_series", "jsma", ["theta", "max_iter", "max_fraction", "increase", "clip_min", "clip_max", "targeted", "target_label"]),
        # Text
        ("text", "hotflip", ["max_modifications", "candidate_vocab_size", "targeted", "target_label"]),
        ("text", "embedding_pgd", ["epsilon", "alpha", "num_steps", "norm", "k_nearest", "max_modifications", "targeted", "target_label"]),
        ("text", "textbugger_wb", ["max_modifications", "bug_types", "targeted", "target_label"]),
        ("text", "universal_trigger", ["trigger_length", "num_iterations", "candidate_vocab_size", "placement", "targeted", "target_label"]),
        ("text", "gradient_substitution", ["max_modifications", "num_candidates", "similarity_threshold", "targeted", "target_label"]),
    ],
)
def test_all_attacks_param_specs_complete(domain, attack_name, expected_param_names):
    specs = get_attack_param_specs(domain, attack_name)
    assert len(specs) > 0

    spec_dict = get_attack_param_specs_dict(domain, attack_name)
    assert len(spec_dict) == len(specs)

    spec_names = [s.name for s in specs]
    assert spec_names == expected_param_names

    for spec in specs:
        assert isinstance(spec, ParameterSpec)
        assert spec.data_type in ("float", "int", "str", "bool", "list")
        assert spec.type == spec.data_type
        assert isinstance(spec.description, str) and len(spec.description) > 0
        assert spec.ui_control in ("slider", "number_input", "text_input", "select", "multiselect", "checkbox")

        # Test to_dict serialization
        serialized = spec.to_dict()
        assert serialized["name"] == spec.name
        assert serialized["data_type"] == spec.data_type
        assert serialized["type"] == spec.data_type
        assert serialized["default"] == spec.default


@pytest.mark.parametrize(
    "domain,alias_input,canonical_name",
    [
        ("image", "CW", "cw"),
        ("image", "carlini_wagner", "cw"),
        ("image", "carlini-wagner", "cw"),
        ("image", "DeepFool", "deepfool"),
        ("tabular", "CapPGD", "cap_pgd"),
        ("tabular", "cappgd", "cap_pgd"),
        ("tabular", "tabular_pgd", "cap_pgd"),
        ("tabular", "TabularCW", "tabular_cw"),
        ("tabular", "carlini_wagner_tabular", "tabular_cw"),
        ("tabular", "cw_tabular", "tabular_cw"),
        ("tabular", "TabularJSMA", "tabular_jsma"),
        ("tabular", "jsma_tabular", "tabular_jsma"),
        ("tabular", "VAEAttack", "vae_attack"),
        ("tabular", "tabular_vae", "vae_attack"),
        ("time_series", "TimeSeriesPGD", "time_series_pgd"),
        ("time_series", "ts_pgd", "time_series_pgd"),
        ("time_series", "ts-pgd", "time_series_pgd"),
        ("time_series", "pgd", "time_series_pgd"),
        ("time_series", "SmoothPGD", "smooth_pgd"),
        ("time_series", "ts_smooth_pgd", "smooth_pgd"),
        ("time_series", "WaveletAttack", "wavelet_attack"),
        ("time_series", "wavelet", "wavelet_attack"),
        ("time_series", "ts_wavelet", "wavelet_attack"),
        ("time_series", "spectral", "wavelet_attack"),
        ("time_series", "SoftDTWAttack", "soft_dtw_attack"),
        ("time_series", "soft_dtw", "soft_dtw_attack"),
        ("time_series", "dtw", "soft_dtw_attack"),
        ("time_series", "JSMA", "jsma"),
        ("time_series", "ts_jsma", "jsma"),
        ("text", "HotFlip", "hotflip"),
        ("text", "EmbeddingPGD", "embedding_pgd"),
        ("text", "embeddingpgd", "embedding_pgd"),
        ("text", "TextBuggerWB", "textbugger_wb"),
        ("text", "textbugger", "textbugger_wb"),
        ("text", "UniversalTrigger", "universal_trigger"),
        ("text", "GradientSubstitution", "gradient_substitution"),
    ],
)
def test_attack_alias_resolution(domain, alias_input, canonical_name):
    resolved = normalize_attack_name(domain, alias_input)
    assert resolved == canonical_name
    specs = get_attack_param_specs(domain, alias_input)
    assert len(specs) > 0


def test_unknown_attack_error():
    with pytest.raises(AttackConfigurationError) as exc_info:
        get_attack_param_specs("image", "non_existent_attack")
    assert "Unsupported or unknown attack" in str(exc_info.value)


# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================

def test_validate_attack_params_empty():
    validated = validate_attack_params("image", "fgsm", {})
    assert validated == {}

    validated_none = validate_attack_params("image", "fgsm", None)
    assert validated_none == {}


def test_validate_attack_params_fill_defaults():
    validated = validate_attack_params("image", "fgsm", {}, fill_defaults=True)
    assert validated["epsilon"] == 0.1
    assert validated["clip_min"] == 0.0
    assert validated["clip_max"] == 1.0
    assert validated["targeted"] is False
    assert validated["target_label"] is None


def test_validate_attack_params_aliases_pgd():
    raw_params = {
        "steps": 25,
        "step_size": 0.02,
        "norm": "Linf",
    }
    validated = validate_attack_params("image", "pgd", raw_params)
    assert validated["num_steps"] == 25
    assert validated["alpha"] == 0.02
    assert validated["norm"] == "Linf"
    assert "steps" not in validated
    assert "step_size" not in validated


def test_validate_attack_params_norm_normalization():
    # Test "linf", "INF", "l2", "2"
    v1 = validate_attack_params("image", "pgd", {"norm": "linf"})
    assert v1["norm"] == "Linf"

    v2 = validate_attack_params("image", "pgd", {"norm": "INF"})
    assert v2["norm"] == "Linf"

    v3 = validate_attack_params("image", "pgd", {"norm": "l2"})
    assert v3["norm"] == "L2"

    v4 = validate_attack_params("image", "pgd", {"norm": "2"})
    assert v4["norm"] == "L2"

    with pytest.raises(AttackConfigurationError):
        validate_attack_params("image", "pgd", {"norm": "L1"})


def test_validate_attack_params_type_checks():
    # num_steps must be int
    with pytest.raises(AttackConfigurationError) as exc:
        validate_attack_params("image", "pgd", {"num_steps": "10"})
    assert "must be an integer" in str(exc.value)

    # bool is not accepted as int
    with pytest.raises(AttackConfigurationError):
        validate_attack_params("image", "pgd", {"num_steps": True})

    # epsilon must be float/int, not bool or str
    with pytest.raises(AttackConfigurationError):
        validate_attack_params("image", "pgd", {"epsilon": True})

    with pytest.raises(AttackConfigurationError):
        validate_attack_params("image", "pgd", {"epsilon": "0.1"})

    # targeted must be bool
    with pytest.raises(AttackConfigurationError):
        validate_attack_params("image", "pgd", {"targeted": "True"})


def test_validate_attack_params_boundary_checks():
    # Negative epsilon
    with pytest.raises(AttackConfigurationError):
        validate_attack_params("image", "fgsm", {"epsilon": -0.01})

    # Step size <= 0
    with pytest.raises(AttackConfigurationError):
        validate_attack_params("image", "pgd", {"alpha": 0.0})

    with pytest.raises(AttackConfigurationError):
        validate_attack_params("image", "pgd", {"alpha": -0.05})

    # FAB beta must be strictly between 0 and 1
    with pytest.raises(AttackConfigurationError):
        validate_attack_params("image", "fab", {"beta": 0.0})

    with pytest.raises(AttackConfigurationError):
        validate_attack_params("image", "fab", {"beta": 1.0})

    with pytest.raises(AttackConfigurationError):
        validate_attack_params("image", "fab", {"beta": 1.5})

    valid_fab = validate_attack_params("image", "fab", {"beta": 0.5})
    assert valid_fab["beta"] == 0.5

    # FAB eta >= 1.0
    with pytest.raises(AttackConfigurationError):
        validate_attack_params("image", "fab", {"eta": 0.99})

    # JSMA theta cannot be 0
    with pytest.raises(AttackConfigurationError):
        validate_attack_params("time_series", "jsma", {"theta": 0.0})

    # JSMA max_fraction must be in (0, 1]
    with pytest.raises(AttackConfigurationError):
        validate_attack_params("time_series", "jsma", {"max_fraction": 0.0})

    with pytest.raises(AttackConfigurationError):
        validate_attack_params("time_series", "jsma", {"max_fraction": 1.1})


def test_validate_attack_params_allowed_values():
    # Wavelet transform_type
    with pytest.raises(AttackConfigurationError):
        validate_attack_params("time_series", "wavelet_attack", {"transform_type": "invalid"})

    # Wavelet band
    with pytest.raises(AttackConfigurationError):
        validate_attack_params("time_series", "wavelet_attack", {"band": "ultrasound"})

    # Universal trigger placement
    with pytest.raises(AttackConfigurationError):
        validate_attack_params("text", "universal_trigger", {"placement": "middle"})

    # TextBugger bug_types
    with pytest.raises(AttackConfigurationError):
        validate_attack_params("text", "textbugger_wb", {"bug_types": ["invalid_bug"]})

    with pytest.raises(AttackConfigurationError):
        validate_attack_params("text", "textbugger_wb", {"bug_types": []})


def test_validate_attack_params_clip_consistency():
    # clip_min > clip_max
    with pytest.raises(AttackConfigurationError) as exc:
        validate_attack_params("image", "fgsm", {"clip_min": 1.0, "clip_max": 0.0})
    assert "cannot be greater than" in str(exc.value)

    # Valid clip
    v = validate_attack_params("image", "fgsm", {"clip_min": 0.0, "clip_max": 1.0})
    assert v["clip_min"] == 0.0
    assert v["clip_max"] == 1.0


def test_validate_attack_params_conditional_visibility():
    # target_label without targeted=True should fail
    with pytest.raises(AttackConfigurationError) as exc:
        validate_attack_params("image", "fgsm", {"target_label": 3})
    assert "Set 'targeted=True'" in str(exc.value)

    # target_label with targeted=True should succeed
    v = validate_attack_params("image", "fgsm", {"targeted": True, "target_label": 3})
    assert v["targeted"] is True
    assert v["target_label"] == 3


def test_validate_attack_params_unknown_param_error():
    with pytest.raises(AttackConfigurationError) as exc:
        validate_attack_params("image", "fgsm", {"magic_number": 42})
    assert "Unknown parameter 'magic_number'" in str(exc.value)


def test_validate_attack_params_runtime_context_separation():
    # Runtime context keys such as tokenizer, loss_fn, device should pass through
    dummy_tokenizer = object()
    dummy_loss = object()
    params = {
        "max_modifications": 3,
        "tokenizer": dummy_tokenizer,
        "loss_fn": dummy_loss,
        "device": "cpu",
    }
    validated = validate_attack_params("text", "hotflip", params)
    assert validated["max_modifications"] == 3
    assert validated["tokenizer"] is dummy_tokenizer
    assert validated["loss_fn"] is dummy_loss
    assert validated["device"] == "cpu"


def test_runtime_context_keys_constant():
    assert "tokenizer" in RUNTIME_CONTEXT_KEYS
    assert "loss_fn" in RUNTIME_CONTEXT_KEYS
    assert "device" in RUNTIME_CONTEXT_KEYS
    assert "vae" in RUNTIME_CONTEXT_KEYS
    assert "bounds" in RUNTIME_CONTEXT_KEYS

