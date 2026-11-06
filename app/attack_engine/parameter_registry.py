"""
Centralized attack parameter registry for the Adversarial Attack Engine.

Provides specifications, validation rules, default configurations, and UI controls
for all supported adversarial attacks across image, tabular, time-series, and text domains.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union
from app.attack_engine.exceptions import AttackConfigurationError


@dataclass
class ParameterSpec:
    """
    Standardized specification for an adversarial attack parameter.

    Attributes:
        name: Exact parameter identifier used in the attack implementation.
        data_type: Expected primitive data type ("float", "int", "str", "bool", "list").
        default: Existing default value in the attack implementation.
        description: Human-readable description of the parameter.
        required: Whether the parameter is strictly required to be provided by the caller.
        min_value: Optional minimum numerical value.
        max_value: Optional maximum numerical value.
        exclusive_min: Whether the minimum bound is exclusive (> min_value instead of >=).
        exclusive_max: Whether the maximum bound is exclusive (< max_value instead of <=).
        allowed_values: Optional list/set of allowable categorical values.
        conditional_visibility: Optional condition for UI visibility (e.g. {"targeted": True}).
        aliases: Alternative parameter names mapped to this canonical parameter.
        ui_control: Recommended UI input control ("slider", "number_input", "text_input", "select", "multiselect", "checkbox").
    """

    name: str
    data_type: str
    default: Any
    description: str = ""
    required: bool = False
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    exclusive_min: bool = False
    exclusive_max: bool = False
    allowed_values: Optional[List[Any]] = None
    conditional_visibility: Optional[Dict[str, Any]] = None
    aliases: List[str] = field(default_factory=list)
    ui_control: str = "number_input"

    @property
    def type(self) -> str:
        """Alias for data_type for backwards and external compatibility."""
        return self.data_type

    def to_dict(self) -> Dict[str, Any]:
        """Convert specification to a JSON-serializable dictionary."""
        return {
            "name": self.name,
            "data_type": self.data_type,
            "type": self.data_type,
            "default": self.default,
            "description": self.description,
            "required": self.required,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "exclusive_min": self.exclusive_min,
            "exclusive_max": self.exclusive_max,
            "allowed_values": list(self.allowed_values) if self.allowed_values is not None else None,
            "conditional_visibility": self.conditional_visibility,
            "aliases": list(self.aliases),
            "ui_control": self.ui_control,
        }


# Canonical supported domains
SUPPORTED_DOMAINS: List[str] = ["image", "tabular", "time_series", "text"]

# Domain alias normalizer
_DOMAIN_ALIASES: Dict[str, str] = {
    "image": "image",
    "images": "image",
    "vision": "image",
    "tabular": "tabular",
    "tables": "tabular",
    "table": "tabular",
    "time_series": "time_series",
    "time-series": "time_series",
    "timeseries": "time_series",
    "ts": "time_series",
    "text": "text",
    "nlp": "text",
}

# Canonical attack lists per domain
_SUPPORTED_ATTACKS: Dict[str, List[str]] = {
    "image": ["fgsm", "pgd", "deepfool", "cw", "fab"],
    "tabular": ["cap_pgd", "lowprofool", "tabular_cw", "tabular_jsma", "vae_attack"],
    "time_series": ["time_series_pgd", "smooth_pgd", "wavelet_attack", "soft_dtw_attack", "jsma"],
    "text": ["hotflip", "embedding_pgd", "textbugger_wb", "universal_trigger", "gradient_substitution"],
}

# Display / Class name mappings per domain
_ATTACK_DISPLAY_NAMES: Dict[str, Dict[str, str]] = {
    "image": {
        "fgsm": "FGSM",
        "pgd": "PGD",
        "deepfool": "DeepFool",
        "cw": "CW",
        "fab": "FAB",
    },
    "tabular": {
        "cap_pgd": "CapPGD",
        "lowprofool": "LowProFool",
        "tabular_cw": "TabularCW",
        "tabular_jsma": "TabularJSMA",
        "vae_attack": "VAEAttack",
    },
    "time_series": {
        "time_series_pgd": "TimeSeriesPGD",
        "smooth_pgd": "SmoothPGD",
        "wavelet_attack": "WaveletAttack",
        "soft_dtw_attack": "SoftDTWAttack",
        "jsma": "JSMA",
    },
    "text": {
        "hotflip": "HotFlip",
        "embedding_pgd": "EmbeddingPGD",
        "textbugger_wb": "TextBuggerWB",
        "universal_trigger": "UniversalTrigger",
        "gradient_substitution": "GradientSubstitution",
    },
}

# Attack name aliases per domain
_ATTACK_ALIASES: Dict[str, Dict[str, str]] = {
    "image": {
        "fgsm": "fgsm",
        "pgd": "pgd",
        "deepfool": "deepfool",
        "cw": "cw",
        "carlini_wagner": "cw",
        "carlini-wagner": "cw",
        "carliniwagner": "cw",
        "fab": "fab",
    },
    "tabular": {
        "cap_pgd": "cap_pgd",
        "cappgd": "cap_pgd",
        "tabular_pgd": "cap_pgd",
        "lowprofool": "lowprofool",
        "tabular_cw": "tabular_cw",
        "tabularcw": "tabular_cw",
        "carlini_wagner_tabular": "tabular_cw",
        "cw_tabular": "tabular_cw",
        "tabular_jsma": "tabular_jsma",
        "tabularjsma": "tabular_jsma",
        "jsma_tabular": "tabular_jsma",
        "vae_attack": "vae_attack",
        "vaeattack": "vae_attack",
        "tabular_vae": "vae_attack",
    },
    "time_series": {
        "time_series_pgd": "time_series_pgd",
        "timeseriespgd": "time_series_pgd",
        "ts_pgd": "time_series_pgd",
        "ts-pgd": "time_series_pgd",
        "pgd": "time_series_pgd",
        "smooth_pgd": "smooth_pgd",
        "smoothpgd": "smooth_pgd",
        "smooth-pgd": "smooth_pgd",
        "ts_smooth_pgd": "smooth_pgd",
        "temporal_pgd": "smooth_pgd",
        "wavelet_attack": "wavelet_attack",
        "wavelet": "wavelet_attack",
        "waveletattack": "wavelet_attack",
        "ts_wavelet": "wavelet_attack",
        "spectral": "wavelet_attack",
        "soft_dtw_attack": "soft_dtw_attack",
        "soft_dtw": "soft_dtw_attack",
        "softdtw": "soft_dtw_attack",
        "softdtwattack": "soft_dtw_attack",
        "ts_soft_dtw": "soft_dtw_attack",
        "dtw": "soft_dtw_attack",
        "jsma": "jsma",
        "ts_jsma": "jsma",
        "time_series_jsma": "jsma",
    },
    "text": {
        "hotflip": "hotflip",
        "embedding_pgd": "embedding_pgd",
        "embeddingpgd": "embedding_pgd",
        "textbugger_wb": "textbugger_wb",
        "textbugger": "textbugger_wb",
        "textbuggerwb": "textbugger_wb",
        "universal_trigger": "universal_trigger",
        "universaltrigger": "universal_trigger",
        "gradient_substitution": "gradient_substitution",
        "gradientsubstitution": "gradient_substitution",
    },
}

# Runtime dependencies and dataset constraints kept separate from attack hyperparameters
RUNTIME_CONTEXT_KEYS: Set[str] = {
    # Models, tokenizers, auxiliary encoders
    "tokenizer",
    "embedding_layer",
    "vae",
    "vae_model",
    "device",
    "loss_fn",
    # Tabular dataset constraints and metadata
    "feature_metadata",
    "tabular_metadata",
    "immutable_features",
    "categorical_features",
    "integer_features",
    "binary_features",
    "feature_names",
    "feature_importance",
    "importance_weights",
    "feature_perturbation_limits",
    "perturbation_limits",
    "distance_metric",
    "random_init",
    # Time-series constraints and options
    "feature_bounds",
    "bounds",
    "feature_min",
    "feature_max",
    "tv_norm",
    "regularize_perturbation",
    "max_allowed_delta",
    "max_delta",
    # Text tokenization constraints
    "max_length",
}


# =============================================================================
# ATTACK PARAMETER REGISTRY DICTIONARY
# =============================================================================

_ATTACK_PARAM_REGISTRY: Dict[str, Dict[str, List[ParameterSpec]]] = {
    # -------------------------------------------------------------------------
    # IMAGE DOMAIN
    # -------------------------------------------------------------------------
    "image": {
        "fgsm": [
            ParameterSpec(
                name="epsilon",
                data_type="float",
                default=0.1,
                min_value=0.0,
                description="Perturbation budget (magnitude) under Linf norm",
                required=False,
                ui_control="slider",
            ),
            ParameterSpec(
                name="clip_min",
                data_type="float",
                default=0.0,
                description="Minimum value bound for pixel clipping",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="clip_max",
                data_type="float",
                default=1.0,
                description="Maximum value bound for pixel clipping",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="targeted",
                data_type="bool",
                default=False,
                description="Whether to execute a targeted attack towards target_label",
                required=False,
                ui_control="checkbox",
            ),
            ParameterSpec(
                name="target_label",
                data_type="int",
                default=None,
                description="Target class index for targeted attack",
                required=False,
                conditional_visibility={"targeted": True},
                aliases=["target"],
                ui_control="number_input",
            ),
        ],
        "pgd": [
            ParameterSpec(
                name="epsilon",
                data_type="float",
                default=0.1,
                min_value=0.0,
                description="Maximum perturbation budget under chosen norm",
                required=False,
                ui_control="slider",
            ),
            ParameterSpec(
                name="alpha",
                data_type="float",
                default=0.01,
                min_value=0.0,
                exclusive_min=True,
                description="Step size for each gradient iteration",
                required=False,
                aliases=["step_size"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="num_steps",
                data_type="int",
                default=40,
                min_value=1,
                description="Number of iterative gradient steps",
                required=False,
                aliases=["steps", "iters", "iterations", "max_iter", "max_iterations"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="norm",
                data_type="str",
                default="Linf",
                allowed_values=["Linf", "L2"],
                description="Distance norm for perturbation projection",
                required=False,
                ui_control="select",
            ),
            ParameterSpec(
                name="random_start",
                data_type="bool",
                default=True,
                description="Whether to initialize perturbation randomly within epsilon ball",
                required=False,
                ui_control="checkbox",
            ),
            ParameterSpec(
                name="clip_min",
                data_type="float",
                default=0.0,
                description="Minimum pixel clip bound",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="clip_max",
                data_type="float",
                default=1.0,
                description="Maximum pixel clip bound",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="targeted",
                data_type="bool",
                default=False,
                description="Whether to execute targeted attack",
                required=False,
                ui_control="checkbox",
            ),
            ParameterSpec(
                name="target_label",
                data_type="int",
                default=None,
                description="Target class index for targeted attack",
                required=False,
                conditional_visibility={"targeted": True},
                aliases=["target"],
                ui_control="number_input",
            ),
        ],
        "deepfool": [
            ParameterSpec(
                name="max_iter",
                data_type="int",
                default=50,
                min_value=1,
                description="Maximum iterations to reach decision boundary",
                required=False,
                aliases=["num_steps", "steps"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="overshoot",
                data_type="float",
                default=0.02,
                min_value=0.0,
                exclusive_min=True,
                description="Overshoot factor beyond the decision boundary",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="top_k",
                data_type="int",
                default=10,
                min_value=1,
                description="Number of nearest classes to evaluate at each step",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="epsilon",
                data_type="float",
                default=None,
                min_value=0.0,
                description="Optional maximum perturbation radius epsilon cap (unbounded search if None)",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="clip_min",
                data_type="float",
                default=0.0,
                description="Minimum pixel clip bound",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="clip_max",
                data_type="float",
                default=1.0,
                description="Maximum pixel clip bound",
                required=False,
                ui_control="number_input",
            ),
        ],
        "cw": [
            ParameterSpec(
                name="confidence",
                data_type="float",
                default=0.0,
                min_value=0.0,
                description="Confidence parameter (kappa) for margin between logits",
                required=False,
                aliases=["kappa"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="learning_rate",
                data_type="float",
                default=0.01,
                min_value=0.0,
                exclusive_min=True,
                description="Optimizer learning rate for perturbation optimization",
                required=False,
                aliases=["lr"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="max_iterations",
                data_type="int",
                default=20,
                min_value=1,
                description="Optimization steps per binary search step",
                required=False,
                aliases=["iters", "num_steps", "steps"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="binary_search_steps",
                data_type="int",
                default=1,
                min_value=1,
                description="Number of binary search steps for trade-off constant c",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="initial_const",
                data_type="float",
                default=1.0,
                min_value=0.0,
                exclusive_min=True,
                description="Initial trade-off constant c for binary search",
                required=False,
                aliases=["c"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="targeted",
                data_type="bool",
                default=False,
                description="Whether to execute targeted attack",
                required=False,
                ui_control="checkbox",
            ),
            ParameterSpec(
                name="target_label",
                data_type="int",
                default=None,
                description="Target class index for targeted attack",
                required=False,
                conditional_visibility={"targeted": True},
                aliases=["target"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="clip_min",
                data_type="float",
                default=0.0,
                description="Minimum pixel clip bound",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="clip_max",
                data_type="float",
                default=1.0,
                description="Maximum pixel clip bound",
                required=False,
                ui_control="number_input",
            ),
        ],
        "fab": [
            ParameterSpec(
                name="norm",
                data_type="str",
                default="Linf",
                allowed_values=["Linf", "L2"],
                description="Distance norm for projection",
                required=False,
                ui_control="select",
            ),
            ParameterSpec(
                name="n_restarts",
                data_type="int",
                default=1,
                min_value=1,
                description="Number of random restarts from diverse points",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="n_iter",
                data_type="int",
                default=100,
                min_value=1,
                description="Number of iterations per restart",
                required=False,
                aliases=["max_iter", "num_steps"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="eps",
                data_type="float",
                default=None,
                min_value=0.0,
                description="Optional maximum perturbation radius epsilon (unbounded search if None)",
                required=False,
                aliases=["epsilon"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="top_k",
                data_type="int",
                default=None,
                min_value=1,
                description="Number of nearest classes to evaluate at each step",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="alpha_max",
                data_type="float",
                default=0.1,
                min_value=0.0,
                exclusive_min=True,
                description="Maximum step size interpolation factor",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="eta",
                data_type="float",
                default=1.05,
                min_value=1.0,
                description="Overshoot parameter >= 1.0",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="beta",
                data_type="float",
                default=0.5,
                min_value=0.0,
                max_value=1.0,
                exclusive_min=True,
                exclusive_max=True,
                description="Backward step interpolation factor along ray (0 < beta < 1)",
                required=False,
                ui_control="slider",
            ),
            ParameterSpec(
                name="targeted",
                data_type="bool",
                default=False,
                description="Whether to execute targeted attack",
                required=False,
                ui_control="checkbox",
            ),
            ParameterSpec(
                name="target_label",
                data_type="int",
                default=None,
                description="Target class index for targeted attack",
                required=False,
                conditional_visibility={"targeted": True},
                aliases=["target", "target_labels"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="clip_min",
                data_type="float",
                default=0.0,
                description="Minimum pixel clip bound",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="clip_max",
                data_type="float",
                default=1.0,
                description="Maximum pixel clip bound",
                required=False,
                ui_control="number_input",
            ),
        ],
    },

    # -------------------------------------------------------------------------
    # TABULAR DOMAIN
    # -------------------------------------------------------------------------
    "tabular": {
        "cap_pgd": [
            ParameterSpec(
                name="epsilon",
                data_type="float",
                default=0.1,
                min_value=0.0,
                description="Maximum perturbation budget under chosen norm",
                required=False,
                ui_control="slider",
            ),
            ParameterSpec(
                name="alpha",
                data_type="float",
                default=0.05,
                min_value=0.0,
                exclusive_min=True,
                description="Step size per gradient iteration",
                required=False,
                aliases=["step_size"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="num_steps",
                data_type="int",
                default=10,
                min_value=1,
                description="Number of PGD optimization iterations",
                required=False,
                aliases=["steps", "iterations", "iters", "max_iter", "max_iterations"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="norm",
                data_type="str",
                default="Linf",
                allowed_values=["Linf", "L2"],
                description="Distance norm for projection",
                required=False,
                ui_control="select",
            ),
            ParameterSpec(
                name="targeted",
                data_type="bool",
                default=False,
                description="Whether attack is targeted",
                required=False,
                ui_control="checkbox",
            ),
            ParameterSpec(
                name="target_label",
                data_type="int",
                default=None,
                description="Target class index for targeted attack",
                required=False,
                conditional_visibility={"targeted": True},
                aliases=["target"],
                ui_control="number_input",
            ),
        ],
        "lowprofool": [
            ParameterSpec(
                name="alpha",
                data_type="float",
                default=0.05,
                min_value=0.0,
                exclusive_min=True,
                description="Step size for iterative feature updates",
                required=False,
                aliases=["step_size"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="num_steps",
                data_type="int",
                default=20,
                min_value=1,
                description="Number of iterative optimization steps",
                required=False,
                aliases=["steps", "iterations", "iters", "max_iter", "max_iterations"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="lambda_weight",
                data_type="float",
                default=0.1,
                min_value=0.0,
                description="Trade-off weight between adversarial loss and feature distance penalty",
                required=False,
                aliases=["tradeoff"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="budget",
                data_type="float",
                default=None,
                min_value=0.0,
                exclusive_min=True,
                description="Optional max weighted feature perturbation budget",
                required=False,
                aliases=["epsilon"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="targeted",
                data_type="bool",
                default=False,
                description="Whether attack is targeted",
                required=False,
                ui_control="checkbox",
            ),
            ParameterSpec(
                name="target_label",
                data_type="int",
                default=None,
                description="Target class index for targeted attack",
                required=False,
                conditional_visibility={"targeted": True},
                aliases=["target"],
                ui_control="number_input",
            ),
        ],
        "tabular_cw": [
            ParameterSpec(
                name="c_penalty",
                data_type="float",
                default=1.0,
                min_value=0.0,
                exclusive_min=True,
                description="Trade-off constant balancing distance and classification loss",
                required=False,
                aliases=["c", "initial_const"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="confidence",
                data_type="float",
                default=0.0,
                min_value=0.0,
                description="Margin confidence requirement for adversarial classification",
                required=False,
                aliases=["kappa"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="learning_rate",
                data_type="float",
                default=0.05,
                min_value=0.0,
                exclusive_min=True,
                description="Optimizer learning rate",
                required=False,
                aliases=["lr"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="max_iterations",
                data_type="int",
                default=50,
                min_value=1,
                description="Number of Adam optimization iterations",
                required=False,
                aliases=["num_steps", "steps", "iters", "iterations", "max_iter"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="targeted",
                data_type="bool",
                default=False,
                description="Whether attack is targeted",
                required=False,
                ui_control="checkbox",
            ),
            ParameterSpec(
                name="target_label",
                data_type="int",
                default=None,
                description="Target class index for targeted attack",
                required=False,
                conditional_visibility={"targeted": True},
                aliases=["target"],
                ui_control="number_input",
            ),
        ],
        "tabular_jsma": [
            ParameterSpec(
                name="theta",
                data_type="float",
                default=0.1,
                min_value=0.0,
                exclusive_min=True,
                description="Feature perturbation step size",
                required=False,
                aliases=["step_size"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="max_features",
                data_type="int",
                default=5,
                min_value=1,
                description="Maximum number of distinct tabular features to modify",
                required=False,
                aliases=["max_modifications"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="targeted",
                data_type="bool",
                default=False,
                description="Whether attack is targeted",
                required=False,
                ui_control="checkbox",
            ),
            ParameterSpec(
                name="target_label",
                data_type="int",
                default=None,
                description="Target class index for targeted attack",
                required=False,
                conditional_visibility={"targeted": True},
                aliases=["target"],
                ui_control="number_input",
            ),
        ],
        "vae_attack": [
            ParameterSpec(
                name="eps_latent",
                data_type="float",
                default=0.5,
                min_value=0.0,
                description="Latent space perturbation budget epsilon",
                required=False,
                aliases=["epsilon"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="alpha",
                data_type="float",
                default=0.05,
                min_value=0.0,
                exclusive_min=True,
                description="Step size in latent space optimization",
                required=False,
                aliases=["step_size"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="num_steps",
                data_type="int",
                default=20,
                min_value=1,
                description="Number of latent space gradient steps",
                required=False,
                aliases=["steps", "iterations", "iters", "max_iter", "max_iterations"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="recon_weight",
                data_type="float",
                default=0.1,
                min_value=0.0,
                description="Weight on VAE reconstruction penalty",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="targeted",
                data_type="bool",
                default=False,
                description="Whether attack is targeted",
                required=False,
                ui_control="checkbox",
            ),
            ParameterSpec(
                name="target_label",
                data_type="int",
                default=None,
                description="Target class index for targeted attack",
                required=False,
                conditional_visibility={"targeted": True},
                aliases=["target"],
                ui_control="number_input",
            ),
        ],
    },

    # -------------------------------------------------------------------------
    # TIME-SERIES DOMAIN
    # -------------------------------------------------------------------------
    "time_series": {
        "time_series_pgd": [
            ParameterSpec(
                name="epsilon",
                data_type="float",
                default=0.1,
                min_value=0.0,
                description="Maximum perturbation budget under chosen norm",
                required=False,
                ui_control="slider",
            ),
            ParameterSpec(
                name="alpha",
                data_type="float",
                default=0.01,
                min_value=0.0,
                exclusive_min=True,
                description="Step size per iteration",
                required=False,
                aliases=["step_size"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="num_steps",
                data_type="int",
                default=10,
                min_value=1,
                description="Number of gradient iterations",
                required=False,
                aliases=["steps", "iters", "iterations", "max_iter", "max_iterations"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="norm",
                data_type="str",
                default="Linf",
                allowed_values=["Linf", "L2"],
                description="Distance norm for projection",
                required=False,
                ui_control="select",
            ),
            ParameterSpec(
                name="random_start",
                data_type="bool",
                default=True,
                description="Whether to initialize perturbation randomly within epsilon ball",
                required=False,
                ui_control="checkbox",
            ),
            ParameterSpec(
                name="clip_min",
                data_type="float",
                default=None,
                description="Optional minimum value bound for clipping",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="clip_max",
                data_type="float",
                default=None,
                description="Optional maximum value bound for clipping",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="targeted",
                data_type="bool",
                default=False,
                description="Whether attack is targeted",
                required=False,
                ui_control="checkbox",
            ),
            ParameterSpec(
                name="target_label",
                data_type="int",
                default=None,
                description="Target class index for targeted attack",
                required=False,
                conditional_visibility={"targeted": True},
                aliases=["target"],
                ui_control="number_input",
            ),
        ],
        "smooth_pgd": [
            ParameterSpec(
                name="epsilon",
                data_type="float",
                default=0.1,
                min_value=0.0,
                description="Maximum perturbation budget",
                required=False,
                ui_control="slider",
            ),
            ParameterSpec(
                name="alpha",
                data_type="float",
                default=0.01,
                min_value=0.0,
                exclusive_min=True,
                description="Step size per iteration",
                required=False,
                aliases=["step_size"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="num_steps",
                data_type="int",
                default=10,
                min_value=1,
                description="Number of gradient iterations",
                required=False,
                aliases=["steps", "iters", "iterations", "max_iter", "max_iterations"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="norm",
                data_type="str",
                default="Linf",
                allowed_values=["Linf", "L2"],
                description="Distance norm for projection",
                required=False,
                ui_control="select",
            ),
            ParameterSpec(
                name="tv_weight",
                data_type="float",
                default=0.1,
                min_value=0.0,
                description="Total Variation smoothness penalty weight",
                required=False,
                aliases=["lambda_tv", "smoothness_factor"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="random_start",
                data_type="bool",
                default=True,
                description="Whether to initialize perturbation randomly within epsilon ball",
                required=False,
                ui_control="checkbox",
            ),
            ParameterSpec(
                name="clip_min",
                data_type="float",
                default=None,
                description="Optional minimum value bound for clipping",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="clip_max",
                data_type="float",
                default=None,
                description="Optional maximum value bound for clipping",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="targeted",
                data_type="bool",
                default=False,
                description="Whether attack is targeted",
                required=False,
                ui_control="checkbox",
            ),
            ParameterSpec(
                name="target_label",
                data_type="int",
                default=None,
                description="Target class index for targeted attack",
                required=False,
                conditional_visibility={"targeted": True},
                aliases=["target"],
                ui_control="number_input",
            ),
        ],
        "wavelet_attack": [
            ParameterSpec(
                name="epsilon",
                data_type="float",
                default=0.1,
                min_value=0.0,
                description="Maximum perturbation budget in time/wavelet domain",
                required=False,
                ui_control="slider",
            ),
            ParameterSpec(
                name="alpha",
                data_type="float",
                default=0.01,
                min_value=0.0,
                exclusive_min=True,
                description="Step size per iteration",
                required=False,
                aliases=["step_size"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="num_steps",
                data_type="int",
                default=10,
                min_value=1,
                description="Number of gradient iterations",
                required=False,
                aliases=["steps", "iters", "iterations", "max_iter", "max_iterations"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="norm",
                data_type="str",
                default="Linf",
                allowed_values=["Linf", "L2"],
                description="Distance norm for projection",
                required=False,
                ui_control="select",
            ),
            ParameterSpec(
                name="transform_type",
                data_type="str",
                default="dwt",
                allowed_values=["dwt", "haar", "fft"],
                description="Frequency decomposition method",
                required=False,
                ui_control="select",
            ),
            ParameterSpec(
                name="band",
                data_type="str",
                default="all",
                allowed_values=["all", "low", "high", "mid", "approx", "detail"],
                description="Targeted frequency band",
                required=False,
                ui_control="select",
            ),
            ParameterSpec(
                name="clip_min",
                data_type="float",
                default=None,
                description="Optional minimum value bound for clipping",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="clip_max",
                data_type="float",
                default=None,
                description="Optional maximum value bound for clipping",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="targeted",
                data_type="bool",
                default=False,
                description="Whether attack is targeted",
                required=False,
                ui_control="checkbox",
            ),
            ParameterSpec(
                name="target_label",
                data_type="int",
                default=None,
                description="Target class index for targeted attack",
                required=False,
                conditional_visibility={"targeted": True},
                aliases=["target"],
                ui_control="number_input",
            ),
        ],
        "soft_dtw_attack": [
            ParameterSpec(
                name="epsilon",
                data_type="float",
                default=0.1,
                min_value=0.0,
                description="Maximum perturbation budget",
                required=False,
                ui_control="slider",
            ),
            ParameterSpec(
                name="alpha",
                data_type="float",
                default=0.01,
                min_value=0.0,
                exclusive_min=True,
                description="Step size per iteration",
                required=False,
                aliases=["step_size", "lr"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="num_steps",
                data_type="int",
                default=10,
                min_value=1,
                description="Number of gradient iterations",
                required=False,
                aliases=["steps", "iters", "iterations", "max_iter", "max_iterations"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="max_warp",
                data_type="float",
                default=0.1,
                min_value=0.0,
                description="Maximum temporal warp budget",
                required=False,
                aliases=["epsilon_warp", "epsilon"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="gamma",
                data_type="float",
                default=0.5,
                min_value=0.0,
                exclusive_min=True,
                description="Soft-DTW smoothing parameter",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="lambda_dtw",
                data_type="float",
                default=0.05,
                min_value=0.0,
                description="DTW shape loss penalty weight",
                required=False,
                aliases=["dtw_weight"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="lambda_smooth",
                data_type="float",
                default=0.01,
                min_value=0.0,
                description="Temporal smoothness penalty weight",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="downsample_threshold",
                data_type="int",
                default=64,
                min_value=1,
                description="Sequence length threshold to downsample for soft-DTW",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="clip_min",
                data_type="float",
                default=None,
                description="Optional minimum value bound for clipping",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="clip_max",
                data_type="float",
                default=None,
                description="Optional maximum value bound for clipping",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="targeted",
                data_type="bool",
                default=False,
                description="Whether attack is targeted",
                required=False,
                ui_control="checkbox",
            ),
            ParameterSpec(
                name="target_label",
                data_type="int",
                default=None,
                description="Target class index for targeted attack",
                required=False,
                conditional_visibility={"targeted": True},
                aliases=["target"],
                ui_control="number_input",
            ),
        ],
        "jsma": [
            ParameterSpec(
                name="theta",
                data_type="float",
                default=0.1,
                description="Perturbation step size added to selected saliency points (cannot be 0)",
                required=False,
                aliases=["step_size"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="max_iter",
                data_type="int",
                default=20,
                min_value=1,
                description="Maximum saliency modification steps",
                required=False,
                aliases=["max_iterations", "max_locations", "max_steps", "num_steps", "steps", "iters"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="max_fraction",
                data_type="float",
                default=1.0,
                min_value=0.0,
                max_value=1.0,
                exclusive_min=True,
                description="Maximum fraction of time-series points allowed to be modified",
                required=False,
                aliases=["max_features_ratio"],
                ui_control="slider",
            ),
            ParameterSpec(
                name="increase",
                data_type="bool",
                default=True,
                description="Whether to increase or decrease values at selected saliency coordinates",
                required=False,
                ui_control="checkbox",
            ),
            ParameterSpec(
                name="clip_min",
                data_type="float",
                default=None,
                description="Optional minimum value bound for clipping",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="clip_max",
                data_type="float",
                default=None,
                description="Optional maximum value bound for clipping",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="targeted",
                data_type="bool",
                default=False,
                description="Whether attack is targeted",
                required=False,
                ui_control="checkbox",
            ),
            ParameterSpec(
                name="target_label",
                data_type="int",
                default=None,
                description="Target class index for targeted attack",
                required=False,
                conditional_visibility={"targeted": True},
                aliases=["target"],
                ui_control="number_input",
            ),
        ],
    },

    # -------------------------------------------------------------------------
    # TEXT DOMAIN
    # -------------------------------------------------------------------------
    "text": {
        "hotflip": [
            ParameterSpec(
                name="max_modifications",
                data_type="int",
                default=5,
                min_value=1,
                description="Maximum number of word/token flips allowed",
                required=False,
                aliases=["max_flips"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="candidate_vocab_size",
                data_type="int",
                default=50,
                min_value=1,
                description="Top-k nearest gradient candidate vocabulary tokens to evaluate",
                required=False,
                aliases=["top_k_candidates"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="targeted",
                data_type="bool",
                default=False,
                description="Whether attack is targeted",
                required=False,
                ui_control="checkbox",
            ),
            ParameterSpec(
                name="target_label",
                data_type="int",
                default=None,
                description="Target class index for targeted attack",
                required=False,
                conditional_visibility={"targeted": True},
                aliases=["target"],
                ui_control="number_input",
            ),
        ],
        "embedding_pgd": [
            ParameterSpec(
                name="epsilon",
                data_type="float",
                default=0.5,
                min_value=0.0,
                description="Perturbation radius budget in embedding space",
                required=False,
                ui_control="slider",
            ),
            ParameterSpec(
                name="alpha",
                data_type="float",
                default=0.1,
                min_value=0.0,
                exclusive_min=True,
                description="Step size for embedding space iterations",
                required=False,
                aliases=["step_size"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="num_steps",
                data_type="int",
                default=10,
                min_value=1,
                description="Number of PGD iterations in embedding space",
                required=False,
                aliases=["steps", "iterations"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="norm",
                data_type="str",
                default="Linf",
                allowed_values=["Linf", "L2"],
                description="Distance norm for embedding perturbation",
                required=False,
                ui_control="select",
            ),
            ParameterSpec(
                name="k_nearest",
                data_type="int",
                default=1,
                min_value=1,
                description="Number of nearest token embedding neighbors to sample",
                required=False,
                aliases=["num_candidates"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="max_modifications",
                data_type="int",
                default=None,
                min_value=1,
                description="Optional limit on the number of token positions modified",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="targeted",
                data_type="bool",
                default=False,
                description="Whether attack is targeted",
                required=False,
                ui_control="checkbox",
            ),
            ParameterSpec(
                name="target_label",
                data_type="int",
                default=None,
                description="Target class index for targeted attack",
                required=False,
                conditional_visibility={"targeted": True},
                aliases=["target"],
                ui_control="number_input",
            ),
        ],
        "textbugger_wb": [
            ParameterSpec(
                name="max_modifications",
                data_type="int",
                default=5,
                min_value=1,
                description="Maximum number of words to bug/modify",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="bug_types",
                data_type="list",
                default=["insert", "delete", "swap", "sub_c", "sub_w"],
                allowed_values=["insert", "delete", "swap", "sub_c", "sub_w"],
                description="Allowed character/word bug perturbation operators",
                required=False,
                ui_control="multiselect",
            ),
            ParameterSpec(
                name="targeted",
                data_type="bool",
                default=False,
                description="Whether attack is targeted",
                required=False,
                ui_control="checkbox",
            ),
            ParameterSpec(
                name="target_label",
                data_type="int",
                default=None,
                description="Target class index for targeted attack",
                required=False,
                conditional_visibility={"targeted": True},
                aliases=["target"],
                ui_control="number_input",
            ),
        ],
        "universal_trigger": [
            ParameterSpec(
                name="trigger_length",
                data_type="int",
                default=3,
                min_value=1,
                description="Number of token positions in the universal trigger sequence",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="num_iterations",
                data_type="int",
                default=5,
                min_value=1,
                description="Number of optimization epochs over dataset tokens",
                required=False,
                aliases=["num_steps"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="candidate_vocab_size",
                data_type="int",
                default=20,
                min_value=1,
                description="Number of candidate tokens considered per trigger position",
                required=False,
                aliases=["num_candidates"],
                ui_control="number_input",
            ),
            ParameterSpec(
                name="placement",
                data_type="str",
                default="prefix",
                allowed_values=["prefix", "suffix"],
                description="Trigger placement location in text",
                required=False,
                ui_control="select",
            ),
            ParameterSpec(
                name="targeted",
                data_type="bool",
                default=False,
                description="Whether attack is targeted",
                required=False,
                ui_control="checkbox",
            ),
            ParameterSpec(
                name="target_label",
                data_type="int",
                default=None,
                description="Target class index for targeted attack",
                required=False,
                conditional_visibility={"targeted": True},
                aliases=["target"],
                ui_control="number_input",
            ),
        ],
        "gradient_substitution": [
            ParameterSpec(
                name="max_modifications",
                data_type="int",
                default=5,
                min_value=1,
                description="Maximum number of words replaced",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="num_candidates",
                data_type="int",
                default=10,
                min_value=1,
                description="Top candidates filtered by gradient first-order Taylor approximation",
                required=False,
                ui_control="number_input",
            ),
            ParameterSpec(
                name="similarity_threshold",
                data_type="float",
                default=0.4,
                min_value=0.0,
                max_value=1.0,
                description="Minimum cosine similarity required with original embedding",
                required=False,
                ui_control="slider",
            ),
            ParameterSpec(
                name="targeted",
                data_type="bool",
                default=False,
                description="Whether attack is targeted",
                required=False,
                ui_control="checkbox",
            ),
            ParameterSpec(
                name="target_label",
                data_type="int",
                default=None,
                description="Target class index for targeted attack",
                required=False,
                conditional_visibility={"targeted": True},
                aliases=["target"],
                ui_control="number_input",
            ),
        ],
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_domain(domain: str) -> str:
    """
    Normalize domain string to canonical form.

    Args:
        domain: Input domain identifier (e.g. 'image', 'Vision', 'time-series', 'ts').

    Returns:
        Canonical domain string.

    Raises:
        AttackConfigurationError: If domain is invalid or unsupported.
    """
    if not isinstance(domain, str):
        raise AttackConfigurationError(
            f"Domain must be a string, got {type(domain).__name__} ({domain})."
        )

    clean_domain = domain.lower().strip().replace("-", "_")
    canonical = _DOMAIN_ALIASES.get(clean_domain)
    if canonical is None:
        raise AttackConfigurationError(
            f"Unsupported domain '{domain}'. Supported domains: {SUPPORTED_DOMAINS}"
        )
    return canonical


def normalize_attack_name(domain: str, attack_name: str) -> str:
    """
    Normalize attack name string to canonical form for a given domain.

    Args:
        domain: Input domain identifier.
        attack_name: Input attack identifier (e.g. 'FGSM', 'TimeSeriesPGD', 'cw', 'carlini_wagner').

    Returns:
        Canonical lowercase attack identifier string.

    Raises:
        AttackConfigurationError: If domain or attack name is invalid.
    """
    norm_domain = normalize_domain(domain)
    if not isinstance(attack_name, str):
        raise AttackConfigurationError(
            f"Attack name must be a string, got {type(attack_name).__name__} ({attack_name})."
        )

    clean_attack = attack_name.lower().strip().replace("-", "_")
    domain_aliases = _ATTACK_ALIASES.get(norm_domain, {})

    if clean_attack in domain_aliases:
        return domain_aliases[clean_attack]

    # Try lookup without underscores (e.g. 'time_series_pgd' vs 'timeseriespgd')
    clean_no_us = clean_attack.replace("_", "")
    for alias_key, canonical in domain_aliases.items():
        if alias_key.replace("_", "") == clean_no_us:
            return canonical

    supported = get_supported_attacks(norm_domain)
    raise AttackConfigurationError(
        f"Unsupported or unknown attack '{attack_name}' for domain '{norm_domain}'. "
        f"Supported attacks: {supported}"
    )


# =============================================================================
# PUBLIC API FUNCTIONS
# =============================================================================

def get_supported_attacks(domain: str, return_display_names: bool = False) -> List[str]:
    """
    Get the list of supported attacks for a specified domain.

    Args:
        domain: Target input domain ('image', 'tabular', 'time_series', 'text').
        return_display_names: If True, returns attack class/display names (e.g. 'FGSM');
            if False (default), returns canonical lowercase identifiers (e.g. 'fgsm').

    Returns:
        List of attack identifier strings.

    Raises:
        AttackConfigurationError: If domain is invalid.
    """
    norm_domain = normalize_domain(domain)
    attacks = _SUPPORTED_ATTACKS[norm_domain]

    if return_display_names:
        display_map = _ATTACK_DISPLAY_NAMES.get(norm_domain, {})
        return [display_map.get(atk, atk.upper()) for atk in attacks]

    return list(attacks)


def get_attack_param_specs(domain: str, attack_name: str) -> List[ParameterSpec]:
    """
    Retrieve parameter specifications for a specific attack.

    Args:
        domain: Target domain ('image', 'tabular', 'time_series', 'text').
        attack_name: Attack name or alias (e.g. 'PGD', 'TimeSeriesPGD', 'cw').

    Returns:
        List of ParameterSpec instances.

    Raises:
        AttackConfigurationError: If domain or attack_name is invalid.
    """
    norm_domain = normalize_domain(domain)
    canonical_attack = normalize_attack_name(norm_domain, attack_name)

    domain_specs = _ATTACK_PARAM_REGISTRY.get(norm_domain, {})
    specs = domain_specs.get(canonical_attack)

    if specs is None:
        raise AttackConfigurationError(
            f"No parameter specification registered for attack '{attack_name}' in domain '{norm_domain}'."
        )

    # Return shallow copy of specs list to prevent mutation of internal registry
    return list(specs)


def get_attack_param_specs_dict(domain: str, attack_name: str) -> Dict[str, ParameterSpec]:
    """
    Retrieve parameter specifications mapped by canonical parameter name.

    Args:
        domain: Target domain ('image', 'tabular', 'time_series', 'text').
        attack_name: Attack name or alias.

    Returns:
        Dictionary mapping canonical parameter names to ParameterSpec instances.
    """
    specs = get_attack_param_specs(domain, attack_name)
    return {spec.name: spec for spec in specs}


def validate_attack_params(
    domain: str,
    attack_name: str,
    params: Optional[Dict[str, Any]] = None,
    fill_defaults: bool = False,
) -> Dict[str, Any]:
    """
    Validate and normalize attack parameters against the parameter registry.

    Performs type checks, boundary checks (min, max, exclusive ranges), allowed value
    checks, conditional visibility checks, and resolves parameter aliases to their canonical names.
    Preserves runtime dependencies and context objects without raising unknown parameter errors.

    Args:
        domain: Target domain ('image', 'tabular', 'time_series', 'text').
        attack_name: Attack name or alias.
        params: Dictionary of parameters to validate. If None, treated as empty dict.
        fill_defaults: If True, missing parameters are populated with their registry default values.

    Returns:
        Dictionary of validated and normalized parameters with aliases resolved.

    Raises:
        AttackConfigurationError: If domain, attack, or parameter values fail validation.
    """
    norm_domain = normalize_domain(domain)
    canonical_attack = normalize_attack_name(norm_domain, attack_name)
    specs = get_attack_param_specs(norm_domain, canonical_attack)

    if params is None:
        params = {}
    elif not isinstance(params, dict):
        raise AttackConfigurationError(
            f"Parameters must be a dictionary, got {type(params).__name__}."
        )

    spec_by_name: Dict[str, ParameterSpec] = {spec.name: spec for spec in specs}
    spec_by_alias: Dict[str, ParameterSpec] = {}
    for spec in specs:
        for alias in spec.aliases:
            spec_by_alias[alias] = spec

    validated_params: Dict[str, Any] = {}

    # Validate provided parameters
    for raw_key, raw_val in params.items():
        # Pass through runtime dependencies and context objects untouched
        if raw_key in RUNTIME_CONTEXT_KEYS:
            validated_params[raw_key] = raw_val
            continue

        # Resolve parameter spec
        if raw_key in spec_by_name:
            spec = spec_by_name[raw_key]
            canonical_name = raw_key
        elif raw_key in spec_by_alias:
            spec = spec_by_alias[raw_key]
            canonical_name = spec.name
        else:
            allowed_names = sorted(list(spec_by_name.keys()))
            raise AttackConfigurationError(
                f"Unknown parameter '{raw_key}' for attack '{canonical_attack}' in domain '{norm_domain}'. "
                f"Allowed parameters: {allowed_names}"
            )

        # Handle None value
        if raw_val is None:
            if spec.required:
                raise AttackConfigurationError(
                    f"Parameter '{canonical_name}' is required and cannot be None."
                )
            validated_params[canonical_name] = None
            continue

        # Type and Value Validation
        if spec.data_type == "int":
            if canonical_name == "target_label":
                is_valid_target = (
                    (isinstance(raw_val, int) and not isinstance(raw_val, bool))
                    or hasattr(raw_val, "shape")
                    or isinstance(raw_val, (list, tuple))
                )
                if not is_valid_target:
                    raise AttackConfigurationError(
                        f"Parameter '{canonical_name}' must be an integer class index or target tensor/list, got {type(raw_val).__name__} ({raw_val})."
                    )
                if isinstance(raw_val, int) and raw_val < 0:
                    raise AttackConfigurationError(
                        f"Parameter '{canonical_name}' must be at least 0, got {raw_val}."
                    )
            else:
                if not isinstance(raw_val, int) or isinstance(raw_val, bool):
                    raise AttackConfigurationError(
                        f"Parameter '{canonical_name}' must be an integer, got {type(raw_val).__name__} ({raw_val})."
                    )
                if spec.min_value is not None:
                    if spec.exclusive_min and raw_val <= spec.min_value:
                        raise AttackConfigurationError(
                            f"Parameter '{canonical_name}' must be strictly greater than {spec.min_value}, got {raw_val}."
                        )
                    elif not spec.exclusive_min and raw_val < spec.min_value:
                        raise AttackConfigurationError(
                            f"Parameter '{canonical_name}' must be at least {spec.min_value}, got {raw_val}."
                        )
                if spec.max_value is not None:
                    if spec.exclusive_max and raw_val >= spec.max_value:
                        raise AttackConfigurationError(
                            f"Parameter '{canonical_name}' must be strictly less than {spec.max_value}, got {raw_val}."
                        )
                    elif not spec.exclusive_max and raw_val > spec.max_value:
                        raise AttackConfigurationError(
                            f"Parameter '{canonical_name}' must be at most {spec.max_value}, got {raw_val}."
                        )
            validated_params[canonical_name] = raw_val

        elif spec.data_type == "float":
            if not isinstance(raw_val, (int, float)) or isinstance(raw_val, bool):
                raise AttackConfigurationError(
                    f"Parameter '{canonical_name}' must be a float or integer, got {type(raw_val).__name__} ({raw_val})."
                )
            val_float = float(raw_val)
            if spec.min_value is not None:
                if spec.exclusive_min and val_float <= spec.min_value:
                    raise AttackConfigurationError(
                        f"Parameter '{canonical_name}' must be strictly greater than {spec.min_value}, got {raw_val}."
                    )
                elif not spec.exclusive_min and val_float < spec.min_value:
                    raise AttackConfigurationError(
                        f"Parameter '{canonical_name}' must be at least {spec.min_value}, got {raw_val}."
                    )
            if spec.max_value is not None:
                if spec.exclusive_max and val_float >= spec.max_value:
                    raise AttackConfigurationError(
                        f"Parameter '{canonical_name}' must be strictly less than {spec.max_value}, got {raw_val}."
                    )
                elif not spec.exclusive_max and val_float > spec.max_value:
                    raise AttackConfigurationError(
                        f"Parameter '{canonical_name}' must be at most {spec.max_value}, got {raw_val}."
                    )
            # Special constraint for time-series JSMA theta: cannot be 0
            if canonical_attack == "jsma" and canonical_name == "theta" and val_float == 0.0:
                raise AttackConfigurationError("JSMA theta (step size) cannot be 0.")

            validated_params[canonical_name] = val_float

        elif spec.data_type == "bool":
            if not isinstance(raw_val, bool):
                raise AttackConfigurationError(
                    f"Parameter '{canonical_name}' must be a boolean, got {type(raw_val).__name__} ({raw_val})."
                )
            validated_params[canonical_name] = raw_val

        elif spec.data_type == "str":
            if not isinstance(raw_val, str):
                raise AttackConfigurationError(
                    f"Parameter '{canonical_name}' must be a string, got {type(raw_val).__name__} ({raw_val})."
                )
            str_val = raw_val.strip()
            if spec.allowed_values is not None:
                # Normalization for norm strings
                if canonical_name == "norm":
                    upper_norm = str_val.upper()
                    if upper_norm in ("LINF", "INF"):
                        str_val = "Linf"
                    elif upper_norm in ("L2", "2"):
                        str_val = "L2"
                    else:
                        raise AttackConfigurationError(
                            f"Unsupported norm '{raw_val}'. Supported norms: {spec.allowed_values}"
                        )
                else:
                    allowed_lower = [str(av).lower() for av in spec.allowed_values]
                    if str_val.lower() not in allowed_lower:
                        raise AttackConfigurationError(
                            f"Unsupported value '{raw_val}' for parameter '{canonical_name}'. "
                            f"Allowed values: {spec.allowed_values}"
                        )
                    # Use canonical case from allowed_values
                    for av in spec.allowed_values:
                        if str(av).lower() == str_val.lower():
                            str_val = str(av)
                            break
            validated_params[canonical_name] = str_val

        elif spec.data_type == "list":
            if not isinstance(raw_val, (list, tuple)):
                raise AttackConfigurationError(
                    f"Parameter '{canonical_name}' must be a list or tuple, got {type(raw_val).__name__} ({raw_val})."
                )
            list_val = list(raw_val)
            if len(list_val) == 0 and spec.name == "bug_types":
                raise AttackConfigurationError(
                    f"Parameter '{canonical_name}' cannot be empty."
                )
            if spec.allowed_values is not None:
                allowed_lower = [str(av).lower() for av in spec.allowed_values]
                for item in list_val:
                    if not isinstance(item, str) or item.lower() not in allowed_lower:
                        raise AttackConfigurationError(
                            f"Invalid item '{item}' in parameter '{canonical_name}'. "
                            f"Allowed values: {spec.allowed_values}"
                        )
            validated_params[canonical_name] = list_val

    # Check clip constraint consistency if both clip_min and clip_max are provided
    if "clip_min" in validated_params and "clip_max" in validated_params:
        c_min = validated_params["clip_min"]
        c_max = validated_params["clip_max"]
        if c_min is not None and c_max is not None and c_min > c_max:
            raise AttackConfigurationError(
                f"clip_min ({c_min}) cannot be greater than clip_max ({c_max})"
            )

    # Check conditional visibility consistency: target_label is only valid when targeted=True
    if validated_params.get("target_label") is not None:
        targeted_flag = validated_params.get("targeted", False)
        if not targeted_flag:
            raise AttackConfigurationError(
                "Parameter 'target_label' was specified, but 'targeted' is False. "
                "Set 'targeted=True' to execute a targeted attack."
            )

    # If fill_defaults is True, populate omitted parameters with their default values
    if fill_defaults:
        for spec in specs:
            if spec.name not in validated_params:
                validated_params[spec.name] = spec.default

    return validated_params


def prepare_attack_config(
    domain: str,
    attack_name: str,
    config: Optional[Any] = None,
) -> Any:
    """
    Validate and normalize an AttackConfig for a specific attack using the parameter registry.

    1. Resolves canonical domain and attack name. If attack is uncataloged, leaves config as-is.
    2. Collects attack-specific parameters from config.params and relevant direct fields.
    3. Validates hyperparameters using validate_attack_params():
       - Enforces data types, range bounds, allowed values, and conditional visibility.
       - Resolves aliases to canonical parameter names.
       - Rejects unsupported parameters with clear error messages.
       - Preserves runtime context (tokenizer, VAE, loss function, bounds, device) untouched.
    4. Updates config.params with the validated and normalized dictionary.

    Args:
        domain: Input domain identifier ('image', 'tabular', 'time_series', 'text').
        attack_name: Attack name or alias.
        config: Optional AttackConfig instance. If None, a new default AttackConfig is created.

    Returns:
        The normalized and validated AttackConfig instance.

    Raises:
        AttackConfigurationError: If any parameter value or constraint is invalid.
    """
    if config is None:
        from app.attack_engine.config import AttackConfig
        config = AttackConfig()

    try:
        norm_domain = normalize_domain(domain)
        canonical_attack = normalize_attack_name(norm_domain, attack_name)
        specs = get_attack_param_specs(norm_domain, canonical_attack)
        spec_names = {s.name for s in specs}
    except AttackConfigurationError:
        # Non-registered / custom attack: preserve backward compatibility
        return config

    raw_params = dict(config.params) if getattr(config, "params", None) else {}

    # Incorporate direct config attributes if supported by this attack and not already in raw_params
    if "epsilon" in spec_names and "epsilon" not in raw_params:
        if getattr(config, "epsilon", None) is not None:
            raw_params["epsilon"] = config.epsilon

    if "clip_min" in spec_names and "clip_min" not in raw_params:
        if norm_domain == "image":
            if getattr(config, "clip_min", None) is not None:
                raw_params["clip_min"] = config.clip_min
        elif norm_domain == "time_series":
            if getattr(config, "clip_min", None) not in (None, 0.0):
                raw_params["clip_min"] = config.clip_min

    if "clip_max" in spec_names and "clip_max" not in raw_params:
        if norm_domain == "image":
            if getattr(config, "clip_max", None) is not None:
                raw_params["clip_max"] = config.clip_max
        elif norm_domain == "time_series":
            if getattr(config, "clip_max", None) not in (None, 1.0):
                raw_params["clip_max"] = config.clip_max

    if getattr(config, "loss_fn", None) is not None and "loss_fn" not in raw_params:
        raw_params["loss_fn"] = config.loss_fn

    if getattr(config, "targeted", None) is not None and "targeted" not in raw_params:
        raw_params["targeted"] = config.targeted

    if getattr(config, "target_label", None) is not None and "target_label" not in raw_params:
        raw_params["target_label"] = config.target_label

    validated_params = validate_attack_params(
        domain=norm_domain,
        attack_name=canonical_attack,
        params=raw_params,
        fill_defaults=False,
    )

    config.params = validated_params
    if "epsilon" in validated_params:
        config.epsilon = validated_params["epsilon"]
    if "clip_min" in validated_params:
        config.clip_min = validated_params["clip_min"]
    if "clip_max" in validated_params:
        config.clip_max = validated_params["clip_max"]
    if "loss_fn" in validated_params:
        config.loss_fn = validated_params["loss_fn"]

    return config


__all__ = [
    "ParameterSpec",
    "SUPPORTED_DOMAINS",
    "RUNTIME_CONTEXT_KEYS",
    "normalize_domain",
    "normalize_attack_name",
    "get_supported_attacks",
    "get_attack_param_specs",
    "get_attack_param_specs_dict",
    "validate_attack_params",
    "prepare_attack_config",
]

