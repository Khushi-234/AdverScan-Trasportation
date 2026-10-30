"""
Tabular feature metadata representation and constraint projection utilities.

Provides structured parsing and batched projection for:
- Continuous features (with arbitrary [min, max] bounds)
- Integer features (discrete rounded values within bounds)
- Binary features (discrete {0, 1} values)
- Categorical features (nearest valid category projection)
- Immutable features (strictly preserved from original inputs)
- Feature-specific perturbation limits and weights
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import torch

from app.attack_engine.exceptions import (
    AttackConfigurationError,
    AttackExecutionError,
)
from app.ingestion.adapters.base_adapter import BaseModelAdapter


def get_raw_model(model: Any) -> Any:
    """Extract raw framework model if model is wrapped in a BaseModelAdapter."""
    if isinstance(model, BaseModelAdapter):
        return model.get_model()
    return model


@dataclass
class TabularFeature:
    """
    Specification for a single tabular feature.

    Attributes:
        name: Name or string identifier of the feature.
        index: Column index in the input tensor.
        type: Feature type ('continuous', 'integer', 'binary', 'categorical').
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).
        categories: Allowed discrete categories if categorical.
        mutable: Whether this feature can be modified by adversarial perturbations.
        weight: Sensitivity / cost weight associated with perturbing this feature.
        perturbation_limit: Maximum allowed absolute perturbation |x_adv - x_orig|.
    """

    name: str
    index: int
    type: str = "continuous"
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    categories: Optional[List[float]] = None
    mutable: bool = True
    weight: float = 1.0
    perturbation_limit: Optional[float] = None

    def __post_init__(self):
        self.type = self.type.lower()
        if self.type not in ("continuous", "integer", "binary", "categorical"):
            raise AttackConfigurationError(
                f"Unknown feature type '{self.type}' for feature '{self.name}'. "
                "Supported types: ['continuous', 'integer', 'binary', 'categorical']"
            )

        if self.type == "binary":
            if self.min_val is None:
                self.min_val = 0.0
            if self.max_val is None:
                self.max_val = 1.0
            if self.categories is None:
                self.categories = [0.0, 1.0]

        if self.min_val is not None and self.max_val is not None:
            if self.min_val > self.max_val:
                raise AttackConfigurationError(
                    f"Feature '{self.name}' has min_val ({self.min_val}) > max_val ({self.max_val})"
                )

        if self.type == "categorical":
            if self.categories is not None:
                self.categories = [float(c) for c in self.categories]
                if len(self.categories) == 0:
                    raise AttackConfigurationError(
                        f"Categorical feature '{self.name}' must have at least one allowed category."
                    )

        if self.weight <= 0:
            raise AttackConfigurationError(
                f"Feature weight for '{self.name}' must be positive, got {self.weight}"
            )

        if self.perturbation_limit is not None and self.perturbation_limit < 0:
            raise AttackConfigurationError(
                f"perturbation_limit for '{self.name}' cannot be negative, got {self.perturbation_limit}"
            )


class TabularMetadata:
    """
    Container managing tabular feature specifications and batched constraint projections.
    """

    def __init__(self, features: List[TabularFeature]):
        self.features = features
        self.num_features = len(features)
        self.name_to_idx = {f.name: f.index for f in features}

    def __len__(self) -> int:
        return self.num_features

    def __getitem__(self, idx: int) -> TabularFeature:
        return self.features[idx]

    def get_mutable_mask(self, device: torch.device) -> torch.Tensor:
        """Boolean tensor of shape (num_features,) where True = mutable."""
        mask = [f.mutable for f in self.features]
        return torch.tensor(mask, dtype=torch.bool, device=device)

    def get_weights_tensor(self, device: torch.device) -> torch.Tensor:
        """Tensor of shape (num_features,) containing feature perturbation weights."""
        weights = [float(f.weight) for f in self.features]
        return torch.tensor(weights, dtype=torch.float32, device=device)

    def get_bounds_tensors(
        self,
        device: torch.device,
        global_clip_min: Optional[float] = None,
        global_clip_max: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (min_bounds, max_bounds) tensors of shape (num_features,).
        """
        mins = []
        maxs = []
        for f in self.features:
            f_min = (
                f.min_val
                if f.min_val is not None
                else (global_clip_min if global_clip_min is not None else float("-inf"))
            )
            f_max = (
                f.max_val
                if f.max_val is not None
                else (global_clip_max if global_clip_max is not None else float("inf"))
            )
            mins.append(f_min)
            maxs.append(f_max)

        return (
            torch.tensor(mins, dtype=torch.float32, device=device),
            torch.tensor(maxs, dtype=torch.float32, device=device),
        )

    def project(
        self,
        x_adv: torch.Tensor,
        x_orig: torch.Tensor,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Project adversarial tensor onto the valid tabular constraint manifold.

        Enforces:
            1. Immutable feature restoration (x_adv[:, j] = x_orig[:, j])
            2. Continuous bounds [min_val, max_val] (or fallback to clip_min/clip_max)
            3. Feature-specific perturbation limits |x_adv - x_orig| <= limit
            4. Integer rounding for integer features
            5. Binary discretization {0, 1}
            6. Discrete category projection for categorical features

        Args:
            x_adv: Adversarial batch tensor of shape (batch_size, num_features).
            x_orig: Original clean batch tensor of shape (batch_size, num_features).
            clip_min: Optional global minimum bound (applied if feature has no min_val).
            clip_max: Optional global maximum bound (applied if feature has no max_val).

        Returns:
            Projected adversarial tensor satisfying all tabular constraints.
        """
        device = x_adv.device
        x_proj = x_adv.clone()

        for j, feat in enumerate(self.features):
            # 1. Immutable features: strictly restore to original
            if not feat.mutable:
                x_proj[:, j] = x_orig[:, j]
                continue

            col = x_proj[:, j]
            orig_col = x_orig[:, j]

            # 2. Feature-specific perturbation limit
            if feat.perturbation_limit is not None:
                limit = feat.perturbation_limit
                col = torch.clamp(col, min=orig_col - limit, max=orig_col + limit)

            # 3. Continuous bounds (feature min_val/max_val take precedence)
            f_min = feat.min_val if feat.min_val is not None else clip_min
            f_max = feat.max_val if feat.max_val is not None else clip_max

            if f_min is not None or f_max is not None:
                col = torch.clamp(
                    col,
                    min=f_min if f_min is not None else float("-inf"),
                    max=f_max if f_max is not None else float("inf"),
                )

            # 4. Integer features: round to nearest integer and re-clamp
            if feat.type == "integer":
                col = torch.round(col)
                if f_min is not None or f_max is not None:
                    col = torch.clamp(
                        col,
                        min=f_min if f_min is not None else float("-inf"),
                        max=f_max if f_max is not None else float("inf"),
                    )

            # 5. Binary features: round to {0.0, 1.0}
            elif feat.type == "binary":
                col = torch.clamp(torch.round(col), min=0.0, max=1.0)

            # 6. Categorical features: project to nearest allowed category
            elif feat.type == "categorical":
                if feat.categories is not None and len(feat.categories) > 0:
                    cats_tensor = torch.tensor(
                        feat.categories, dtype=col.dtype, device=device
                    )
                    # col shape (B, 1), cats_tensor shape (1, num_cats)
                    diffs = (col.unsqueeze(-1) - cats_tensor.unsqueeze(0)).abs()
                    nearest_idx = torch.argmin(diffs, dim=-1)
                    col = cats_tensor[nearest_idx]

            x_proj[:, j] = col

        return x_proj


def parse_tabular_metadata(
    model: Any,
    config_params: Optional[Dict[str, Any]] = None,
    num_features: int = 0,
) -> TabularMetadata:
    """
    Parse tabular feature metadata from attack config parameters or model attributes.

    Supports:
        - Dict of {feature_name: {type, min, max, mutable, weight, categories}}
        - List of feature spec dicts
        - Explicit config parameters: immutable_features, categorical_features, etc.

    Args:
        model: Target model or adapter.
        config_params: Optional dictionary of attack parameters.
        num_features: Expected number of input features.

    Returns:
        TabularMetadata instance.
    """
    params = config_params or {}

    raw_meta = params.get("feature_metadata", None)
    if raw_meta is None:
        raw_meta = params.get("tabular_metadata", None)
    if raw_meta is None:
        raw_meta = getattr(model, "feature_metadata", None)
    if raw_meta is None:
        raw_model = get_raw_model(model)
        raw_meta = getattr(raw_model, "feature_metadata", None)

    # Explicit individual config parameters for ease of use
    immutable_indices: Set[int] = set()
    for imm in params.get("immutable_features", []):
        if isinstance(imm, int):
            immutable_indices.add(imm)

    categorical_specs: Dict[int, List[float]] = {}
    cat_param = params.get("categorical_features", [])
    if isinstance(cat_param, dict):
        for k, v in cat_param.items():
            idx = int(k)
            categorical_specs[idx] = (
                [float(x) for x in v] if isinstance(v, (list, tuple)) else None
            )
    elif isinstance(cat_param, (list, tuple)):
        for c in cat_param:
            if isinstance(c, int):
                categorical_specs[c] = None

    integer_indices: Set[int] = set(params.get("integer_features", []))
    binary_indices: Set[int] = set(params.get("binary_features", []))
    feature_bounds: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
    if "feature_bounds" in params and isinstance(params["feature_bounds"], dict):
        for k, v in params["feature_bounds"].items():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                feature_bounds[int(k)] = (
                    float(v[0]) if v[0] is not None else None,
                    float(v[1]) if v[1] is not None else None,
                )

    feature_weights: Dict[int, float] = {}
    if "feature_weights" in params:
        fw = params["feature_weights"]
        if isinstance(fw, dict):
            for k, v in fw.items():
                feature_weights[int(k)] = float(v)
        elif isinstance(fw, (list, tuple, torch.Tensor)):
            for idx, v in enumerate(fw):
                feature_weights[idx] = float(v)

    feature_limits: Dict[int, float] = {}
    fl_param = params.get(
        "feature_limits",
        params.get(
            "feature_perturbation_limits", params.get("perturbation_limits", None)
        ),
    )
    if fl_param is not None:
        if isinstance(fl_param, dict):
            for k, v in fl_param.items():
                if v is not None:
                    feature_limits[int(k)] = float(v)
        elif isinstance(fl_param, (list, tuple, torch.Tensor)):
            for idx, v in enumerate(fl_param):
                if v is not None:
                    feature_limits[idx] = float(v)

    features: List[TabularFeature] = []

    # Case A: Detailed feature metadata dictionary or list provided
    if isinstance(raw_meta, dict):
        # Could be {feature_name: dict} or {index: dict}
        items = list(raw_meta.items())
        total = max(len(items), num_features)
        for idx in range(total):
            if idx < len(items):
                key, val = items[idx]
                feat_name = str(key)
                if not isinstance(val, dict):
                    val = {}
            else:
                feat_name = f"f_{idx}"
                val = {}

            f_type = val.get("type", "continuous").lower()
            min_v = val.get("min", val.get("min_val", None))
            max_v = val.get("max", val.get("max_val", None))
            mutable = val.get("mutable", True)
            weight = float(val.get("weight", 1.0))
            cats = val.get("categories", None)
            limit = val.get("perturbation_limit", None)

            # Apply overrides from explicit config params if present
            if idx in immutable_indices or feat_name in params.get(
                "immutable_features", []
            ):
                mutable = False
            if idx in integer_indices:
                f_type = "integer"
            if idx in binary_indices:
                f_type = "binary"
            if idx in categorical_specs:
                f_type = "categorical"
                if categorical_specs[idx] is not None:
                    cats = categorical_specs[idx]
            if idx in feature_bounds:
                min_v, max_v = feature_bounds[idx]
            if idx in feature_weights:
                weight = feature_weights[idx]
            if idx in feature_limits:
                limit = feature_limits[idx]

            features.append(
                TabularFeature(
                    name=feat_name,
                    index=idx,
                    type=f_type,
                    min_val=float(min_v) if min_v is not None else None,
                    max_val=float(max_v) if max_v is not None else None,
                    categories=cats,
                    mutable=mutable,
                    weight=weight,
                    perturbation_limit=float(limit) if limit is not None else None,
                )
            )

    elif isinstance(raw_meta, (list, tuple)):
        total = max(len(raw_meta), num_features)
        for idx in range(total):
            val = (
                raw_meta[idx]
                if idx < len(raw_meta) and isinstance(raw_meta[idx], dict)
                else {}
            )
            feat_name = val.get("name", f"f_{idx}")
            f_type = val.get("type", "continuous").lower()
            min_v = val.get("min", val.get("min_val", None))
            max_v = val.get("max", val.get("max_val", None))
            mutable = val.get("mutable", True)
            weight = float(val.get("weight", 1.0))
            cats = val.get("categories", None)
            limit = val.get("perturbation_limit", None)

            if idx in immutable_indices:
                mutable = False
            if idx in integer_indices:
                f_type = "integer"
            if idx in binary_indices:
                f_type = "binary"
            if idx in categorical_specs:
                f_type = "categorical"
                if categorical_specs[idx] is not None:
                    cats = categorical_specs[idx]
            if idx in feature_bounds:
                min_v, max_v = feature_bounds[idx]
            if idx in feature_weights:
                weight = feature_weights[idx]
            if idx in feature_limits:
                limit = feature_limits[idx]

            features.append(
                TabularFeature(
                    name=feat_name,
                    index=idx,
                    type=f_type,
                    min_val=float(min_v) if min_v is not None else None,
                    max_val=float(max_v) if max_v is not None else None,
                    categories=cats,
                    mutable=mutable,
                    weight=weight,
                    perturbation_limit=float(limit) if limit is not None else None,
                )
            )

    else:
        # Case B: Default metadata synthesized from num_features and explicit config params
        for idx in range(num_features):
            feat_name = f"f_{idx}"
            f_type = "continuous"
            min_v = None
            max_v = None
            mutable = True
            weight = 1.0
            cats = None
            limit = None

            if idx in immutable_indices or feat_name in params.get(
                "immutable_features", []
            ):
                mutable = False
            if idx in integer_indices:
                f_type = "integer"
            if idx in binary_indices:
                f_type = "binary"
            if idx in categorical_specs:
                f_type = "categorical"
                cats = categorical_specs[idx]
            if idx in feature_bounds:
                min_v, max_v = feature_bounds[idx]
            if idx in feature_weights:
                weight = feature_weights[idx]
            if idx in feature_limits:
                limit = feature_limits[idx]

            features.append(
                TabularFeature(
                    name=feat_name,
                    index=idx,
                    type=f_type,
                    min_val=min_v,
                    max_val=max_v,
                    categories=cats,
                    mutable=mutable,
                    weight=weight,
                    perturbation_limit=limit,
                )
            )

    return TabularMetadata(features)


def validate_tabular_inputs(
    inputs: Any,
    labels: Any,
    attack_name: str = "TabularAttack",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Validate that inputs and labels are valid 2D tabular tensors with matching batch size.

    Returns:
        (inputs_tensor, labels_tensor) on the appropriate float32 / long types.

    Raises:
        AttackExecutionError: If inputs/labels are empty, invalid, or mismatched.
    """
    if inputs is None:
        raise AttackExecutionError(f"{attack_name}: Inputs tensor cannot be None.")

    if not isinstance(inputs, torch.Tensor):
        try:
            inputs = torch.as_tensor(inputs, dtype=torch.float32)
        except Exception as e:
            raise AttackExecutionError(
                f"{attack_name} expects inputs convertible to torch.Tensor, got {type(inputs)}"
            ) from e

    if inputs.numel() == 0:
        raise AttackExecutionError(f"{attack_name}: Inputs tensor is empty.")

    if inputs.dim() == 1:
        inputs = inputs.unsqueeze(0)
    elif inputs.dim() != 2:
        raise AttackExecutionError(
            f"{attack_name} expects 2D tabular input tensor of shape (batch_size, num_features), "
            f"got shape {tuple(inputs.shape)}"
        )

    if labels is None:
        raise AttackExecutionError(f"{attack_name}: Labels cannot be None.")

    if not isinstance(labels, torch.Tensor):
        try:
            labels = torch.as_tensor(labels, dtype=torch.long)
        except Exception as e:
            raise AttackExecutionError(
                f"{attack_name} expects labels convertible to torch.Tensor, got {type(labels)}"
            ) from e

    if labels.dim() == 0:
        labels = labels.repeat(inputs.size(0))
    elif labels.dim() == 1:
        if labels.size(0) != inputs.size(0):
            raise AttackExecutionError(
                f"{attack_name}: Inputs batch size ({inputs.size(0)}) does not match "
                f"labels size ({labels.size(0)})."
            )
    elif labels.dim() == 2 and labels.size(1) == 1:
        labels = labels.squeeze(1)
        if labels.size(0) != inputs.size(0):
            raise AttackExecutionError(f"{attack_name}: Mismatched batch sizes.")
    else:
        raise AttackExecutionError(
            f"{attack_name} expects 1D labels tensor, got shape {tuple(labels.shape)}"
        )

    return inputs.float(), labels.long()
