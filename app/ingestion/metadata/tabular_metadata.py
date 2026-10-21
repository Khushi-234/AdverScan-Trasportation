"""
Tabular metadata container and feature inference for AdverScan.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch


@dataclass
class TabularMetadata:
    """
    Metadata container preserving tabular-specific schema, feature categorization,
    target information, class mapping, and preprocessing details.
    """

    feature_names: List[str] = field(default_factory=list)
    feature_types: Dict[str, str] = field(default_factory=dict)
    numerical_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    binary_features: List[str] = field(default_factory=list)
    ordinal_features: List[str] = field(default_factory=list)
    target_column: Optional[str] = None
    num_features: int = 0
    num_classes: Optional[int] = None
    class_mapping: Optional[Dict[Any, int]] = None
    target_distribution: Optional[Dict[Any, int]] = None
    feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    categorical_cardinalities: Optional[Dict[str, int]] = None
    preprocessing_info: Optional[Dict[str, Any]] = None
    target_available: bool = True
    target_status: Optional[str] = None

    # Attack-Aware Constraints and Attack Space
    feature_constraints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    immutable_features: List[str] = field(default_factory=list)
    mutable_features: List[str] = field(default_factory=list)
    representation_space: str = "raw"  # 'raw', 'scaled', 'encoded', 'scaled_and_encoded'
    categorical_encoding_type: str = "none"  # 'none', 'one-hot', 'ordinal', 'label', 'embedding', 'custom'
    encoding_info: Dict[str, Any] = field(default_factory=dict)
    attack_space: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.num_features and self.feature_names:
            self.num_features = len(self.feature_names)
        elif self.num_features and not self.feature_names:
            self.feature_names = [f"feature_{i}" for i in range(self.num_features)]

        # Synchronize categorized lists with feature_types dictionary if provided
        if self.feature_types:
            for feat, ftype in self.feature_types.items():
                clean_type = str(ftype).lower()
                if clean_type == "numerical" and feat not in self.numerical_features:
                    self.numerical_features.append(feat)
                elif clean_type == "categorical" and feat not in self.categorical_features:
                    self.categorical_features.append(feat)
                elif clean_type == "binary" and feat not in self.binary_features:
                    self.binary_features.append(feat)
                elif clean_type == "ordinal" and feat not in self.ordinal_features:
                    self.ordinal_features.append(feat)
        else:
            # Populate feature_types from individual lists
            for f in self.numerical_features:
                self.feature_types[f] = "numerical"
            for f in self.categorical_features:
                self.feature_types[f] = "categorical"
            for f in self.binary_features:
                self.feature_types[f] = "binary"
            for f in self.ordinal_features:
                self.feature_types[f] = "ordinal"

        # Build feature_constraints and attack_space for each feature
        for feat in self.feature_names:
            ftype = self.feature_types.get(feat, "numerical")
            existing_c = self.feature_constraints.get(feat, {})

            # Determine mutability
            if feat in self.immutable_features:
                is_mut = False
            elif feat in self.mutable_features:
                is_mut = True
            elif "mutable" in existing_c:
                is_mut = bool(existing_c["mutable"])
            else:
                is_mut = True

            c: Dict[str, Any] = {
                "type": ftype,
                "mutable": is_mut,
            }

            if ftype == "numerical":
                c["is_continuous"] = True
                # Use explicit bounds if provided, otherwise observed ranges if known, else None
                min_val = existing_c.get("min")
                max_val = existing_c.get("max")
                if min_val is None and self.feature_ranges and feat in self.feature_ranges:
                    min_val = self.feature_ranges[feat][0]
                if max_val is None and self.feature_ranges and feat in self.feature_ranges:
                    max_val = self.feature_ranges[feat][1]
                c["min"] = min_val
                c["max"] = max_val
                if "step" in existing_c:
                    c["step"] = existing_c["step"]

            elif ftype == "binary":
                c["is_continuous"] = False
                c["allowed_values"] = existing_c.get("allowed_values", [0, 1])

            elif ftype == "categorical":
                c["is_continuous"] = False
                c["allowed_values"] = existing_c.get("allowed_values")

            elif ftype == "ordinal":
                c["is_continuous"] = False
                c["ordering"] = existing_c.get("ordering")
                if "allowed_values" in existing_c:
                    c["allowed_values"] = existing_c["allowed_values"]
                elif c["ordering"]:
                    c["allowed_values"] = list(c["ordering"])

            # Merge any other custom attributes from existing constraint
            for k, v in existing_c.items():
                if k not in c:
                    c[k] = v

            self.feature_constraints[feat] = c

            # Build attack space guidelines
            self.attack_space[feat] = {
                "type": ftype,
                "mutable": is_mut,
                "is_continuous": c.get("is_continuous", False),
                "perturbation_mode": (
                    "continuous" if ftype == "numerical"
                    else "flip" if ftype == "binary"
                    else "category_swap" if ftype == "categorical"
                    else "ordinal_shift"
                ),
                "valid_domain": (
                    {"min": c.get("min"), "max": c.get("max")} if ftype == "numerical"
                    else c.get("allowed_values") if ftype in ("binary", "categorical")
                    else c.get("ordering")
                ),
            }

        # Synchronize immutable and mutable lists
        self.immutable_features = [f for f in self.feature_names if not self.feature_constraints[f].get("mutable", True)]
        self.mutable_features = [f for f in self.feature_names if self.feature_constraints[f].get("mutable", True)]

    @property
    def index_to_class(self) -> Optional[Dict[int, Any]]:
        """Return the inverted class mapping (model output index -> original class label)."""
        if self.class_mapping is None:
            return None
        return {v: k for k, v in self.class_mapping.items()}

    def get_feature_constraint(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """Return constraint dictionary for a given feature."""
        return self.feature_constraints.get(feature_name)

    def is_mutable(self, feature_name: str) -> bool:
        """Check whether a feature is allowed to be perturbed."""
        c = self.get_feature_constraint(feature_name)
        if c is not None and "mutable" in c:
            return bool(c["mutable"])
        return feature_name not in self.immutable_features

    def get_mutable_features(self) -> List[str]:
        """Return all features that can be perturbed."""
        return list(self.mutable_features)

    def get_immutable_features(self) -> List[str]:
        """Return all features that must remain fixed."""
        return list(self.immutable_features)

    def validate_constraints(
        self,
        sample: Any,
        reference_sample: Optional[Any] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Validate whether a given feature sample satisfies all defined constraints,
        including bounds, allowed values, and immutability relative to reference_sample.

        Args:
            sample: dict, pd.Series, or 1D array/list of values.
            reference_sample: Optional original unperturbed sample to verify immutability.

        Returns:
            Tuple (is_valid: bool, violations: List[str])
        """
        import pandas as pd
        violations: List[str] = []

        if isinstance(sample, (pd.Series, dict)):
            sample_dict = dict(sample)
        elif hasattr(sample, "columns") and len(sample) == 1:
            sample_dict = sample.iloc[0].to_dict()
        else:
            sample_dict = {
                name: sample[i]
                for i, name in enumerate(self.feature_names)
                if i < len(sample)
            }

        ref_dict = None
        if reference_sample is not None:
            if isinstance(reference_sample, (pd.Series, dict)):
                ref_dict = dict(reference_sample)
            elif hasattr(reference_sample, "columns") and len(reference_sample) == 1:
                ref_dict = reference_sample.iloc[0].to_dict()
            else:
                ref_dict = {
                    name: reference_sample[i]
                    for i, name in enumerate(self.feature_names)
                    if i < len(reference_sample)
                }

        for feat, val in sample_dict.items():
            if feat not in self.feature_constraints:
                continue
            c = self.feature_constraints[feat]

            # Immutability check
            if not c.get("mutable", True) and ref_dict is not None and feat in ref_dict:
                if val != ref_dict[feat]:
                    violations.append(
                        f"Immutable feature '{feat}' was modified: original={ref_dict[feat]}, new={val}"
                    )

            ftype = c.get("type", "numerical")
            if ftype == "numerical":
                min_b = c.get("min")
                max_b = c.get("max")
                try:
                    num_val = float(val)
                    if min_b is not None and num_val < float(min_b):
                        violations.append(f"Feature '{feat}' value {num_val} is below minimum bound {min_b}")
                    if max_b is not None and num_val > float(max_b):
                        violations.append(f"Feature '{feat}' value {num_val} is above maximum bound {max_b}")
                except (ValueError, TypeError):
                    violations.append(f"Numerical feature '{feat}' has non-numeric value: {val}")

            elif ftype in ("binary", "categorical"):
                allowed = c.get("allowed_values")
                if allowed is not None and val not in allowed:
                    violations.append(f"Feature '{feat}' value {val} not in allowed values: {allowed}")

            elif ftype == "ordinal":
                ordering = c.get("ordering") or c.get("allowed_values")
                if ordering is not None and val not in ordering:
                    violations.append(f"Ordinal feature '{feat}' value {val} not in defined ordering: {ordering}")

        return (len(violations) == 0, violations)

    def project_to_constraints(
        self,
        sample: Any,
        reference_sample: Optional[Any] = None,
    ) -> Any:
        """
        Project a perturbed sample into the valid constrained attack space:
        - Freezes immutable features to their reference values
        - Clips numerical features to [min, max]
        - Snaps binary features to valid values (e.g. 0 or 1, avoiding floats like 0.37)
        - Snaps categorical/ordinal to nearest allowed category

        Args:
            sample: pd.DataFrame, pd.Series, np.ndarray, or dict.
            reference_sample: Original unperturbed sample.

        Returns:
            Projected sample matching the input type.
        """
        import pandas as pd
        is_df = isinstance(sample, pd.DataFrame)
        is_series = isinstance(sample, pd.Series)
        is_array = isinstance(sample, np.ndarray)

        if is_df:
            df = sample.copy()
            ref_df = reference_sample if isinstance(reference_sample, pd.DataFrame) else None
            for feat in self.feature_names:
                if feat not in df.columns:
                    continue
                c = self.feature_constraints.get(feat, {})
                if not c.get("mutable", True) and ref_df is not None and feat in ref_df.columns:
                    df[feat] = ref_df[feat].values
                    continue
                ftype = c.get("type", "numerical")
                if ftype == "numerical":
                    min_bnd = c.get("min")
                    max_bnd = c.get("max")
                    if min_bnd is not None or max_bnd is not None:
                        df[feat] = df[feat].clip(lower=min_bnd, upper=max_bnd)
                elif ftype == "binary":
                    # Snap to nearest binary value (e.g. 0 or 1)
                    allowed = c.get("allowed_values", [0, 1])
                    if len(allowed) == 2 and all(isinstance(x, (int, float)) for x in allowed):
                        mid = (allowed[0] + allowed[1]) / 2.0
                        df[feat] = np.where(df[feat] >= mid, allowed[1], allowed[0])
            return df

        elif is_series or isinstance(sample, dict):
            d = dict(sample)
            ref_d = dict(reference_sample) if reference_sample is not None else {}
            for feat in self.feature_names:
                if feat not in d:
                    continue
                c = self.feature_constraints.get(feat, {})
                if not c.get("mutable", True) and feat in ref_d:
                    d[feat] = ref_d[feat]
                    continue
                ftype = c.get("type", "numerical")
                if ftype == "numerical":
                    try:
                        val = float(d[feat])
                        min_bnd = c.get("min")
                        max_bnd = c.get("max")
                        if min_bnd is not None:
                            val = max(val, float(min_bnd))
                        if max_bnd is not None:
                            val = min(val, float(max_bnd))
                        d[feat] = val
                    except Exception:
                        pass
                elif ftype == "binary":
                    allowed = c.get("allowed_values", [0, 1])
                    if len(allowed) == 2 and all(isinstance(x, (int, float)) for x in allowed):
                        try:
                            val = float(d[feat])
                            mid = (allowed[0] + allowed[1]) / 2.0
                            d[feat] = allowed[1] if val >= mid else allowed[0]
                        except Exception:
                            pass
            return pd.Series(d) if is_series else d

        elif is_array:
            arr = sample.copy()
            ref_arr = np.asarray(reference_sample) if reference_sample is not None else None
            for i, feat in enumerate(self.feature_names):
                if i >= arr.shape[-1]:
                    break
                c = self.feature_constraints.get(feat, {})
                if not c.get("mutable", True) and ref_arr is not None:
                    arr[..., i] = ref_arr[..., i]
                    continue
                ftype = c.get("type", "numerical")
                if ftype == "numerical":
                    min_bnd = c.get("min")
                    max_bnd = c.get("max")
                    if min_bnd is not None:
                        arr[..., i] = np.maximum(arr[..., i], min_bnd)
                    if max_bnd is not None:
                        arr[..., i] = np.minimum(arr[..., i], max_bnd)
                elif ftype == "binary":
                    allowed = c.get("allowed_values", [0, 1])
                    if len(allowed) == 2 and all(isinstance(x, (int, float)) for x in allowed):
                        mid = (allowed[0] + allowed[1]) / 2.0
                        arr[..., i] = np.where(arr[..., i] >= mid, allowed[1], allowed[0])
            return arr

        return sample

    def to_dict(self) -> Dict[str, Any]:
        """Convert tabular metadata container to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TabularMetadata":
        """Instantiate TabularMetadata from a dictionary of fields."""
        fields_set = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in fields_set}
        return cls(**filtered)

    @staticmethod
    def infer_feature_types(
        data: Any,
        feature_names: Optional[Sequence[str]] = None,
        explicit_types: Optional[Dict[str, str]] = None,
        numerical_overrides: Optional[Sequence[str]] = None,
        categorical_overrides: Optional[Sequence[str]] = None,
        binary_overrides: Optional[Sequence[str]] = None,
        ordinal_overrides: Optional[Sequence[str]] = None,
        cardinality_threshold: int = 10,
    ) -> Dict[str, Any]:
        """
        Infer tabular feature types distinguishing numerical, binary, categorical, and ordinal.
        Prioritizes explicit configurations and user overrides over heuristic inference.
        Does NOT blindly assume integer-valued features are numerical.

        Args:
            data: pandas DataFrame, 2D numpy array, or list of records.
            feature_names: Optional list of feature names.
            explicit_types: Dict mapping feature names directly to their type.
            numerical_overrides: Explicit list of features to treat as numerical.
            categorical_overrides: Explicit list of features to treat as categorical.
            binary_overrides: Explicit list of features to treat as binary.
            ordinal_overrides: Explicit list of features to treat as ordinal.
            cardinality_threshold: Cardinality below which integer columns are checked for categorical heuristic.

        Returns:
            Dict containing feature_names, feature_types, numerical_features,
            categorical_features, binary_features, ordinal_features, ranges, and cardinalities.
        """
        import pandas as pd

        # Convert to DataFrame if not already
        if isinstance(data, pd.DataFrame):
            df = data
            names = list(df.columns)
        elif isinstance(data, np.ndarray):
            names = list(feature_names) if feature_names else [f"feature_{i}" for i in range(data.shape[1])]
            df = pd.DataFrame(data, columns=names)
        elif isinstance(data, torch.Tensor) or (hasattr(data, "detach") and hasattr(data, "cpu")):
            arr = data.detach().cpu().numpy()
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            names = list(feature_names) if feature_names else [f"feature_{i}" for i in range(arr.shape[1])]
            df = pd.DataFrame(arr, columns=names)
        elif isinstance(data, (list, tuple)) and data and isinstance(data[0], dict):
            df = pd.DataFrame(data)
            names = list(df.columns)
        elif hasattr(data, "values") and hasattr(data, "columns"):
            df = pd.DataFrame(data.values, columns=data.columns)
            names = list(df.columns)
        else:
            names = list(feature_names) if feature_names else []
            df = pd.DataFrame(columns=names)

        # Normalize overrides
        num_set = set(numerical_overrides or [])
        cat_set = set(categorical_overrides or [])
        bin_set = set(binary_overrides or [])
        ord_set = set(ordinal_overrides or [])
        exp_dict = {k: str(v).lower() for k, v in (explicit_types or {}).items()}

        feature_types: Dict[str, str] = {}
        numerical: List[str] = []
        categorical: List[str] = []
        binary: List[str] = []
        ordinal: List[str] = []
        ranges: Dict[str, Tuple[float, float]] = {}
        cardinalities: Dict[str, int] = {}

        for col in names:
            # 1. Check explicit overrides first
            if col in exp_dict:
                assigned_type = exp_dict[col]
            elif col in bin_set:
                assigned_type = "binary"
            elif col in ord_set:
                assigned_type = "ordinal"
            elif col in cat_set:
                assigned_type = "categorical"
            elif col in num_set:
                assigned_type = "numerical"
            else:
                # 2. Heuristic inference
                if col in df.columns:
                    series = df[col].dropna()
                    num_unique = series.nunique()
                    cardinalities[col] = int(num_unique)
                    dtype = series.dtype

                    if num_unique == 2:
                        assigned_type = "binary"
                    elif pd.api.types.is_bool_dtype(dtype):
                        assigned_type = "binary"
                    elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
                        assigned_type = "categorical"
                    elif pd.api.types.is_integer_dtype(dtype):
                        # Integer-valued: Check if it looks like label/index encoding (e.g. 0, 1, 2)
                        min_v = float(series.min())
                        max_v = float(series.max())
                        span = max_v - min_v
                        # Label encoded categories typically start at 0 or 1 with small consecutive values
                        is_likely_code = (
                            min_v >= 0
                            and max_v <= (num_unique + 1)
                            and span <= num_unique
                            and num_unique <= cardinality_threshold
                        )
                        if is_likely_code and len(series) > cardinality_threshold * 2:
                            assigned_type = "categorical"
                        else:
                            assigned_type = "numerical"
                    elif pd.api.types.is_float_dtype(dtype):
                        # Float values with very low cardinality could be discrete
                        if num_unique <= 2:
                            assigned_type = "binary"
                        else:
                            assigned_type = "numerical"
                    else:
                        assigned_type = "numerical"
                else:
                    assigned_type = "numerical"

            clean_assigned = assigned_type.lower()
            feature_types[col] = clean_assigned
            if clean_assigned == "numerical":
                numerical.append(col)
                if col in df.columns:
                    try:
                        vals = df[col].dropna().values.astype(float)
                        if len(vals) > 0:
                            ranges[col] = (float(np.min(vals)), float(np.max(vals)))
                    except Exception:
                        pass
            elif clean_assigned == "binary":
                binary.append(col)
            elif clean_assigned == "ordinal":
                ordinal.append(col)
                if col in df.columns:
                    try:
                        vals = df[col].dropna().values.astype(float)
                        if len(vals) > 0:
                            ranges[col] = (float(np.min(vals)), float(np.max(vals)))
                    except Exception:
                        pass
            else:
                categorical.append(col)

            if col in df.columns and col not in cardinalities:
                try:
                    cardinalities[col] = int(df[col].dropna().nunique())
                except Exception:
                    pass

        return {
            "feature_names": names,
            "feature_types": feature_types,
            "numerical_features": numerical,
            "categorical_features": categorical,
            "binary_features": binary,
            "ordinal_features": ordinal,
            "feature_ranges": ranges,
            "categorical_cardinalities": cardinalities,
            "num_features": len(names),
        }
