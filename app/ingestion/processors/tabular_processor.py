"""
Tabular processor implementation for tabular models in AdverScan.
Handles tabular metadata extraction, feature type inference, class mapping,
raw-to-model space transformation, and model input validation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import torch

from app.ingestion.adapters.base_adapter import AdapterCapabilities, BaseModelAdapter
from app.ingestion.exceptions import ModelValidationError
from app.ingestion.metadata.tabular_metadata import TabularMetadata
from app.ingestion.processors.base_processor import BaseProcessor


@dataclass
class AttackCompatibility:
    """
    Standardized attack compatibility evaluation report.
    Explains whether an attack algorithm is runnable against a model given its
    framework adapter capabilities, domain, feature constraints, and preprocessing.
    """
    compatible: bool
    attack_name: str
    reason: str
    required_capabilities: Dict[str, Any] = field(default_factory=dict)
    provided_capabilities: Dict[str, Any] = field(default_factory=dict)


class TabularProcessor(BaseProcessor):
    """
    Dedicated processor for tabular models in AdverScan.
    Standardizes tabular feature preprocessing, raw/model space bidirectional mappings,
    feature-type taxonomy (numerical, categorical, binary, ordinal), target validation,
    class mapping preservation, attack-aware feature constraints, and attack compatibility.
    """

    def __init__(
        self,
        processor_name_or_path: Optional[str] = None,
        processor: Optional[Any] = None,
        preprocessor: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
        feature_names: Optional[Sequence[str]] = None,
        feature_types: Optional[Dict[str, str]] = None,
        numerical_features: Optional[Sequence[str]] = None,
        categorical_features: Optional[Sequence[str]] = None,
        binary_features: Optional[Sequence[str]] = None,
        ordinal_features: Optional[Sequence[str]] = None,
        target_column: Optional[str] = None,
        num_features: Optional[int] = None,
        num_classes: Optional[int] = None,
        class_mapping: Optional[Dict[Any, int]] = None,
        preprocessing_info: Optional[Dict[str, Any]] = None,
        categorical_threshold: int = 10,
        sample_data: Optional[Any] = None,
        feature_constraints: Optional[Dict[str, Dict[str, Any]]] = None,
        immutable_features: Optional[Sequence[str]] = None,
        mutable_features: Optional[Sequence[str]] = None,
        representation_space: Optional[str] = None,
        categorical_encoding_type: Optional[str] = None,
        encoding_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """
        Initialize TabularProcessor.

        Args:
            processor_name_or_path: Optional identifier or path for the processor.
            processor: Optional existing processor or transformer object.
            preprocessor: Optional existing fitted scikit-learn or custom preprocessor (e.g. StandardScaler,
                          ColumnTransformer, Pipeline).
            device: Target execution device.
            feature_names: Ordered list of feature names expected by the model.
            feature_types: Mapping of feature names to 'numerical', 'categorical', 'binary', or 'ordinal'.
            numerical_features: List of numerical features.
            categorical_features: List of nominal categorical features.
            binary_features: List of binary features.
            ordinal_features: List of ordinal features.
            target_column: Name of the target column in the dataset (if present).
            num_features: Expected total count of features.
            num_classes: Expected number of classification classes.
            class_mapping: Mapping from original class label to model output index.
            preprocessing_info: Explicit metadata describing existing model preprocessing.
            categorical_threshold: Cardinality threshold for categorical heuristic inference.
            sample_data: Optional sample DataFrame or array to extract metadata from.
            feature_constraints: Explicit constraints dictionary per feature (bounds, allowed values, mutability).
            immutable_features: Explicit list of feature names that must remain fixed during attack.
            mutable_features: Explicit list of feature names that can be perturbed.
            representation_space: Representation space ('raw', 'scaled', 'encoded', 'scaled_and_encoded').
            categorical_encoding_type: Categorical encoding ('none', 'one-hot', 'ordinal', 'label', 'embedding', 'custom').
            encoding_info: Detailed encoding information per categorical feature.
            **kwargs: Extra parameters.
        """
        self.processor_name = processor_name_or_path
        self._preprocessor = preprocessor or processor
        self._device = torch.device(device) if device is not None else None
        self.target_column = target_column
        self.categorical_threshold = categorical_threshold

        # Initial attributes
        self.feature_names = list(feature_names) if feature_names is not None else []
        self.feature_types = dict(feature_types) if feature_types is not None else {}
        self.numerical_features = list(numerical_features) if numerical_features is not None else []
        self.categorical_features = list(categorical_features) if categorical_features is not None else []
        self.binary_features = list(binary_features) if binary_features is not None else []
        self.ordinal_features = list(ordinal_features) if ordinal_features is not None else []
        self.num_features = num_features or (len(self.feature_names) if self.feature_names else None)
        self.num_classes = num_classes
        self.class_mapping = dict(class_mapping) if class_mapping is not None else None

        # Attack-aware constraints configuration
        self._configured_constraints = dict(feature_constraints) if feature_constraints is not None else {}
        self._configured_immutable = list(immutable_features) if immutable_features is not None else []
        self._configured_mutable = list(mutable_features) if mutable_features is not None else []
        self._configured_rep_space = representation_space
        self._configured_encoding_type = categorical_encoding_type
        self._configured_encoding_info = dict(encoding_info) if encoding_info is not None else {}

        # Inspect preprocessor if available
        self.preprocessing_info = self._inspect_preprocessor(self._preprocessor, preprocessing_info)

        # Build TabularMetadata
        self.tabular_metadata: TabularMetadata = self._build_tabular_metadata(sample_data=sample_data)

        # Synchronize processor fields with tabular_metadata
        self.feature_names = list(self.tabular_metadata.feature_names)
        self.feature_types = dict(self.tabular_metadata.feature_types)
        self.numerical_features = list(self.tabular_metadata.numerical_features)
        self.categorical_features = list(self.tabular_metadata.categorical_features)
        self.binary_features = list(self.tabular_metadata.binary_features)
        self.ordinal_features = list(self.tabular_metadata.ordinal_features)
        self.num_features = self.tabular_metadata.num_features
        self.class_mapping = self.tabular_metadata.class_mapping

    @property
    def data_domain(self) -> str:
        return "tabular"

    def get_processor(self) -> Any:
        """Return underlying raw preprocessor/transformer object if available."""
        return self._preprocessor or self

    def _inspect_preprocessor(
        self,
        preprocessor: Optional[Any],
        explicit_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze an existing fitted preprocessing object (StandardScaler, MinMaxScaler, RobustScaler,
        OneHotEncoder, OrdinalEncoder, LabelEncoder, ColumnTransformer, Pipeline, or custom).
        Does NOT fit or alter the preprocessor.
        """
        if explicit_info is not None:
            return dict(explicit_info)

        if preprocessor is None:
            return {
                "status": "none",
                "description": "No external preprocessing pipeline configured; raw inputs match model space.",
                "transformers": [],
            }

        prep_type = preprocessor.__class__.__name__
        info: Dict[str, Any] = {
            "status": "present",
            "type": prep_type,
            "has_transform": callable(getattr(preprocessor, "transform", None)),
            "has_inverse_transform": callable(getattr(preprocessor, "inverse_transform", None)),
        }

        # Inspect standard scalers
        if prep_type == "StandardScaler":
            if hasattr(preprocessor, "mean_") and preprocessor.mean_ is not None:
                info["mean"] = preprocessor.mean_.tolist() if hasattr(preprocessor.mean_, "tolist") else list(preprocessor.mean_)
            if hasattr(preprocessor, "scale_") and preprocessor.scale_ is not None:
                info["scale"] = preprocessor.scale_.tolist() if hasattr(preprocessor.scale_, "tolist") else list(preprocessor.scale_)
            if hasattr(preprocessor, "var_") and preprocessor.var_ is not None:
                info["var"] = preprocessor.var_.tolist() if hasattr(preprocessor.var_, "tolist") else list(preprocessor.var_)
        elif prep_type == "MinMaxScaler":
            if hasattr(preprocessor, "data_min_") and preprocessor.data_min_ is not None:
                info["data_min"] = preprocessor.data_min_.tolist() if hasattr(preprocessor.data_min_, "tolist") else list(preprocessor.data_min_)
            if hasattr(preprocessor, "data_max_") and preprocessor.data_max_ is not None:
                info["data_max"] = preprocessor.data_max_.tolist() if hasattr(preprocessor.data_max_, "tolist") else list(preprocessor.data_max_)
            if hasattr(preprocessor, "scale_") and preprocessor.scale_ is not None:
                info["scale"] = preprocessor.scale_.tolist() if hasattr(preprocessor.scale_, "tolist") else list(preprocessor.scale_)
            if hasattr(preprocessor, "min_") and preprocessor.min_ is not None:
                info["min"] = preprocessor.min_.tolist() if hasattr(preprocessor.min_, "tolist") else list(preprocessor.min_)
        elif prep_type == "RobustScaler":
            if hasattr(preprocessor, "center_") and preprocessor.center_ is not None:
                info["center"] = preprocessor.center_.tolist() if hasattr(preprocessor.center_, "tolist") else list(preprocessor.center_)
            if hasattr(preprocessor, "scale_") and preprocessor.scale_ is not None:
                info["scale"] = preprocessor.scale_.tolist() if hasattr(preprocessor.scale_, "tolist") else list(preprocessor.scale_)
        elif prep_type in ("OneHotEncoder", "OrdinalEncoder"):
            if hasattr(preprocessor, "categories_") and preprocessor.categories_ is not None:
                info["categories"] = [
                    cat.tolist() if hasattr(cat, "tolist") else list(cat)
                    for cat in preprocessor.categories_
                ]
            if hasattr(preprocessor, "drop_") and preprocessor.drop_ is not None:
                info["drop"] = preprocessor.drop_.tolist() if hasattr(preprocessor.drop_, "tolist") else list(preprocessor.drop_)
        elif prep_type == "LabelEncoder":
            if hasattr(preprocessor, "classes_") and preprocessor.classes_ is not None:
                info["classes"] = preprocessor.classes_.tolist() if hasattr(preprocessor.classes_, "tolist") else list(preprocessor.classes_)
                if self.class_mapping is None:
                    self.class_mapping = {cls_name: i for i, cls_name in enumerate(info["classes"])}
        elif prep_type in ("ColumnTransformer", "Pipeline"):
            steps_info = []
            if hasattr(preprocessor, "named_steps"):
                for step_name, step_obj in preprocessor.named_steps.items():
                    steps_info.append({
                        "name": step_name,
                        "type": step_obj.__class__.__name__,
                        "has_transform": callable(getattr(step_obj, "transform", None)),
                        "has_inverse_transform": callable(getattr(step_obj, "inverse_transform", None)),
                    })
            elif hasattr(preprocessor, "transformers_"):
                for trans_name, trans_obj, trans_cols in preprocessor.transformers_:
                    steps_info.append({
                        "name": trans_name,
                        "type": trans_obj.__class__.__name__ if hasattr(trans_obj, "__class__") else str(trans_obj),
                        "columns": list(trans_cols) if isinstance(trans_cols, (list, tuple, np.ndarray)) else str(trans_cols),
                    })
            info["steps"] = steps_info

        # Determine representation space & encoding type
        rep_space = self._configured_rep_space
        enc_type = self._configured_encoding_type
        enc_info = dict(self._configured_encoding_info)

        if rep_space is None:
            if prep_type in ("StandardScaler", "MinMaxScaler", "RobustScaler"):
                rep_space = "scaled"
            elif prep_type in ("OneHotEncoder", "OrdinalEncoder", "LabelEncoder"):
                rep_space = "encoded"
            elif prep_type in ("ColumnTransformer", "Pipeline"):
                has_scale = any(s in str(info.get("steps", [])).lower() for s in ("scale", "scaler", "minmax", "standard", "robust"))
                has_enc = any(s in str(info.get("steps", [])).lower() for s in ("encoder", "onehot", "ordinal", "label"))
                if has_scale and has_enc:
                    rep_space = "scaled_and_encoded"
                elif has_scale:
                    rep_space = "scaled"
                elif has_enc:
                    rep_space = "encoded"
                else:
                    rep_space = "transformed"
            else:
                rep_space = "raw"

        if enc_type is None:
            if prep_type == "OneHotEncoder":
                enc_type = "one-hot"
            elif prep_type == "OrdinalEncoder":
                enc_type = "ordinal"
            elif prep_type == "LabelEncoder":
                enc_type = "label"
            elif "onehot" in str(info.get("steps", [])).lower():
                enc_type = "one-hot"
            elif "ordinal" in str(info.get("steps", [])).lower():
                enc_type = "ordinal"
            else:
                enc_type = "none"

        info["representation_space"] = rep_space
        info["categorical_encoding_type"] = enc_type
        info["encoding_info"] = enc_info

        return info

    def _build_tabular_metadata(self, sample_data: Optional[Any] = None) -> TabularMetadata:
        """
        Construct TabularMetadata using explicit configuration and optional sample data inference.
        """
        inferred: Dict[str, Any] = {}
        if sample_data is not None:
            # If sample_data has target column, split it before inferring feature types
            X_data = sample_data
            if isinstance(sample_data, pd.DataFrame) and self.target_column and self.target_column in sample_data.columns:
                X_data = sample_data.drop(columns=[self.target_column])

            inferred = TabularMetadata.infer_feature_types(
                data=X_data,
                feature_names=self.feature_names if self.feature_names else None,
                explicit_types=self.feature_types if self.feature_types else None,
                numerical_overrides=self.numerical_features if self.numerical_features else None,
                categorical_overrides=self.categorical_features if self.categorical_features else None,
                binary_overrides=self.binary_features if self.binary_features else None,
                ordinal_overrides=self.ordinal_features if self.ordinal_features else None,
                cardinality_threshold=self.categorical_threshold,
            )

        names = self.feature_names or inferred.get("feature_names", [])
        num_feats = self.num_features or len(names) or inferred.get("num_features", 0)

        # Merge inferred and configured lists
        types_map = dict(inferred.get("feature_types", {}))
        types_map.update(self.feature_types)

        num_list = self.numerical_features or inferred.get("numerical_features", [])
        cat_list = self.categorical_features or inferred.get("categorical_features", [])
        bin_list = self.binary_features or inferred.get("binary_features", [])
        ord_list = self.ordinal_features or inferred.get("ordinal_features", [])

        # Target inspection if sample_data is a DataFrame
        target_dist = None
        target_avail = True
        target_stat = None
        if isinstance(sample_data, pd.DataFrame):
            if self.target_column:
                if self.target_column in sample_data.columns:
                    y_series = sample_data[self.target_column]
                    target_dist = {k: int(v) for k, v in y_series.value_counts().to_dict().items()}
                    if self.num_classes is None:
                        self.num_classes = int(y_series.nunique())
                    if self.class_mapping is None and y_series.dtype == "object":
                        # Preserve original unique labels mapping in order of appearance
                        unique_labels = list(dict.fromkeys(y_series.dropna()))
                        self.class_mapping = {lbl: idx for idx, lbl in enumerate(unique_labels)}
                else:
                    target_avail = False
                    target_stat = f"Specified target_column '{self.target_column}' not found in sample_data."
            else:
                target_avail = False
                target_stat = "target_column not specified; target is unavailable."
        elif sample_data is not None and not self.target_column:
            target_avail = False
            target_stat = "target_column not specified; target is unavailable."

        # If class_mapping is provided but num_classes not set
        if self.class_mapping is not None and self.num_classes is None:
            self.num_classes = len(self.class_mapping)

        return TabularMetadata(
            feature_names=names,
            feature_types=types_map,
            numerical_features=num_list,
            categorical_features=cat_list,
            binary_features=bin_list,
            ordinal_features=ord_list,
            target_column=self.target_column,
            num_features=num_feats,
            num_classes=self.num_classes,
            class_mapping=self.class_mapping,
            target_distribution=target_dist,
            feature_ranges=inferred.get("feature_ranges"),
            categorical_cardinalities=inferred.get("categorical_cardinalities"),
            preprocessing_info=self.preprocessing_info,
            target_available=target_avail,
            target_status=target_stat,
            feature_constraints=self._configured_constraints,
            immutable_features=self._configured_immutable,
            mutable_features=self._configured_mutable,
            representation_space=self.preprocessing_info.get("representation_space", self._configured_rep_space or "raw"),
            categorical_encoding_type=self.preprocessing_info.get("categorical_encoding_type", self._configured_encoding_type or "none"),
            encoding_info=self.preprocessing_info.get("encoding_info", self._configured_encoding_info or {}),
        )

    @property
    def feature_constraints(self) -> Dict[str, Dict[str, Any]]:
        """Return attack-aware feature constraints dictionary."""
        return self.tabular_metadata.feature_constraints

    @property
    def immutable_features(self) -> List[str]:
        """Return list of immutable feature names."""
        return self.tabular_metadata.immutable_features

    @property
    def mutable_features(self) -> List[str]:
        """Return list of mutable feature names."""
        return self.tabular_metadata.mutable_features

    @property
    def representation_space(self) -> str:
        """Return representation space ('raw', 'scaled', 'encoded', 'scaled_and_encoded')."""
        return self.tabular_metadata.representation_space

    @property
    def categorical_encoding_type(self) -> str:
        """Return categorical encoding type."""
        return self.tabular_metadata.categorical_encoding_type

    @property
    def encoding_info(self) -> Dict[str, Any]:
        """Return detailed encoding metadata."""
        return self.tabular_metadata.encoding_info

    @property
    def attack_space(self) -> Dict[str, Dict[str, Any]]:
        """Return structured attack space guidelines per feature."""
        return self.tabular_metadata.attack_space

    def is_mutable(self, feature_name: str) -> bool:
        """Check whether a feature is allowed to be perturbed."""
        return self.tabular_metadata.is_mutable(feature_name)

    def get_mutable_features(self) -> List[str]:
        """Return all features that can be perturbed."""
        return self.tabular_metadata.get_mutable_features()

    def get_immutable_features(self) -> List[str]:
        """Return all features that must remain fixed."""
        return self.tabular_metadata.get_immutable_features()

    def get_feature_constraint(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """Return constraint dictionary for a given feature."""
        return self.tabular_metadata.get_feature_constraint(feature_name)

    def split_features_and_target(
        self,
        data: Any,
        target_column: Optional[str] = None,
    ) -> Tuple[Any, Optional[Any]]:
        """
        Clearly separate features (X) and target (y).
        Never silently assumes the last column is the target.

        Args:
            data: pandas DataFrame, tuple of (X, y), or feature data.
            target_column: Explicit target column name. Defaults to self.target_column.

        Returns:
            Tuple (X, y) where y is None if target is unavailable.

        Raises:
            ModelValidationError: If target_column is specified but not found, or X/y lengths mismatch.
        """
        effective_target = target_column or self.target_column

        # Case 1: Tuple/List of (X, y)
        if isinstance(data, (tuple, list)) and len(data) == 2 and not isinstance(data[0], (str, int, float)):
            X, y = data
            if len(X) != len(y):
                raise ModelValidationError(
                    f"X and y length mismatch: X has {len(X)} samples, y has {len(y)} samples."
                )
            self._validate_missing_values(X, y)
            return X, y

        # Case 2: pandas DataFrame
        if isinstance(data, pd.DataFrame):
            if effective_target:
                if effective_target not in data.columns:
                    raise ModelValidationError(
                        f"Target column '{effective_target}' not found in dataset columns: {list(data.columns)}"
                    )
                X = data.drop(columns=[effective_target])
                y = data[effective_target]
                self._validate_missing_values(X, y)
                return X, y
            else:
                # Do NOT silently assume last column is target!
                self.tabular_metadata.target_available = False
                self.tabular_metadata.target_status = "Target column not specified; target is unavailable."
                self._validate_missing_values(data, None)
                return data, None

        # Case 3: Other data types (numpy array, tensor, etc.)
        self._validate_missing_values(data, None)
        return data, None

    def _validate_missing_values(self, X: Any, y: Optional[Any] = None) -> None:
        """Check for NaN/Inf values where relevant and report issues."""
        if isinstance(X, pd.DataFrame):
            missing_count = int(X.isna().sum().sum())
            if missing_count > 0:
                cols_with_na = list(X.columns[X.isna().any()])
                # Record in tabular metadata
                self.tabular_metadata.target_status = (
                    (self.tabular_metadata.target_status or "")
                    + f" [Warning: X contains {missing_count} missing values across columns {cols_with_na}]"
                ).strip()
        elif isinstance(X, np.ndarray):
            if np.issubdtype(X.dtype, np.number):
                if np.isnan(X).any() or np.isinf(X).any():
                    raise ModelValidationError("Features input array X contains NaN or Inf values.")

        if y is not None:
            if isinstance(y, (pd.Series, pd.DataFrame)):
                if y.isna().any().any() if isinstance(y, pd.DataFrame) else y.isna().any():
                    raise ModelValidationError(f"Target column '{self.target_column}' contains NaN / missing values.")
            elif isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.number):
                if np.isnan(y).any() or np.isinf(y).any():
                    raise ModelValidationError("Target array y contains NaN or Inf values.")

    def transform(self, raw_inputs: Any) -> Union[torch.Tensor, np.ndarray]:
        """
        Transform raw feature space representation into model input space.
        Applies existing preprocessor without fitting new parameters.
        """
        if self._preprocessor is not None and hasattr(self._preprocessor, "transform"):
            transformed = self._preprocessor.transform(raw_inputs)
            if isinstance(transformed, torch.Tensor):
                return transformed
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()
            return np.asarray(transformed, dtype=np.float32)

        # No preprocessor: convert to standard float32 numpy array
        if isinstance(raw_inputs, pd.DataFrame):
            # If explicit feature_names known, align columns
            if self.feature_names and all(col in raw_inputs.columns for col in self.feature_names):
                raw_inputs = raw_inputs[self.feature_names]
            return raw_inputs.values.astype(np.float32)
        elif isinstance(raw_inputs, pd.Series):
            return raw_inputs.values.astype(np.float32)
        elif isinstance(raw_inputs, np.ndarray):
            return raw_inputs.astype(np.float32)
        elif isinstance(raw_inputs, torch.Tensor):
            return raw_inputs.float()
        elif isinstance(raw_inputs, (list, tuple)):
            return np.array(raw_inputs, dtype=np.float32)
        return raw_inputs

    def inverse_transform(self, model_inputs: Any) -> Any:
        """
        Convert model input space representation back to raw/original feature space.
        Preserves original scale, categories, and feature names whenever possible.
        """
        # Convert tensor to numpy if needed
        data = model_inputs.detach().cpu().numpy() if isinstance(model_inputs, torch.Tensor) else np.asarray(model_inputs)

        if self._preprocessor is not None and hasattr(self._preprocessor, "inverse_transform"):
            try:
                inverted = self._preprocessor.inverse_transform(data)
                if self.feature_names and len(self.feature_names) == inverted.shape[-1]:
                    return pd.DataFrame(inverted, columns=self.feature_names)
                return inverted
            except Exception:
                pass

        # If feature names are known, reconstruct DataFrame
        if self.feature_names and len(data.shape) >= 2 and data.shape[-1] == len(self.feature_names):
            return pd.DataFrame(data, columns=self.feature_names)

        return data

    def to_model_space(self, raw_inputs: Any) -> Union[torch.Tensor, np.ndarray]:
        """Alias for transform: map raw space -> model space."""
        return self.transform(raw_inputs)

    def to_raw_space(self, model_inputs: Any) -> Any:
        """Alias for inverse_transform: map model space -> raw space."""
        return self.inverse_transform(model_inputs)

    def process(self, inputs: Any, **kwargs: Any) -> torch.Tensor:
        """
        Process raw inputs into a model-compatible torch.Tensor on target device.
        Implements BaseProcessor.process().
        """
        # If inputs is a DataFrame with target_column, split it
        if isinstance(inputs, pd.DataFrame) and self.target_column and self.target_column in inputs.columns:
            inputs = inputs.drop(columns=[self.target_column])

        # If inputs is already a PyTorch Tensor
        if isinstance(inputs, torch.Tensor):
            tensor = inputs.float()
            return tensor.to(self._device) if self._device is not None else tensor

        # Otherwise transform via preprocessor / numpy
        transformed = self.transform(inputs)

        if isinstance(transformed, torch.Tensor):
            tensor = transformed.float()
        elif isinstance(transformed, np.ndarray):
            tensor = torch.from_numpy(transformed).float()
        else:
            tensor = torch.tensor(transformed, dtype=torch.float32)

        return tensor.to(self._device) if self._device is not None else tensor

    def validate_model_input(
        self,
        adapter: BaseModelAdapter,
        sample_input: Optional[Any] = None,
    ) -> bool:
        """
        Validate model input compatibility before ingestion is considered successful.

        Validates:
            - Expected feature count vs actual feature count
            - Input shape compatibility
            - Feature ordering preservation (rejects silent reordering)
            - Missing/unexpected features
            - Dtype compatibility
            - Output shape compatibility
            - Number of classes match

        Raises:
            ModelValidationError: If any compatibility check fails.
        """
        if sample_input is None:
            return True

        # Determine expected feature count from processor metadata or adapter
        model_feat_count = None
        if hasattr(adapter, "n_features_in_") and adapter.n_features_in_ is not None:
            model_feat_count = int(adapter.n_features_in_)
        elif hasattr(adapter.get_model(), "n_features_in_") and getattr(adapter.get_model(), "n_features_in_") is not None:
            model_feat_count = int(adapter.get_model().n_features_in_)

        expected_features = model_feat_count if model_feat_count is not None else self.num_features

        # Check raw sample input structure and feature ordering before processing
        if isinstance(sample_input, pd.DataFrame):
            input_cols = [c for c in sample_input.columns if c != self.target_column]
            actual_features = len(input_cols)

            if self.feature_names:
                missing = [f for f in self.feature_names if f not in input_cols]
                if missing:
                    raise ModelValidationError(
                        f"Model expects {len(self.feature_names)} features but the provided tabular data is missing: {missing}."
                    )
                extra = [c for c in input_cols if c not in self.feature_names]
                if extra:
                    raise ModelValidationError(
                        f"Model expects {len(self.feature_names)} features but the provided tabular data contains unexpected features: {extra}."
                    )
                # Feature ordering validation: do not silently reorder
                if list(input_cols) != list(self.feature_names):
                    raise ModelValidationError(
                        f"Feature ordering mismatch: provided columns {list(input_cols)} do not match expected feature ordering {list(self.feature_names)}. Silently reordering features is disabled."
                    )

            if expected_features is not None and expected_features != actual_features:
                raise ModelValidationError(
                    f"Feature count mismatch: model expects {expected_features} features but the provided tabular data contains {actual_features}."
                )

        elif hasattr(sample_input, "shape") and len(sample_input.shape) >= 2:
            actual_features = int(sample_input.shape[-1])
            if expected_features is not None and expected_features != actual_features:
                raise ModelValidationError(
                    f"Feature count mismatch: model expects {expected_features} features but the provided tabular data contains {actual_features}."
                )

        # Process sample input to inspect shape and dtype
        proc_input = self.process(sample_input)

        if not isinstance(proc_input, torch.Tensor):
            raise ModelValidationError(
                f"TabularProcessor.process() must produce a torch.Tensor, got {type(proc_input)}"
            )

        if proc_input.ndim < 2:
            raise ModelValidationError(
                f"Expected 2D tabular model input [batch_size, num_features], got shape {tuple(proc_input.shape)}"
            )

        actual_processed_features = int(proc_input.shape[1])
        if expected_features is not None and expected_features != actual_processed_features:
            raise ModelValidationError(
                f"Feature count mismatch: model expects {expected_features} features but the provided tabular data contains {actual_processed_features}."
            )

        # Forward inference validation
        try:
            output = adapter.predict(proc_input)
        except Exception as e:
            raise ModelValidationError(
                f"Tabular model inference failed on sample input: {str(e)}"
            ) from e

        if output is None:
            raise ModelValidationError("Tabular model returned None during sample forward inference.")

        # Check output shape
        out_shape = tuple(output.shape) if hasattr(output, "shape") else None
        if out_shape is not None:
            if len(out_shape) == 1:
                # 1D output of class labels
                pass
            elif len(out_shape) == 2:
                out_classes = int(out_shape[-1])
                if self.num_classes is not None and self.num_classes != out_classes:
                    raise ModelValidationError(
                        f"Class count mismatch: metadata expects {self.num_classes} classes, but model output has {out_classes} classes."
                    )
            else:
                raise ModelValidationError(
                    f"Expected 2D classification output [batch_size, num_classes], got output shape {out_shape}"
                )

        return True

    def interpret_perturbation(
        self,
        raw_original: Any,
        model_perturbed: Optional[Any] = None,
        model_adversarial: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Interpret a model-space adversarial perturbation in the original/raw feature space.
        Maps model_perturbed -> inverse_transform -> raw space, and compares with raw_original.
        Prevents reporting scaled space perturbations (e.g. +0.1) as raw space changes.

        Args:
            raw_original: Original sample in raw feature space (pd.DataFrame, pd.Series, dict, or array).
            model_perturbed: Perturbed sample in model/input space (Tensor, array, or DataFrame).
            model_adversarial: Alias for model_perturbed.

        Returns:
            Dict containing raw original, raw perturbed, semantic deltas, and representation space.
        """
        effective_perturbed = model_perturbed if model_perturbed is not None else model_adversarial
        if effective_perturbed is None:
            raise ValueError("Either model_perturbed or model_adversarial must be provided.")

        # Convert raw_original to DataFrame
        if isinstance(raw_original, pd.DataFrame):
            orig_df = raw_original
        elif isinstance(raw_original, pd.Series):
            orig_df = pd.DataFrame([raw_original])
        elif isinstance(raw_original, dict):
            orig_df = pd.DataFrame([raw_original])
        else:
            orig_df = pd.DataFrame(np.asarray(raw_original), columns=self.feature_names if self.feature_names else None)

        # Invert model_perturbed to raw space
        inverted_raw = self.inverse_transform(effective_perturbed)
        if isinstance(inverted_raw, pd.DataFrame):
            pert_df = inverted_raw
        elif isinstance(inverted_raw, pd.Series):
            pert_df = pd.DataFrame([inverted_raw])
        elif isinstance(inverted_raw, dict):
            pert_df = pd.DataFrame([inverted_raw])
        else:
            pert_df = pd.DataFrame(np.asarray(inverted_raw), columns=self.feature_names if self.feature_names else None)

        deltas: Dict[str, Any] = {}
        features = list(orig_df.columns) if hasattr(orig_df, "columns") else self.feature_names
        for feat in features:
            if feat in orig_df.columns and feat in pert_df.columns:
                orig_v = orig_df[feat].iloc[0] if len(orig_df) > 0 else None
                pert_v = pert_df[feat].iloc[0] if len(pert_df) > 0 else None
                ftype = self.feature_types.get(feat, "numerical")
                if ftype == "numerical":
                    try:
                        delta = float(pert_v) - float(orig_v)
                        deltas[feat] = {
                            "original": float(orig_v),
                            "perturbed": float(pert_v),
                            "delta": delta,
                            "type": "numerical",
                        }
                    except (ValueError, TypeError):
                        deltas[feat] = {
                            "original": orig_v,
                            "perturbed": pert_v,
                            "changed": (orig_v != pert_v),
                            "type": ftype,
                        }
                else:
                    deltas[feat] = {
                        "original": orig_v,
                        "perturbed": pert_v,
                        "changed": (orig_v != pert_v),
                        "transition": f"{orig_v} -> {pert_v}",
                        "type": ftype,
                    }

        delta_num = {k: v["delta"] for k, v in deltas.items() if "delta" in v}
        return {
            "representation_space": self.representation_space,
            "has_inverse_transform": bool(self.preprocessing_info.get("has_inverse_transform", True)),
            "raw_original": orig_df.iloc[0].to_dict() if len(orig_df) > 0 else {},
            "raw_perturbed": pert_df.iloc[0].to_dict() if len(pert_df) > 0 else {},
            "raw_deltas": deltas,
            "delta_numerical": delta_num,
        }

    def check_attack_compatibility(
        self,
        attack_name: Optional[Any] = None,
        adapter: Optional[BaseModelAdapter] = None,
        **kwargs: Any,
    ) -> AttackCompatibility:
        """
        Evaluate compatibility of a specified attack algorithm with this tabular model.
        Checks gradient requirements, domain constraints, and representation needs.
        Accepts arguments in either order: (attack_name, adapter) or (adapter, attack_name),
        as well as keyword arguments.

        Args:
            attack_name: Name of attack (e.g. 'CAP', 'PGD', 'FGSM', 'C&W', 'LowProFool', 'HopSkipJump').
            adapter: Standardized model adapter to inspect capabilities for.

        Returns:
            AttackCompatibility report with boolean decision and rationale.
        """
        actual_attack = attack_name
        actual_adapter = adapter

        if not isinstance(actual_attack, str) and (isinstance(actual_attack, BaseModelAdapter) or hasattr(actual_attack, "capabilities")):
            # Swapped positional arguments: check_attack_compatibility(adapter, "PGD")
            actual_adapter, actual_attack = actual_attack, str(actual_adapter or "")
        elif actual_attack is None and "attack" in kwargs:
            actual_attack = kwargs["attack"]

        clean_name = str(actual_attack or "").strip().upper()
        adapter = actual_adapter
        attack_name = str(actual_attack or "")

        # Extract adapter capabilities if adapter provided
        caps = getattr(adapter, "capabilities", None) if adapter is not None else None
        supports_grads = caps.supports_gradients if caps is not None else False
        supports_pred = caps.supports_predict if caps is not None else True

        # Gradient-requiring attacks
        gradient_attacks = {
            "PGD", "FGSM", "CW", "C&W", "CARLINI-WAGNER", "JSMA", "BIM", "MIM",
            "DEEPFOOL", "APGD", "AUTOPGD", "RFGSM"
        }

        # Tabular-specific constraint-aware attacks
        constraint_attacks = {"CAP", "CONSTRAINT-AWARE PGD", "LOWPROFOOL"}

        # Black-box / Query-based attacks
        blackbox_attacks = {"HOPSKIPJUMP", "ZOO", "BOUNDARY", "SQUARE", "NES", "SIMBA", "RANDOMSEARCH", "RANDOM_SEARCH", "DECISION_TREE"}

        # Generative / Manifold attacks
        manifold_attacks = {"VAE", "ON-MANIFOLD", "GENERATIVE"}

        provided = {
            "framework_adapter": adapter.__class__.__name__ if adapter else "Unknown",
            "supports_gradients": supports_grads,
            "supports_predict": supports_pred,
            "data_domain": "tabular",
            "representation_space": self.representation_space,
            "num_features": self.num_features,
            "has_constraints": bool(self.feature_constraints),
        }

        if any(clean_name == g or clean_name.startswith(g) for g in gradient_attacks):
            required = {"supports_gradients": True}
            if not supports_grads:
                return AttackCompatibility(
                    compatible=False,
                    attack_name=attack_name,
                    reason=(
                        f"Attack '{attack_name}' requires gradient capabilities (supports_gradients=True), "
                        f"but model adapter ({provided['framework_adapter']}) does not support gradients."
                    ),
                    required_capabilities=required,
                    provided_capabilities=provided,
                )
            return AttackCompatibility(
                compatible=True,
                attack_name=attack_name,
                reason=f"Model adapter supports gradients and inference required by {attack_name}.",
                required_capabilities=required,
                provided_capabilities=provided,
            )

        if any(clean_name == c or clean_name.startswith(c) for c in constraint_attacks):
            required = {"data_domain": "tabular", "feature_constraints": True}
            if not supports_grads:
                return AttackCompatibility(
                    compatible=False,
                    attack_name=attack_name,
                    reason=(
                        f"Constraint-aware attack '{attack_name}' requires differentiable model for gradient steps, "
                        f"but model adapter ({provided['framework_adapter']}) does not support gradients."
                    ),
                    required_capabilities={"supports_gradients": True, **required},
                    provided_capabilities=provided,
                )
            return AttackCompatibility(
                compatible=True,
                attack_name=attack_name,
                reason=f"TabularProcessor provides constraint-aware feature space for {attack_name}.",
                required_capabilities=required,
                provided_capabilities=provided,
            )

        if any(clean_name == b or clean_name.startswith(b) for b in blackbox_attacks):
            required = {"supports_predict": True}
            if not supports_pred:
                return AttackCompatibility(
                    compatible=False,
                    attack_name=attack_name,
                    reason=f"Attack '{attack_name}' requires prediction capability.",
                    required_capabilities=required,
                    provided_capabilities=provided,
                )
            return AttackCompatibility(
                compatible=True,
                attack_name=attack_name,
                reason=f"Black-box query attack '{attack_name}' is compatible with model prediction capability.",
                required_capabilities=required,
                provided_capabilities=provided,
            )

        if any(clean_name == m or clean_name.startswith(m) for m in manifold_attacks):
            return AttackCompatibility(
                compatible=False,
                attack_name=attack_name,
                reason=f"On-manifold attack '{attack_name}' requires a pre-trained generative/manifold model (VAE), which is currently unconfigured.",
                required_capabilities={"learned_manifold": True},
                provided_capabilities=provided,
            )

        # Default fallback
        if not supports_grads:
            return AttackCompatibility(
                compatible=False,
                attack_name=attack_name,
                reason=f"Attack '{attack_name}' is not supported on non-gradient model adapter.",
                required_capabilities={"supports_gradients": True},
                provided_capabilities=provided,
            )

        return AttackCompatibility(
            compatible=True,
            attack_name=attack_name,
            reason=f"Attack '{attack_name}' is compatible with tabular processor and adapter capabilities.",
            required_capabilities={},
            provided_capabilities=provided,
        )

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
        """
        return self.tabular_metadata.project_to_constraints(sample, reference_sample)

    def validate_constraints(
        self,
        sample: Any,
        reference_sample: Optional[Any] = None,
    ) -> Tuple[bool, List[str]]:
        """Validate whether sample satisfies all feature constraints."""
        return self.tabular_metadata.validate_constraints(sample, reference_sample)
