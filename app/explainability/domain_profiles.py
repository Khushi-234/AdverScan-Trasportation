"""
Domain profiles and modality-specific configurations for AdverScan XAI.
Includes presets for Fraud Detection, Network IDS, Healthcare/Medical, and Predictive Maintenance.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from app.explainability.input_utils import to_numpy
from app.explainability.modality import Modality


@dataclass
class DomainProfile:
    """
    Configuration profile defining domain constraints, features, techniques, and validity rules.
    """

    name: str
    modality: Modality
    default_technique: str
    compatible_techniques: List[str]
    baseline_strategy: str = "zeros"
    description: str = ""
    default_params: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    integer_features: List[str] = field(default_factory=list)
    immutable_features: List[str] = field(default_factory=list)
    feature_bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = field(default_factory=dict)
    sensor_names: Optional[List[str]] = None
    class_names: Optional[List[str]] = None

    def is_technique_compatible(self, technique_name: str) -> bool:
        """Check if an XAI technique is compatible with this domain profile."""
        norm_name = technique_name.lower().replace("-", "_")
        return any(norm_name == t.lower().replace("-", "_") for t in self.compatible_techniques)


# Preset Domain Profiles
DOMAIN_PROFILES: Dict[str, DomainProfile] = {
    # 1. Financial Fraud Detection (Tabular)
    "fraud": DomainProfile(
        name="fraud",
        modality=Modality.TABULAR,
        default_technique="treeshap",
        compatible_techniques=["treeshap", "shap", "lime", "saliency", "integrated_gradients"],
        baseline_strategy="mean",
        description="Financial transaction fraud detection domain.",
        feature_names=[
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "type_TRANSFER",
            "type_CASH_OUT",
        ],
        categorical_features=["type_TRANSFER", "type_CASH_OUT"],
        integer_features=[],
        immutable_features=["type_TRANSFER", "type_CASH_OUT"],
        feature_bounds={
            "amount": (0.0, None),
            "oldbalanceOrg": (0.0, None),
            "newbalanceOrig": (0.0, None),
            "oldbalanceDest": (0.0, None),
            "newbalanceDest": (0.0, None),
            "type_TRANSFER": (0.0, 1.0),
            "type_CASH_OUT": (0.0, 1.0),
        },
        class_names=["Legitimate", "Fraud"],
        default_params={"num_samples": 100, "steps": 50},
    ),
    # 2. Network Intrusion Detection (IDS Tabular)
    "ids": DomainProfile(
        name="ids",
        modality=Modality.TABULAR,
        default_technique="treeshap",
        compatible_techniques=["treeshap", "shap", "lime", "saliency", "integrated_gradients"],
        baseline_strategy="mean",
        description="Network Intrusion Detection System (NSL-KDD / CIC-IDS).",
        feature_names=[
            "duration",
            "protocol_type",
            "src_bytes",
            "dst_bytes",
            "count",
            "srv_count",
        ],
        categorical_features=["protocol_type"],
        integer_features=["protocol_type", "src_bytes", "dst_bytes", "count", "srv_count"],
        immutable_features=["protocol_type"],
        feature_bounds={
            "duration": (0.0, None),
            "protocol_type": (0.0, 2.0),
            "src_bytes": (0.0, None),
            "dst_bytes": (0.0, None),
            "count": (0.0, None),
            "srv_count": (0.0, None),
        },
        class_names=["Normal", "Attack"],
        default_params={"num_samples": 100, "steps": 50},
    ),
    # 3. Medical Imaging & Clinical Records
    "medical": DomainProfile(
        name="medical",
        modality=Modality.IMAGE,
        default_technique="gradcam",
        compatible_techniques=["gradcam", "integrated_gradients", "saliency", "shap", "lime"],
        baseline_strategy="zeros",
        description="Medical diagnostic domain (clinical markers and medical imaging).",
        feature_bounds={"pixel": (0.0, 1.0)},
        class_names=["Benign", "Malignant"],
        default_params={"steps": 50},
    ),
    # 4. Predictive Maintenance (Time-Series Sensor Streams)
    "maintenance": DomainProfile(
        name="maintenance",
        modality=Modality.TIMESERIES,
        default_technique="integrated_gradients",
        compatible_techniques=["integrated_gradients", "saliency", "shap", "lime"],
        baseline_strategy="mean",
        description="Predictive maintenance turbofan/machinery degradation (CMAPSS).",
        sensor_names=[
            "T24", "T30", "T50", "P30", "Nf", "Nc", "Ps30", "phi", "NRf", "NRc",
            "BPR", "htBleed", "W31", "W32",
        ],
        class_names=["Normal_Degradation", "Imminent_Failure"],
        default_params={"steps": 50},
    ),
    # Modality Fallback Profiles
    "tabular": DomainProfile(
        name="tabular",
        modality=Modality.TABULAR,
        default_technique="shap",
        compatible_techniques=["shap", "treeshap", "lime", "saliency", "integrated_gradients"],
        baseline_strategy="mean",
        description="Generic tabular domain.",
        default_params={"num_samples": 100, "steps": 50},
    ),
    "image": DomainProfile(
        name="image",
        modality=Modality.IMAGE,
        default_technique="shap",
        compatible_techniques=["shap", "lime", "saliency", "integrated_gradients", "gradcam"],
        baseline_strategy="zeros",
        description="Generic computer vision domain.",
        default_params={"num_samples": 100, "steps": 50},
    ),
    "timeseries": DomainProfile(
        name="timeseries",
        modality=Modality.TIMESERIES,
        default_technique="integrated_gradients",
        compatible_techniques=["integrated_gradients", "saliency", "shap", "lime"],
        baseline_strategy="zeros",
        description="Generic sequential/time-series domain.",
        default_params={"steps": 50},
    ),
    "text": DomainProfile(
        name="text",
        modality=Modality.TEXT,
        default_technique="shap",
        compatible_techniques=["shap", "lime", "saliency", "integrated_gradients"],
        baseline_strategy="mask",
        description="Generic natural language domain.",
        default_params={"num_samples": 100},
    ),
}


def is_valid_adversarial(
    clean_sample: Any,
    adversarial_sample: Any,
    profile: DomainProfile,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate that an adversarial sample respects physical, domain, and data integrity constraints.

    Checks:
    1. Bounds violations (e.g., negative byte counts, probabilities not in [0, 1]).
    2. Fractional violations on integer/discrete features (e.g., protocol_type=1.42).
    3. Perturbations to immutable features (e.g., altering protocol_type or customer ID).

    Args:
        clean_sample: Original clean input sample.
        adversarial_sample: Perturbed adversarial input sample.
        profile: DomainProfile governing domain constraints.

    Returns:
        Tuple of (is_valid: bool, validity_flags: Dict[str, Any]).
    """
    violations = []
    flags: Dict[str, Any] = {
        "is_valid": True,
        "bounds_violations": [],
        "integer_violations": [],
        "immutable_violations": [],
    }

    try:
        clean_arr = to_numpy(clean_sample, dtype=np.float64).ravel()
        adv_arr = to_numpy(adversarial_sample, dtype=np.float64).ravel()

        if len(clean_arr) != len(adv_arr):
            flags["is_valid"] = False
            flags["message"] = "Sample lengths do not match."
            return False, flags

        feature_names = profile.feature_names or [f"f{i}" for i in range(len(clean_arr))]

        for idx, val in enumerate(adv_arr):
            if idx >= len(feature_names):
                break
            fname = feature_names[idx]
            clean_val = clean_arr[idx]

            # 1. Bounds check
            if fname in profile.feature_bounds:
                low, high = profile.feature_bounds[fname]
                if low is not None and val < (low - 1e-4):
                    flag_entry = {"feature": fname, "value": float(val), "bound": float(low), "type": "lower"}
                    flags["bounds_violations"].append(flag_entry)
                    violations.append(flag_entry)
                if high is not None and val > (high + 1e-4):
                    flag_entry = {"feature": fname, "value": float(val), "bound": float(high), "type": "upper"}
                    flags["bounds_violations"].append(flag_entry)
                    violations.append(flag_entry)

            # 2. Integer constraint check
            if fname in profile.integer_features:
                if not np.isclose(val, np.round(val), atol=1e-3):
                    flag_entry = {"feature": fname, "value": float(val), "expected": "integer"}
                    flags["integer_violations"].append(flag_entry)
                    violations.append(flag_entry)

            # 3. Immutable feature check
            if fname in profile.immutable_features:
                if not np.isclose(val, clean_val, atol=1e-4):
                    flag_entry = {
                        "feature": fname,
                        "clean_value": float(clean_val),
                        "adv_value": float(val),
                        "type": "immutable_feature_modified",
                    }
                    flags["immutable_violations"].append(flag_entry)
                    violations.append(flag_entry)

        is_valid = len(violations) == 0
        flags["is_valid"] = is_valid
        return is_valid, flags
    except Exception as e:
        flags["is_valid"] = False
        flags["error"] = str(e)
        return False, flags


def get_domain_profile(name_or_modality: Union[str, Modality]) -> DomainProfile:
    """
    Retrieve domain profile by domain name ('fraud', 'ids', 'medical', 'maintenance') or modality.

    Args:
        name_or_modality: Domain name string or Modality enum.

    Returns:
        DomainProfile object.
    """
    if isinstance(name_or_modality, Modality):
        key = name_or_modality.value.lower()
    else:
        key = name_or_modality.lower().strip()

    if key in DOMAIN_PROFILES:
        return DOMAIN_PROFILES[key]

    # Try mapping as Modality string
    try:
        mod = Modality.from_str(key)
        if mod.value.lower() in DOMAIN_PROFILES:
            return DOMAIN_PROFILES[mod.value.lower()]
    except Exception:
        pass

    raise ValueError(
        f"Unknown domain profile '{name_or_modality}'. "
        f"Available profiles: {list(DOMAIN_PROFILES.keys())}"
    )
