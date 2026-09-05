"""
Pipeline configuration data model and validator for Module 8 (Orchestration).
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

VALID_METRICS = {
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "average_confidence",
    "average_entropy",
    "per_class_metrics",
    "confusion_matrix",
}


@dataclass
class PipelineConfig:
    """
    Structured configuration container for AdverScan execution pipeline.
    """

    # Experiment Identity (A1.1)
    experiment_id: Optional[str] = None
    experiment_name: str = "Experiment"
    description: Optional[str] = None

    # Model Configuration (M1)
    model_path: Union[str, Any] = "gtsrb_model.pt"
    sample_input: Optional[Any] = None
    device: Optional[str] = "cpu"
    model_class: Optional[Any] = None
    model_name: str = "TargetModel"
    task_type: str = "classification"
    num_classes: int = 43

    # Dataset Configuration (M2)
    dataset_name: str = "bazyl/GTSRB"
    processor_name: str = "bazyl/gtsrb-model"
    split: str = "test"
    batch_size: int = 32
    custom_dataset_loader: Optional[Any] = None

    # Reproducibility Configuration (A1.2)
    seed: Optional[int] = 42
    deterministic: bool = False

    # Sample-Count Control (A1.3)
    sample_count: Optional[int] = None

    # Pipeline Mode ("baseline_only", "attack_assessment", "full")
    mode: str = "full"

    # Attack Configuration (M3)
    attacks: List[str] = field(default_factory=lambda: ["fgsm"])
    attack_configs: Dict[str, Any] = field(default_factory=dict)

    # Evaluation Metric Configuration (A1.4)
    evaluation_metrics: List[str] = field(
        default_factory=lambda: [
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
            "average_confidence",
            "average_entropy",
            "per_class_metrics",
            "confusion_matrix",
        ]
    )

    # XAI Configuration (M6)
    enable_xai: bool = False
    xai_techniques: List[str] = field(default_factory=lambda: ["shap"])

    # Hardening Configuration (M7)
    enable_hardening: bool = False
    defense: str = "auto"
    defense_config: Dict[str, Any] = field(default_factory=dict)

    # Re-Test & Comparison Configuration (M8)
    enable_retest: bool = True

    # Report Generator Configuration (M9)
    enable_report: bool = True

    # MLflow Tracking Configuration (A3)
    enable_mlflow: bool = False
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: Optional[str] = None

    # Output Configuration
    output_dir: Optional[str] = "results/orchestration"

    def validate(self) -> None:
        """
        Validate parameter choices and consistency.
        """
        valid_modes = {"baseline_only", "attack_assessment", "full"}
        if self.mode.lower() not in valid_modes:
            raise ValueError(f"Invalid mode '{self.mode}'. Must be one of {valid_modes}")

        if not self.attacks and self.mode in {"attack_assessment", "full"}:
            raise ValueError("At least one attack must be specified when mode is 'attack_assessment' or 'full'")

        for tech in self.xai_techniques:
            if tech.lower() not in {"shap", "lime"}:
                raise ValueError(f"Unsupported XAI technique '{tech}'. Must be 'shap' or 'lime'")

        if self.sample_count is not None:
            if not isinstance(self.sample_count, int) or isinstance(self.sample_count, bool) or self.sample_count <= 0:
                raise ValueError(f"Invalid sample_count '{self.sample_count}'. Must be a positive integer.")

        if self.seed is not None:
            if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
                raise ValueError(f"Invalid seed '{self.seed}'. Must be a non-negative integer.")

        if self.experiment_id is not None:
            if not isinstance(self.experiment_id, str) or not self.experiment_id.strip():
                raise ValueError("experiment_id must be a non-empty string if provided.")

        for metric in self.evaluation_metrics:
            if metric.lower() not in VALID_METRICS:
                raise ValueError(f"Unsupported evaluation metric '{metric}'. Must be one of {VALID_METRICS}")

    def get_configuration_hash(self) -> str:
        """
        Compute a stable, deterministic configuration hash for checkpoint compatibility.
        """
        from app.orchestration.checkpoint_manager import compute_configuration_hash

        total_samples = self.sample_count if self.sample_count is not None else 0
        extra_params = {
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "mode": self.mode,
            "enable_hardening": self.enable_hardening,
            "enable_xai": self.enable_xai,
        }
        return compute_configuration_hash(
            dataset_name=self.dataset_name,
            total_samples=total_samples,
            batch_size=self.batch_size,
            attacks=self.attacks,
            extra_params=extra_params,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert PipelineConfig to dictionary representation.
        """
        def _clean_val(val: Any) -> Any:
            if val is None or isinstance(val, (int, float, str, bool)):
                return val
            if isinstance(val, (list, tuple)):
                return [_clean_val(item) for item in val]
            if isinstance(val, dict):
                return {str(k): _clean_val(v) for k, v in val.items()}
            if hasattr(val, "to_dict") and callable(val.to_dict):
                return val.to_dict()
            return str(val)

        res = {}
        for key in self.__dataclass_fields__:
            raw_val = getattr(self, key)
            res[key] = _clean_val(raw_val)
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """
        Construct PipelineConfig instance from dictionary.
        """
        known_fields = set(cls.__dataclass_fields__.keys())
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)

    def to_json(self, filepath: Optional[Union[str, Path]] = None) -> str:
        """
        Serialize configuration to JSON string or save to file.
        """
        dict_data = self.to_dict()
        json_str = json.dumps(dict_data, indent=2)
        if filepath is not None:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_json(cls, json_str_or_path: Union[str, Path]) -> "PipelineConfig":
        """
        Deserialize configuration from JSON string or JSON filepath.
        """
        if isinstance(json_str_or_path, Path):
            with open(json_str_or_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            s = str(json_str_or_path).strip()
            if not s.startswith("{") and (s.endswith(".json") or len(s) < 256):
                p = Path(s)
                if p.exists() and p.is_file():
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                else:
                    data = json.loads(s)
            else:
                data = json.loads(s)
        return cls.from_dict(data)

    def to_yaml(self, filepath: Optional[Union[str, Path]] = None) -> str:
        """
        Serialize configuration to YAML string or save to file.
        """
        dict_data = self.to_dict()
        yaml_str = yaml.dump(dict_data, sort_keys=False)
        if filepath is not None:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(yaml_str)
        return yaml_str

    @classmethod
    def from_yaml(cls, yaml_str_or_path: Union[str, Path]) -> "PipelineConfig":
        """
        Deserialize configuration from YAML string or YAML filepath.
        """
        if isinstance(yaml_str_or_path, Path):
            with open(yaml_str_or_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            s = str(yaml_str_or_path).strip()
            if not s.startswith(("{", "---")) and (s.endswith((".yaml", ".yml")) or len(s) < 256):
                p = Path(s)
                if p.exists() and p.is_file():
                    with open(p, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                else:
                    data = yaml.safe_load(s)
            else:
                data = yaml.safe_load(s)
        return cls.from_dict(data)

