"""
Metadata container for storing standard model properties.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class ModelMetadata:
    """
    Data container holding metadata and runtime configuration for an ingested model.
    Normalizes both preset registry configurations and user-provided custom model configurations.
    """

    model_id: Optional[str] = None
    framework: str = "pytorch"
    model_name: Optional[str] = None
    dataset_name: Optional[str] = None
    data_domain: Optional[str] = None
    domain: Optional[str] = None
    task_type: Optional[str] = "classification"
    processor_name: Optional[str] = None
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    num_classes: Optional[int] = None
    device: Optional[str] = "cpu"
    total_parameters: Optional[int] = None
    trainable_parameters: Optional[int] = None
    model_architecture: Optional[str] = None
    tabular_metadata: Optional[Any] = None
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data_domain is not None and self.domain is None:
            self.domain = self.data_domain
        elif self.domain is not None and self.data_domain is None:
            self.data_domain = self.domain

        if self.model_name is None and self.model_id is not None:
            self.model_name = self.model_id.split("/")[-1]

        if self.tabular_metadata is not None and "tabular_metadata" not in self.extra_info:
            self.extra_info["tabular_metadata"] = (
                self.tabular_metadata.to_dict()
                if hasattr(self.tabular_metadata, "to_dict")
                else self.tabular_metadata
            )
        elif self.tabular_metadata is None and "tabular_metadata" in self.extra_info:
            self.tabular_metadata = self.extra_info["tabular_metadata"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata container to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Instantiate ModelMetadata from a dictionary of configuration fields."""
        fields_set = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        extra = {}
        for k, v in data.items():
            if k in fields_set:
                filtered[k] = v
            else:
                extra[k] = v

        if "extra_info" in filtered and isinstance(filtered["extra_info"], dict):
            filtered["extra_info"] = {**filtered["extra_info"], **extra}
        elif extra:
            filtered["extra_info"] = extra

        return cls(**filtered)

