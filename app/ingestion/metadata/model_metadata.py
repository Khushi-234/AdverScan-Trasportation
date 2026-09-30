"""
Metadata container for storing standard model properties.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class ModelMetadata:
    """
    Data container holding metadata for an ingested model.
    """

    framework: str = "pytorch"
    model_name: Optional[str] = None
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    num_classes: Optional[int] = None
    task_type: Optional[str] = "classification"
    device: Optional[str] = "cpu"
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata container to a dictionary."""
        return asdict(self)
