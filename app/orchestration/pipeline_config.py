"""
Pipeline configuration data model and validator for Module 8 (Orchestration).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class PipelineConfig:
    """
    Structured configuration container for AdverScan execution pipeline.
    """

    # Model Configuration (M1)
    model_path: Union[str, Any]
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

    # Pipeline Mode ("baseline_only", "attack_assessment", "full")
    mode: str = "full"

    # Attack Configuration (M3)
    attacks: List[str] = field(default_factory=lambda: ["fgsm"])
    attack_configs: Dict[str, Any] = field(default_factory=dict)

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
