"""
Data models and result contracts for the Adversarial Attack Engine.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class AttackMetadata:
    """
    Standardized metadata describing an adversarial attack implementation and its execution.

    Attributes:
        name: Registered identifier of the attack (e.g. 'fgsm', 'pgd', 'deepfool', 'cw', 'fab').
        domain: Target input domain / modality (e.g. 'image', 'text', 'tabular', 'audio').
        category: Methodology category (e.g. 'gradient', 'boundary', 'optimization').
        attack_type: Threat model access type ('white_box' or 'black_box').
        requires_gradient: Whether the attack requires gradient calculation w.r.t inputs.
        requires_loss: Whether the attack requires an explicit loss function.
        requires_model_weights: Whether the attack requires direct access to model weights.
        iterative: Whether the attack performs iterative optimization/steps.
        targeted_supported: Whether targeted attacks are supported.
        untargeted_supported: Whether untargeted attacks are supported.
        query_based: Whether the attack operates primarily via black-box model queries.
        supported_norms: Norm constraints supported (e.g. ['Linf', 'L2']).
        computational_cost: Relative computational cost indicator ('low', 'medium', 'high').
        supported_model_types: Compatible model types (e.g. ['pytorch']).
        attack_name: Alias for name, for execution backward compatibility.
        attack_class: Class name of the implementation.
        epsilon: Perturbation magnitude or bound used.
        clip_min: Minimum value clip bound.
        clip_max: Maximum value clip bound.
        execution_time_seconds: Time taken to execute the attack.
        parameters: Additional attack-specific parameters.
    """

    name: str
    domain: str = "image"
    category: str = "gradient"
    attack_type: str = "white_box"
    requires_gradient: bool = True
    requires_loss: bool = True
    requires_model_weights: bool = True
    iterative: bool = False
    targeted_supported: bool = False
    untargeted_supported: bool = True
    query_based: bool = False
    supported_norms: List[str] = field(default_factory=lambda: ["Linf"])
    computational_cost: str = "low"
    supported_model_types: List[str] = field(default_factory=lambda: ["pytorch"])

    # Execution-level fields (optional, populated during or after execution)
    attack_name: Optional[str] = None
    attack_class: Optional[str] = None
    epsilon: Optional[float] = 0.0
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None
    execution_time_seconds: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.attack_name is None:
            self.attack_name = self.name
        if self.attack_class is None:
            self.attack_class = self.name.upper()

    @property
    def type(self) -> str:
        """Alias for attack_type."""
        return self.attack_type

    @property
    def modality(self) -> str:
        """Alias for domain."""
        return self.domain

    @property
    def attack_category(self) -> str:
        """Alias for category."""
        return self.category


@dataclass
class AttackResult:
    """
    Standardized result contract returned by every adversarial attack execution.

    Attributes:
        adversarial_examples: Output tensor or batch of generated adversarial examples.
        metadata: AttackMetadata instance describing execution details.
        original_inputs: Optional reference to clean input tensor.
        labels: Optional reference to ground truth labels.
        attack_name: Name of the attack executed.
        success: Success mask or boolean indicator (misclassification).
        original_predictions: Predictions on original clean inputs.
        adversarial_predictions: Predictions on perturbed adversarial inputs.
        perturbation_metrics: Computed perturbation norms (L0, L2, Linf, MAE).
        execution_time: Elapsed time for attack execution in seconds.
        query_count: Optional model query count for query-based attacks.
        attack_metadata: Static capability metadata for the attack.
    """

    adversarial_examples: Any
    metadata: AttackMetadata
    original_inputs: Optional[Any] = None
    labels: Optional[Any] = None
    attack_name: Optional[str] = None
    success: Optional[Any] = None
    original_predictions: Optional[Any] = None
    adversarial_predictions: Optional[Any] = None
    perturbation_metrics: Dict[str, float] = field(default_factory=dict)
    execution_time: Optional[float] = None
    query_count: Optional[int] = None
    attack_metadata: Optional[AttackMetadata] = None

    def __post_init__(self):
        if self.attack_name is None and self.metadata:
            self.attack_name = getattr(self.metadata, "name", getattr(self.metadata, "attack_name", None))
        if self.execution_time is None and self.metadata:
            self.execution_time = getattr(self.metadata, "execution_time_seconds", 0.0)
        if self.attack_metadata is None and self.metadata:
            self.attack_metadata = self.metadata

    @property
    def adv_inputs(self) -> Any:
        """
        Convenience property alias for adversarial_examples.
        """
        return self.adversarial_examples


@dataclass
class AttackResults:
    """
    Collection container for AttackResult objects returned by multi-attack pipelines.

    Maps attack identifiers (e.g. 'fgsm', 'pgd', 'deepfool', 'cw', 'fab') to their AttackResult instances.
    """

    results: Dict[str, AttackResult] = field(default_factory=dict)

    def __getitem__(self, key: str) -> AttackResult:
        return self.results[key.lower()]

    def __setitem__(self, key: str, value: AttackResult) -> None:
        self.results[key.lower()] = value

    def __contains__(self, key: str) -> bool:
        return key.lower() in self.results

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self) -> Iterator[str]:
        return iter(self.results)

    def items(self):
        return self.results.items()

    def keys(self):
        return self.results.keys()

    def values(self):
        return self.results.values()

    def get(self, key: str, default: Optional[AttackResult] = None) -> Optional[AttackResult]:
        return self.results.get(key.lower(), default)


__all__ = ["AttackMetadata", "AttackResult", "AttackResults"]
