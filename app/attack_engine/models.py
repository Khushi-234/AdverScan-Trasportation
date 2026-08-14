"""
Data models and result contracts for the Adversarial Attack Engine.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional


@dataclass
class AttackMetadata:
    """
    Metadata describing an executed adversarial attack.

    Attributes:
        attack_name: Registered identifier of the attack (e.g. 'fgsm', 'pgd', 'deepfool').
        attack_class: Class name of the attack implementation.
        epsilon: Perturbation magnitude or bound used.
        clip_min: Minimum value clip bound.
        clip_max: Maximum value clip bound.
        execution_time_seconds: Time taken to execute the attack.
        parameters: Additional attack-specific parameters.
    """

    attack_name: str
    attack_class: str
    epsilon: Optional[float] = 0.0
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None
    execution_time_seconds: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackResult:
    """
    Standardized result contract returned by every adversarial attack execution.

    Attributes:
        adversarial_examples: Output tensor or batch of generated adversarial examples.
        metadata: AttackMetadata instance describing execution details.
        original_inputs: Optional reference to clean input tensor.
        labels: Optional reference to ground truth labels.
    """

    adversarial_examples: Any
    metadata: AttackMetadata
    original_inputs: Optional[Any] = None
    labels: Optional[Any] = None


@dataclass
class AttackResults:
    """
    Collection container for AttackResult objects returned by multi-attack pipelines.

    Maps attack identifiers (e.g. 'fgsm', 'pgd', 'deepfool') to their AttackResult instances.
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
