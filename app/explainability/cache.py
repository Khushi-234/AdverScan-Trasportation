"""
In-memory caching utilities for AdverScan XAI explanations.
Avoids redundant attribution computation for clean inputs across multiple adversarial attacks.
"""

import hashlib
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch


def cache_key(
    x: Any,
    technique: str,
    target_class: Any,
    model_id: Any = "default",
) -> str:
    """
    Generate a deterministic cache key string based on input bytes and explanation context.

    Args:
        x: Clean input tensor, numpy array, text string, or sequence.
        technique: Name of the explanation technique (e.g. 'saliency', 'integrated_gradients').
        target_class: Target class index or label for attribution.
        model_id: Identifier or representation of the target model.

    Returns:
        Deterministic string key containing the sha256 hash of x bytes.
    """
    if isinstance(x, torch.Tensor):
        raw_bytes = x.detach().cpu().contiguous().numpy().tobytes()
    elif isinstance(x, np.ndarray):
        raw_bytes = np.ascontiguousarray(x).tobytes()
    elif isinstance(x, str):
        raw_bytes = x.encode("utf-8")
    elif isinstance(x, bytes):
        raw_bytes = x
    elif isinstance(x, (list, tuple)):
        raw_bytes = repr(x).encode("utf-8")
    else:
        raw_bytes = str(x).encode("utf-8")

    x_hash = hashlib.sha256(raw_bytes).hexdigest()
    t_cls = int(target_class) if isinstance(target_class, (int, np.integer)) else str(target_class)
    m_id = str(model_id)
    tech = str(technique).lower()
    return f"{m_id}:{tech}:{t_cls}:{x_hash}"


class ExplanationCache:
    """
    Simple in-memory dictionary cache for explanation attribution results.
    """

    def __init__(self) -> None:
        """Initialize an empty in-memory explanation cache."""
        self._cache: Dict[str, Any] = {}

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve cached explanation by key, or None if key is not found.

        Args:
            key: Deterministic cache key string.

        Returns:
            Cached explanation result or None.
        """
        return self._cache.get(key)

    def put(self, key: str, value: Any) -> None:
        """
        Store an explanation result in the in-memory cache.

        Args:
            key: Deterministic cache key string.
            value: Explanation result or (attr_result, stats) tuple to store.
        """
        self._cache[key] = value

    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._cache.clear()

    def __len__(self) -> int:
        """Return the number of cached items."""
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self._cache
