"""
AdverScan Explainability Utilities Module.
Contains modular utilities for data conversions, shape validation, and batched predictions.
"""

from app.explainability.utils.conversions import (
    to_numpy,
    to_tensor,
    unwrap_model,
)
from app.explainability.utils.prediction import (
    configure_cpu_threads,
    extract_prediction_and_confidence,
    predict_batched,
    predict_proba_np,
    safe_predict,
)
from app.explainability.utils.validation import (
    ensure_batch,
    get_shape,
    validate_matching_shapes,
    validate_single_sample,
)

__all__ = [
    # Conversions
    "unwrap_model",
    "to_numpy",
    "to_tensor",
    # Validation
    "ensure_batch",
    "get_shape",
    "validate_single_sample",
    "validate_matching_shapes",
    # Prediction & Batching
    "configure_cpu_threads",
    "predict_batched",
    "predict_proba_np",
    "safe_predict",
    "extract_prediction_and_confidence",
]
