"""
Metrics calculation utilities for machine learning baseline evaluation in AdverScan.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


class MetricsCalculator:
    """
    Calculates classification metrics, confidence, entropy, and confusion matrices.
    """

    @staticmethod
    def compute_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """
        Calculate Shannon entropy per sample across class probability distribution.
        H(p) = - sum(p_i * log2(p_i + eps))

        Args:
            probabilities: Array of shape (N, num_classes) with softmax probabilities.
            eps: Epsilon for numerical stability to avoid log(0).

        Returns:
            np.ndarray: Array of shape (N,) with per-sample entropy values.
        """
        probs_clipped = np.clip(probabilities, eps, 1.0)
        return -np.sum(probs_clipped * np.log2(probs_clipped), axis=-1)

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
        num_classes: int = 43,
    ) -> Dict[str, Any]:
        """
        Compute full baseline evaluation metrics.

        Args:
            y_true: Ground-truth target labels (N,).
            y_pred: Predicted class labels (N,).
            y_probs: Predicted class probabilities (N, num_classes).
            num_classes: Number of target classes.

        Returns:
            Dict containing accuracy, precision/recall/F1 (macro & weighted),
            confidence statistics, average entropy, per-class metrics, and confusion matrix.
        """
        # Overall accuracy
        acc = float(accuracy_score(y_true, y_pred))

        # Macro metrics
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        # Weighted metrics
        prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        # Per-class metrics
        prec_class, rec_class, f1_class, support_class = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(num_classes)), zero_division=0
        )

        per_class_metrics: Dict[str, Dict[str, float]] = {}
        for c in range(num_classes):
            per_class_metrics[str(c)] = {
                "precision": float(prec_class[c]),
                "recall": float(rec_class[c]),
                "f1": float(f1_class[c]),
                "support": int(support_class[c]),
            }

        # Confidence statistics
        confidences = np.max(y_probs, axis=-1)
        avg_confidence = float(np.mean(confidences))

        # Entropy statistics
        entropies = MetricsCalculator.compute_entropy(y_probs)
        avg_entropy = float(np.mean(entropies))

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        cm_list: List[List[int]] = cm.tolist()

        return {
            "accuracy": acc,
            "precision_macro": float(prec_macro),
            "recall_macro": float(rec_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(prec_weighted),
            "recall_weighted": float(rec_weighted),
            "f1_weighted": float(f1_weighted),
            "average_confidence": avg_confidence,
            "average_entropy": avg_entropy,
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": cm_list,
        }
