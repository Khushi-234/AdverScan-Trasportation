"""
Unit tests for MetricsCalculator class.
"""

import numpy as np
from app.evaluation.metrics import MetricsCalculator


def test_compute_entropy():
    """Test Shannon entropy calculation on probability distributions."""
    # Deterministic probability distribution (entropy should be near 0)
    deterministic_probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    entropy_det = MetricsCalculator.compute_entropy(deterministic_probs)
    assert np.allclose(entropy_det, 0.0, atol=1e-3)

    # Uniform probability distribution over 3 classes (entropy = log2(3) approx 1.58496)
    uniform_probs = np.array([[1.0 / 3, 1.0 / 3, 1.0 / 3]])
    entropy_uni = MetricsCalculator.compute_entropy(uniform_probs)
    assert np.isclose(entropy_uni[0], np.log2(3), atol=1e-3)


def test_compute_metrics():
    """Test computing complete metrics dict for synthetic targets and predictions."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 1])  # 5/6 correct -> accuracy approx 0.8333
    y_probs = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.85, 0.1, 0.05],
        [0.05, 0.9, 0.05],
        [0.1, 0.7, 0.2],  # Incorrect prediction (class 1 instead of 2)
    ])

    metrics = MetricsCalculator.compute_metrics(y_true, y_pred, y_probs, num_classes=3)

    assert np.isclose(metrics["accuracy"], 5 / 6)
    assert "precision_macro" in metrics
    assert "recall_macro" in metrics
    assert "f1_macro" in metrics
    assert "precision_weighted" in metrics
    assert "recall_weighted" in metrics
    assert "f1_weighted" in metrics
    assert "average_confidence" in metrics
    assert "average_entropy" in metrics
    assert metrics["average_confidence"] > 0.7
    assert len(metrics["confusion_matrix"]) == 3
    assert len(metrics["per_class_metrics"]) == 3
