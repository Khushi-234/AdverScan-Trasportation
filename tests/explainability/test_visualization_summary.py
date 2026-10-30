"""
Tests for dataset summary visualization plots (Task 3).
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from app.explainability.visualization.summary_plots import (
    plot_attack_summary,
    plot_failure_modes,
    plot_exploited_features,
    plot_attribution_drift,
    plot_confidence_shift,
)


def _make_results(n: int = 20) -> list:
    """Create n synthetic explanation result dicts across 2 attacks and 2 modalities."""
    results = []
    attacks = ["fgsm", "pgd"]
    modalities = ["image", "tabular"]
    np.random.seed(0)

    for i in range(n):
        atk = attacks[i % 2]
        mod = modalities[i % 2]
        success = bool(i % 3 == 0)  # every 3rd is a success

        results.append({
            "attack_name": atk,
            "modality": mod,
            "technique": "integrated_gradients",
            "clean_confidence": float(0.85 + 0.1 * np.random.randn()),
            "adversarial_confidence": float(0.3 + 0.1 * np.random.randn()),
            "failure_analysis": {
                "attack_caused_failure": success,
                "clean_correct": True,
                "failure_mode": "clean_correct_to_adversarial_incorrect" if success
                                 else "clean_correct_to_adversarial_correct",
                "contributing_features": [
                    {"feature": f"f{j}", "importance_shift": float(np.random.randn()), "rank": j + 1}
                    for j in range(3)
                ],
            },
            "comparison": {
                "normalized_attribution_l1": float(abs(np.random.randn() * 0.2)),
                "attribution_cosine_similarity": float(min(1.0, abs(0.8 + np.random.randn() * 0.1))),
                "attribution_comparison_status": "success",
            },
        })
    return results


def test_attack_summary_20_results():
    results = _make_results(20)
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_attack_summary(results, tmp)
        assert path is not None
        assert Path(path).exists()


def test_attack_summary_single_attack():
    results = [_make_results(4)[0]]  # single result
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_attack_summary(results, tmp)
        assert path is not None


def test_attack_summary_empty():
    """Empty results must render 'no data' plot without raising."""
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_attack_summary([], tmp)
        assert path is not None


def test_failure_modes_20_results():
    results = _make_results(20)
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_failure_modes(results, tmp)
        assert path is not None
        assert Path(path).exists()


def test_failure_modes_empty():
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_failure_modes([], tmp)
        assert path is not None


def test_exploited_features_tabular_results():
    """Exploited features aggregation works for non-image modalities."""
    results = [r for r in _make_results(20) if r["modality"] == "tabular"]
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_exploited_features(results, tmp)
        assert path is not None
        assert Path(path).exists()


def test_exploited_features_skips_image(caplog):
    """Image modality results are skipped with a log message."""
    import logging
    results = [r for r in _make_results(20) if r["modality"] == "image"]
    with tempfile.TemporaryDirectory() as tmp:
        with caplog.at_level(logging.INFO, logger="app.explainability.visualization.summary_plots"):
            path = plot_exploited_features(results, tmp)
        # Should still return a path (no data plot)
        assert path is not None
        assert any("skipping image" in rec.message.lower() for rec in caplog.records)


def test_exploited_features_empty():
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_exploited_features([], tmp)
        assert path is not None


def test_attribution_drift_20_results():
    results = _make_results(20)
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_attribution_drift(results, tmp)
        assert path is not None
        assert Path(path).exists()


def test_attribution_drift_empty():
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_attribution_drift([], tmp)
        assert path is not None


def test_confidence_shift_20_results():
    results = _make_results(20)
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_confidence_shift(results, tmp)
        assert path is not None
        assert Path(path).exists()


def test_confidence_shift_empty():
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_confidence_shift([], tmp)
        assert path is not None
