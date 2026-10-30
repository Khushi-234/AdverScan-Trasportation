"""
Tests for runtime and benchmark visualization plots (Task 4).
"""

from pathlib import Path
import csv
import json
import tempfile
import pytest

from app.explainability.visualization.run_plots import (
    plot_run_overview,
    plot_speed_vs_faithfulness,
    plot_deletion_curves,
)


def _make_run_results(n: int = 10) -> list:
    """Synthetic run result dicts."""
    results = []
    for i in range(n):
        fallback = i % 5 == 0
        results.append({
            "attack_name": "fgsm" if i % 2 == 0 else "pgd",
            "technique": "saliency" if not fallback else "lime",
            "attribution_status": "success" if i % 7 != 0 else "error",
            "xai_decision": {
                "fallback": fallback,
                "explain_seconds": 0.01 + i * 0.005,
            },
            "metadata": {"explain_seconds": 0.01 + i * 0.005},
        })
    return results


def _write_benchmark_csv(tmp_dir: str, with_curves: bool = False) -> str:
    """Write a synthetic benchmark.csv and return its path."""
    path = Path(tmp_dir) / "benchmark.csv"
    techniques = ["saliency", "gradcam", "integrated_gradients", "shap"]
    speeds = [0.001, 0.005, 0.05, 0.5]
    aucs = [0.62, 0.58, 0.71, 0.68]

    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["technique", "seconds_per_sample", "deletion_auc"]
        if with_curves:
            fieldnames.append("deletion_curve")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t, s, a in zip(techniques, speeds, aucs):
            row = {"technique": t, "seconds_per_sample": s, "deletion_auc": a}
            if with_curves:
                row["deletion_curve"] = json.dumps([1.0 - 0.5 * x / 10 for x in range(11)])
            writer.writerow(row)

    return str(path)


# ===================================================================
# plot_run_overview
# ===================================================================

def test_run_overview_with_results():
    results = _make_run_results(10)
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_run_overview(results, tmp)
        assert path is not None
        assert Path(path).exists()


def test_run_overview_empty():
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_run_overview([], tmp)
        assert path is not None


# ===================================================================
# plot_speed_vs_faithfulness
# ===================================================================

def test_speed_vs_faithfulness_with_benchmark():
    with tempfile.TemporaryDirectory() as tmp:
        bm = _write_benchmark_csv(tmp)
        path = plot_speed_vs_faithfulness(bm, tmp)
        assert path is not None
        assert Path(path).exists()


def test_speed_vs_faithfulness_missing_file(caplog):
    """Returns None and logs warning when file is missing."""
    import logging
    with tempfile.TemporaryDirectory() as tmp:
        with caplog.at_level(logging.WARNING):
            path = plot_speed_vs_faithfulness(tmp + "/nonexistent.csv", tmp)
        assert path is None
        assert any("not found" in rec.message.lower() for rec in caplog.records)


# ===================================================================
# plot_deletion_curves
# ===================================================================

def test_deletion_curves_with_stored_curves():
    with tempfile.TemporaryDirectory() as tmp:
        bm = _write_benchmark_csv(tmp, with_curves=True)
        path = plot_deletion_curves(bm, tmp)
        assert path is not None
        assert Path(path).exists()


def test_deletion_curves_auc_only(caplog):
    """Logs info message when only AUC (no curves) are stored."""
    import logging
    with tempfile.TemporaryDirectory() as tmp:
        bm = _write_benchmark_csv(tmp, with_curves=False)
        with caplog.at_level(logging.INFO):
            path = plot_deletion_curves(bm, tmp)
        assert path is not None
        assert any("only auc" in rec.message.lower() for rec in caplog.records)


def test_deletion_curves_missing_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_deletion_curves(tmp + "/missing.csv", tmp)
        assert path is None
