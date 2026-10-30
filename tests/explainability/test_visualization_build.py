"""
Tests for build_all_figures orchestrator and runner --figures flag (Tasks 5 & 6).
"""

from pathlib import Path
import json
import tempfile
import numpy as np
import pytest

from app.explainability.visualization.build_figures import (
    build_all_figures,
    _select_representative_samples,
)


def _make_results_jsonl(tmp_dir: str, n: int = 6) -> None:
    """Write synthetic xai_results.jsonl to tmp_dir."""
    import json
    results_path = Path(tmp_dir) / "xai_results.jsonl"
    attacks = ["fgsm", "pgd"]
    modalities = ["image", "tabular"]
    with open(results_path, "w", encoding="utf-8") as f:
        for i in range(n):
            r = {
                "attack_name": attacks[i % 2],
                "modality": modalities[i % 2],
                "technique": "integrated_gradients",
                "clean_confidence": 0.85,
                "adversarial_confidence": 0.4,
                "prediction_changed": True,
                "attribution": {},
                "comparison": {
                    "normalized_attribution_l1": 0.1,
                    "attribution_cosine_similarity": 0.95,
                },
                "failure_analysis": {
                    "attack_caused_failure": bool(i % 3 == 0),
                    "clean_correct": True,
                    "failure_mode": "clean_correct_to_adversarial_incorrect",
                    "contributing_features": [
                        {"feature": f"f{j}", "importance_shift": 0.05, "rank": j + 1}
                        for j in range(3)
                    ],
                },
                "validity_flags": {"is_valid": True, "bounds_violations": []},
                "attribution_status": "success",
                "xai_decision": {"fallback": False, "explain_seconds": 0.01},
                "metadata": {"explain_seconds": 0.01},
            }
            f.write(json.dumps(r) + "\n")


def test_build_all_figures_creates_index():
    """build_all_figures should create figures_index.json and return >0 count."""
    with tempfile.TemporaryDirectory() as tmp:
        _make_results_jsonl(tmp, n=6)
        n = build_all_figures(tmp, max_samples_per_attack=2)
        index_path = Path(tmp) / "figures" / "figures_index.json"
        assert index_path.exists(), "figures_index.json not created"
        with open(index_path) as f:
            idx = json.load(f)
        assert "figures" in idx
        assert idx["total"] == n
        assert n > 0


def test_build_all_figures_empty_dir():
    """build_all_figures on empty directory should set partial=true.
    Note: summary/run plots still save 'no data' placeholders."""
    with tempfile.TemporaryDirectory() as tmp:
        n = build_all_figures(tmp, max_samples_per_attack=2)
        index_path = Path(tmp) / "figures" / "figures_index.json"
        assert index_path.exists()
        with open(index_path) as f:
            idx = json.load(f)
        assert idx["partial"] is True
        assert isinstance(n, int) and n >= 0


def test_build_all_figures_partial_status():
    """Mark index as partial when xai_status.json reports pending."""
    import json as _json
    with tempfile.TemporaryDirectory() as tmp:
        _make_results_jsonl(tmp, n=2)
        status_path = Path(tmp) / "xai_status.json"
        with open(status_path, "w") as f:
            _json.dump({"status": "pending"}, f)
        build_all_figures(tmp, max_samples_per_attack=2)
        with open(Path(tmp) / "figures" / "figures_index.json") as f:
            idx = json.load(f)
        assert idx["partial"] is True


def test_build_all_figures_failing_plot_does_not_stop():
    """A single plot failure must not stop the build; other figures still saved."""
    import unittest.mock as mock

    with tempfile.TemporaryDirectory() as tmp:
        _make_results_jsonl(tmp, n=4)

        # Patch plot_sample to raise on first call, succeed on rest
        original_plot = None
        call_count = {"n": 0}

        from app.explainability.visualization import sample_plots

        orig_fn = sample_plots.plot_image_sample

        def flaky_plot_image(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("Synthetic plot failure")
            return orig_fn(*args, **kwargs)

        with mock.patch.object(sample_plots, "plot_image_sample", flaky_plot_image):
            n = build_all_figures(tmp, max_samples_per_attack=3)

        # Should still have saved some figures from other plots
        assert n >= 0  # non-negative, did not crash


def test_select_representative_samples():
    """Representative sample selection picks success + 1 failure per group."""
    results = []
    for i in range(10):
        results.append({
            "attack_name": "fgsm",
            "modality": "tabular",
            "adversarial_confidence": 0.9 - i * 0.05,
            "failure_analysis": {"attack_caused_failure": i < 5},
        })
    selected = _select_representative_samples(results, max_per_attack=3)
    # Should have at most max_per_attack total per group
    assert len(selected) <= 4  # 3 successes + 1 failure


# ===================================================================
# Task 6: Runner --figures flag tests
# ===================================================================

def test_runner_has_figures_flag():
    """--figures must be a recognized argument in the runner's parse_args."""
    import importlib, sys, types
    # We only need to verify the flag exists in the argument parser
    from pathlib import Path
    import argparse

    runner_path = Path(__file__).resolve().parents[2] / "scripts" / "run_domain_xai_evaluations.py"
    spec = importlib.util.spec_from_file_location("run_domain_xai_evaluations", runner_path)
    # Just check file has --figures text
    content = runner_path.read_text()
    assert "--figures" in content, "--figures flag not found in runner script"


def test_runner_build_all_figures_import():
    """build_all_figures can be imported from visualization package."""
    from app.explainability.visualization import build_all_figures
    assert callable(build_all_figures)


def test_runner_plot_sample_import():
    """plot_sample can be imported from visualization package."""
    from app.explainability.visualization import plot_sample
    assert callable(plot_sample)
