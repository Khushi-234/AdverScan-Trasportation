"""
Tests for per-sample visualization plots (Task 2).
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from app.explainability.visualization.sample_plots import (
    plot_sample,
    plot_image_sample,
    plot_tabular_sample,
    plot_timeseries_sample,
    plot_text_sample,
)


def _base_result(modality: str, attack: str = "fgsm") -> dict:
    """Return a minimal valid explanation result dict."""
    return {
        "attack_name": attack,
        "technique": "integrated_gradients",
        "modality": modality,
        "clean_prediction": 0,
        "adversarial_prediction": 1,
        "clean_confidence": 0.9,
        "adversarial_confidence": 0.4,
        "prediction_changed": True,
        "attribution": {},
        "comparison": {},
        "failure_analysis": {
            "attack_caused_failure": True,
            "failure_mode": "clean_correct_to_adversarial_incorrect",
            "contributing_features": [],
        },
        "validity_flags": {},
        "attribution_status": "success",
        "xai_decision": {"fallback": False, "explain_seconds": 0.01},
        "metadata": {},
    }


# ===================================================================
# Image
# ===================================================================

def test_image_plot_saved():
    """Image plot should save a PNG even with no attribution arrays stored."""
    r = _base_result("image")
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_image_sample(r, tmp, "test_image")
        assert path is not None
        assert Path(path).exists()


def test_image_plot_with_attr_in_npz():
    """Image plot should load attribution from companion .npz and save heatmap."""
    import numpy as np
    r = _base_result("image")
    with tempfile.TemporaryDirectory() as tmp:
        npz_path = Path(tmp) / "fake.npz"
        arr = np.random.randn(3, 32, 32).astype(np.float32)
        np.savez_compressed(npz_path,
                             clean_attribution=arr,
                             adversarial_attribution=arr * 0.9)
        r["metadata"] = {"npz_arrays_path": str(npz_path)}
        r["attribution"] = {
            "clean": {"attribution": "[Stored in .npz: clean_attribution]"},
            "adversarial": {"attribution": "[Stored in .npz: adversarial_attribution]"},
        }
        path = plot_image_sample(r, tmp, "img_with_npz")
        assert path is not None
        assert Path(path).exists()


# ===================================================================
# Tabular
# ===================================================================

def _tabular_result(n_features: int = 6, attack_success: bool = True) -> dict:
    """Return a synthetic tabular explanation result."""
    r = _base_result("tabular", "cap_pgd")
    tops = [
        {"feature": f"f{i}", "importance": round(0.1 * (n_features - i), 4), "rank": i + 1, "index": i}
        for i in range(n_features)
    ]
    r["attribution"] = {
        "comparison_target": "clean_pred",
        "clean": {
            "status": "success",
            "attribution": [[0.1 * i for i in range(n_features)]],
            "postprocessed": {"top_features": tops, "feature_names": [f"f{i}" for i in range(n_features)]},
        },
        "adversarial": {
            "status": "success",
            "attribution": [[0.12 * i for i in range(n_features)]],
            "postprocessed": {
                "top_features": [
                    {"feature": f"f{i}", "importance": round(0.12 * (n_features - i), 4), "rank": i + 1}
                    for i in range(n_features)
                ]
            },
        },
    }
    r["validity_flags"] = {
        "is_valid": True,
        "bounds_violations": [{"feature": "f0"}],
        "integer_violations": [],
        "immutable_violations": [],
    }
    r["failure_analysis"]["attack_caused_failure"] = attack_success
    return r


def test_tabular_plot_saved():
    r = _tabular_result()
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_tabular_sample(r, tmp, "test_tabular")
        assert path is not None
        assert Path(path).exists()


def test_tabular_plot_unavailable_attribution():
    """Tabular plot should show 'unavailable' message when attribution missing."""
    r = _base_result("tabular", "cap_pgd")
    r["attribution"] = {"clean": {"status": "error"}, "adversarial": {"status": "error"}}
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_tabular_sample(r, tmp, "tabular_unavailable")
        assert path is not None


# ===================================================================
# Time-series
# ===================================================================

def _ts_result() -> dict:
    """Return a synthetic time-series explanation result."""
    r = _base_result("timeseries", "time_series_pgd")
    vi = {f"variable_{i}": float(abs(np.random.randn())) for i in range(20)}
    attr = np.random.randn(1, 1, 20).astype(np.float32).tolist()
    r["attribution"] = {
        "comparison_target": "clean_pred",
        "clean": {
            "status": "success",
            "attribution": attr,
            "postprocessed": {
                "attribution": attr,
                "variable_importance": vi,
            },
        },
        "adversarial": {
            "status": "success",
            "attribution": attr,
            "postprocessed": {
                "attribution": attr,
                "variable_importance": {k: v * 1.05 for k, v in vi.items()},
            },
        },
    }
    return r


def test_timeseries_plot_saved():
    r = _ts_result()
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_timeseries_sample(r, tmp, "test_ts")
        assert path is not None
        assert Path(path).exists()


def test_timeseries_plot_unavailable():
    r = _base_result("timeseries")
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_timeseries_sample(r, tmp, "ts_unavailable")
        assert path is not None  # Should still save with 'unavailable' boxes


# ===================================================================
# Text
# ===================================================================

def test_text_plot_saved():
    """Text plot should save figure even with attribution errors (Layer IG failures)."""
    r = _base_result("text", "hotflip")
    r["attribution"] = {
        "clean": {
            "status": "error",
            "message": "Layer Integrated Gradients computation error: too many dimensions 'str'",
        },
        "adversarial": {
            "status": "error",
            "message": "same error",
        },
    }
    r["failure_analysis"] = {
        "attack_caused_failure": False,
        "failure_mode": "clean_incorrect_to_adversarial_incorrect",
        "contributing_features": [],
        "attack_reason": "Attack failed to flip prediction.",
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_text_sample(r, tmp, "test_text")
        assert path is not None
        assert Path(path).exists()


# ===================================================================
# Dispatch: plot_sample
# ===================================================================

def test_plot_sample_dispatch_image():
    r = _base_result("image")
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_sample(r, tmp)
        assert path is not None

def test_plot_sample_dispatch_tabular():
    r = _tabular_result()
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_sample(r, tmp)
        assert path is not None

def test_plot_sample_dispatch_timeseries():
    r = _ts_result()
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_sample(r, tmp)
        assert path is not None

def test_plot_sample_dispatch_text():
    r = _base_result("text")
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_sample(r, tmp)
        assert path is not None

def test_plot_sample_unknown_modality():
    """Unknown modality falls back to tabular plot without raising."""
    r = _base_result("custom_mod")
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_sample(r, tmp)
        assert path is not None
