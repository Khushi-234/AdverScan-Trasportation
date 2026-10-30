"""
Tests for shared visualization helpers in app.explainability.visualization.common.
"""

from pathlib import Path
import json
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import pytest

from app.explainability.explanation_result import ExplanationResult
from app.explainability.visualization.common import (
    CB_COLORS,
    CMAP_MAGNITUDE,
    CMAP_SIGNED,
    CheckpointStore,
    load_array,
    load_results,
    make_title,
    save_fig,
    show_attribution_unavailable,
)


def test_load_results_and_error_filtering():
    """Test loading explanation results from xai_results.jsonl and filtering out error status."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        jsonl_path = tmp_path / "xai_results.jsonl"

        records = [
            {"attack_name": "fgsm", "technique": "saliency", "attribution_status": "success"},
            {"attack_name": "pgd", "technique": "integrated_gradients", "attribution_status": "error"},
            {"attack_name": "cw", "technique": "deep_lift", "attribution_status": "success"},
        ]

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        # Test CheckpointStore.load_all
        all_records = CheckpointStore.load_all(tmp_path)
        assert len(all_records) == 3

        # Test load_results filters error record
        filtered = load_results(tmp_path)
        assert len(filtered) == 2
        assert [r["attack_name"] for r in filtered] == ["fgsm", "cw"]


def test_load_array_from_json_and_missing():
    """Test loading array from JSON list, handling missing or corrupted keys."""
    res = {
        "attack_name": "pgd",
        "attribution": {
            "clean": {
                "status": "success",
                "attribution": [[0.1, -0.2], [0.3, 0.4]],
            },
            "adversarial": {
                "status": "error",
                "message": "failed",
            },
        },
    }

    # Clean attribution present
    arr_clean = load_array(res, "clean")
    assert arr_clean is not None
    assert np.allclose(arr_clean, [[0.1, -0.2], [0.3, 0.4]])

    # Adversarial attribution error status returns None
    arr_adv = load_array(res, "adversarial")
    assert arr_adv is None

    # Missing key returns None
    arr_missing = load_array(res, "non_existent_key")
    assert arr_missing is None

    # None result returns None
    assert load_array(None, "clean") is None


def test_load_array_from_npz_companion():
    """Test loading array offloaded to a companion .npz file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        npz_path = tmp_path / "sample.npz"

        clean_data = np.ones((4, 4), dtype=np.float32) * 2.5
        adv_data = np.zeros((4, 4), dtype=np.float32)
        np.savez_compressed(npz_path, clean_attribution=clean_data, adversarial_attribution=adv_data)

        res = {
            "attack_name": "pgd",
            "metadata": {"npz_arrays_path": str(npz_path)},
            "attribution": {
                "clean": {"attribution": "[Stored in .npz: clean_attribution]"},
                "adversarial": {"attribution": "[Stored in .npz: adversarial_attribution]"},
            },
        }

        loaded_clean = load_array(res, "clean")
        assert loaded_clean is not None
        assert np.allclose(loaded_clean, clean_data)

        loaded_adv = load_array(res, "adversarial_attribution")
        assert loaded_adv is not None
        assert np.allclose(loaded_adv, adv_data)


def test_save_fig_and_always_close():
    """Test save_fig saves 150 DPI PNG in figures/ directory and always closes figure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])

        out_path = save_fig(fig, tmpdir, "test_plot")
        assert out_path.exists()
        assert out_path.name == "test_plot.png"
        assert out_path.parent.name == "figures"

        # Check figure is closed
        assert not plt.fignum_exists(fig.number)


def test_make_title_and_show_unavailable():
    """Test make_title formatting and show_attribution_unavailable axes helper."""
    res = {
        "attack_name": "fgsm",
        "technique": "saliency",
        "modality": "image",
        "xai_decision": {"fallback": True},
    }
    title = make_title(res)
    assert "[Image]" in title
    assert "FGSM" in title
    assert "saliency" in title
    assert "Fallback" in title

    # Test show_attribution_unavailable
    fig, ax = plt.subplots()
    show_attribution_unavailable(ax, "No heatmap computed")
    assert len(ax.texts) > 0
    assert "No heatmap computed" in ax.texts[0].get_text()
    plt.close(fig)
