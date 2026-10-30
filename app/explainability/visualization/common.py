"""
Shared helpers, constants, and data loading utilities for XAI visualizations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib

# Set non-interactive Agg backend before importing pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Colorblind-safe palette (Okabe-Ito / Wong 2011)
CB_COLORS: Dict[str, str] = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "yellow": "#F0E442",
    "sky_blue": "#56B4E9",
    "vermilion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "gray": "#7F7F7F",
}

# Colormaps
CMAP_SIGNED: str = "coolwarm"       # Diverging colormap for signed attributions (- to +)
CMAP_MAGNITUDE: str = "viridis"     # Sequential colormap for absolute magnitude attributions


class CheckpointStore:
    """
    Storage loader for XAI explanation checkpoint records.
    Reads xai_results.jsonl from an evaluation output directory.
    """

    @classmethod
    def load_all(cls, output_dir: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load all serialized explanation records from xai_results.jsonl in output_dir.
        Falls back to individual *_xai_explanation.json files if .jsonl is absent.

        Args:
            output_dir: Directory containing evaluation result files.

        Returns:
            List of parsed explanation result dictionaries.
        """
        out_path = Path(output_dir)
        records: List[Dict[str, Any]] = []

        jsonl_path = out_path / "xai_results.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records

        # Fallback to individual explanation JSONs if jsonl not present
        for p in sorted(out_path.glob("*_xai_explanation.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    records.append(json.load(f))
            except Exception:
                continue

        return records


def load_results(output_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load explanation results using CheckpointStore.load_all and filter out
    records with attribution_status='error'.

    Args:
        output_dir: Directory containing xai_results.jsonl or explanation files.

    Returns:
        List of valid explanation result dictionaries (excluding error status).
    """
    raw_records = CheckpointStore.load_all(output_dir)
    valid_records: List[Dict[str, Any]] = []

    for r in raw_records:
        status = getattr(r, "attribution_status", None)
        if status is None and isinstance(r, dict):
            status = r.get("attribution_status", "success")
        if status != "error":
            valid_records.append(r if isinstance(r, dict) else r.to_dict())

    return valid_records


def load_array(result: Any, key: str) -> Optional[np.ndarray]:
    """
    Extract an attribution or input array from an explanation result.
    Loads from companion .npz files if referenced or directly from JSON lists.
    Returns None if the array is unavailable or corrupted.

    Args:
        result: ExplanationResult object or result dictionary.
        key: Attribution identifier, e.g. 'clean', 'adversarial',
             'clean_attribution', 'adversarial_attribution', or 'clean_input'.

    Returns:
        np.ndarray if available, or None. Expected shapes vary by modality:
        - Image: (C, H, W) or (H, W, C) or (H, W)
        - Tabular: (F,) or (1, F)
        - Time-series: (T, C) or (C, T) or (T,)
    """
    if result is None:
        return None

    res_dict: Dict[str, Any] = result.to_dict() if hasattr(result, "to_dict") else result
    if not isinstance(res_dict, dict):
        return None

    # Determine potential companion .npz path
    npz_path: Optional[Path] = None
    meta = res_dict.get("metadata", {})
    if isinstance(meta, dict) and meta.get("npz_arrays_path"):
        cand = Path(meta["npz_arrays_path"])
        if cand.exists():
            npz_path = cand

    if npz_path is None and "_file_path" in res_dict:
        cand = Path(res_dict["_file_path"]).with_suffix(".npz")
        if cand.exists():
            npz_path = cand

    # Candidate keys to search in .npz
    candidate_keys = [key]
    if not key.endswith("_attribution"):
        candidate_keys.append(f"{key}_attribution")
    if key.endswith("_attribution"):
        candidate_keys.append(key[: -len("_attribution")])

    # 1. Attempt loading from companion .npz
    if npz_path is not None and npz_path.exists():
        try:
            with np.load(npz_path) as npz_data:
                for k in candidate_keys:
                    if k in npz_data:
                        return np.asarray(npz_data[k])
        except Exception:
            pass

    # 2. Check within nested attribution dict
    attr_container = res_dict.get("attribution", {})
    if isinstance(attr_container, dict):
        # Look for sub-dictionary under clean / adversarial
        for subkey in (key, key.replace("_attribution", "")):
            sub_obj = attr_container.get(subkey)
            if isinstance(sub_obj, dict):
                # If execution failed explicitly
                if sub_obj.get("status") == "error":
                    return None

                raw_val = sub_obj.get("attribution")
                if isinstance(raw_val, str) and "[Stored in .npz:" in raw_val:
                    # String marker: extract name and retry NPZ if possible
                    import re
                    match = re.search(r"\[Stored in \.npz:\s*([a-zA-Z0-9_]+)\]", raw_val)
                    if match and npz_path is not None and npz_path.exists():
                        try:
                            with np.load(npz_path) as npz_data:
                                marker_key = match.group(1)
                                if marker_key in npz_data:
                                    return np.asarray(npz_data[marker_key])
                        except Exception:
                            pass
                    # If NPZ not found or missing, check downsampled heatmap
                    hm = sub_obj.get("postprocessed", {}).get("heatmap")
                    if hm is not None:
                        return np.asarray(hm)
                    return None

                if raw_val is not None:
                    try:
                        arr = np.asarray(raw_val)
                        if arr.size > 0:
                            return arr
                    except Exception:
                        pass

                # Check postprocessed attribution or heatmap
                post = sub_obj.get("postprocessed", {})
                if isinstance(post, dict):
                    for post_key in ("attribution", "heatmap"):
                        if post.get(post_key) is not None:
                            try:
                                return np.asarray(post[post_key])
                            except Exception:
                                pass

        # Look directly for key in attribution container
        if key in attr_container:
            val = attr_container[key]
            if isinstance(val, (list, np.ndarray)):
                arr = np.asarray(val)
                if arr.size > 0:
                    return arr

    # 3. Check directly in root result dictionary
    if key in res_dict:
        val = res_dict[key]
        if isinstance(val, (list, np.ndarray)):
            arr = np.asarray(val)
            if arr.size > 0:
                return arr

    return None


def save_fig(fig: plt.Figure, output_dir: Union[str, Path], name: str) -> Path:
    """
    Save Matplotlib figure as a PNG at 150 DPI in the 'figures/' subdirectory
    under output_dir, and always close the figure.

    Args:
        fig: Matplotlib Figure instance.
        output_dir: Base directory path.
        name: Filename (e.g. 'sample_01.png' or 'sample_01').

    Returns:
        Path to the saved PNG figure.
    """
    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    filename = name if name.lower().endswith(".png") else f"{name}.png"
    target_path = fig_dir / filename

    try:
        fig.savefig(target_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)

    return target_path


def make_title(result: Any) -> str:
    """
    Build a standardized title string incorporating attack, technique,
    modality, and fallback status.

    Args:
        result: ExplanationResult instance or result dictionary.

    Returns:
        Formatted title string.
    """
    res_dict: Dict[str, Any] = result.to_dict() if hasattr(result, "to_dict") else result
    if not isinstance(res_dict, dict):
        return "XAI Explanation"

    atk = res_dict.get("attack_name", "unknown").upper()
    tech = res_dict.get("technique", "default")
    mod = res_dict.get("modality", "image").capitalize()

    xai_dec = res_dict.get("xai_decision", {})
    fallback = xai_dec.get("fallback", False) if isinstance(xai_dec, dict) else False
    fallback_str = " (Fallback)" if fallback else ""

    return f"[{mod}] Attack: {atk} | Technique: {tech}{fallback_str}"


def show_attribution_unavailable(
    ax: plt.Axes, message: str = "Attribution unavailable"
) -> None:
    """
    Render an explicit 'attribution unavailable' message in the axes
    to prevent blank or misleading plots.

    Args:
        ax: Matplotlib axes to render message in.
        message: Explanatory text to display.
    """
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=11,
        color="#555555",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#EEEEEE", edgecolor="#CCCCCC"),
    )
    ax.set_xticks([])
    ax.set_yticks([])
