"""
Per-sample visualization plots for XAI explanation results.
Handles image, tabular, time-series, and text modalities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from app.explainability.visualization.common import (
    CB_COLORS,
    CMAP_MAGNITUDE,
    CMAP_SIGNED,
    load_array,
    make_title,
    save_fig,
    show_attribution_unavailable,
)

logger = logging.getLogger(__name__)

# Top-k features to show in tabular bar charts
TOP_K_TABULAR: int = 10


# =====================================================================
# Internal helpers
# =====================================================================

def _to_hwc(arr: np.ndarray) -> np.ndarray:
    """
    Convert image array from CHW to HWC if needed; supports grayscale.
    Args:
        arr: Image array of shape (C, H, W), (H, W, C), or (H, W).
    Returns:
        HWC or HW array suitable for imshow.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        return arr  # Grayscale HW

    # Batch dim – squeeze first
    while arr.ndim > 3:
        arr = arr[0]

    if arr.ndim == 3:
        c, h, w = arr.shape
        # CHW heuristic: first dim is channels when small (1 or 3)
        if c in (1, 3, 4) and h > c and w > c:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]

    return arr


def _normalize_image(arr: np.ndarray) -> np.ndarray:
    """Clip and normalize array to [0, 1] range for display."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _attribution_to_heatmap(attr: np.ndarray) -> np.ndarray:
    """
    Collapse attribution array to 2D spatial heatmap by summing over channels.
    Args:
        attr: Attribution array of shape (C, H, W), (H, W, C), or (H, W).
    Returns:
        2D heatmap array of shape (H, W).
    """
    attr_hwc = _to_hwc(attr)
    if attr_hwc.ndim == 3:
        heatmap = attr_hwc.sum(axis=-1)
    else:
        heatmap = attr_hwc
    return heatmap


def _plot_image_with_heatmap(ax_img: plt.Axes, ax_heat: plt.Axes,
                              img: Optional[np.ndarray],
                              attr: Optional[np.ndarray],
                              title: str, heatmap_note: str = "") -> None:
    """
    Render an image and its attribution heatmap side by side in two axes.
    Args:
        ax_img: Axes for the image.
        ax_heat: Axes for the attribution heatmap.
        img: Image array (None to show unavailable).
        attr: Attribution array (None shows unavailable message).
        title: Caption for ax_img.
        heatmap_note: Optional note appended to heatmap title.
    """
    # Image panel
    if img is not None:
        display = _normalize_image(_to_hwc(img))
        cmap_img = "gray" if display.ndim == 2 else None
        ax_img.imshow(display, cmap=cmap_img, aspect="auto")
    else:
        show_attribution_unavailable(ax_img, "Image unavailable")
    ax_img.set_title(title, fontsize=8)
    ax_img.axis("off")

    # Heatmap panel
    if attr is not None:
        heatmap = _attribution_to_heatmap(attr)
        im = ax_heat.imshow(heatmap, cmap=CMAP_SIGNED, aspect="auto")
        plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        note = " [Downsampled]" if heatmap_note else ""
        ax_heat.set_title(f"Attribution{note}", fontsize=8)
    else:
        show_attribution_unavailable(ax_heat, "Attribution unavailable")
        ax_heat.set_title("Attribution", fontsize=8)
    ax_heat.axis("off")


# =====================================================================
# Image plots
# =====================================================================

def plot_image_sample(result: Dict[str, Any], output_dir: Union[str, Path],
                      name: str) -> Optional[Path]:
    """
    Save a 3-column figure: clean image+heatmap | amplified perturbation |
    adversarial image+heatmap for an image-modality result.

    Args:
        result: Explanation result dict with 'attribution' and 'modality' fields.
        output_dir: Directory to save figure into (figures/ subdir).
        name: Filename stem for the saved PNG.

    Returns:
        Path to saved PNG, or None on failure.
    """
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(make_title(result), fontsize=10, fontweight="bold")

    # 3 columns: [clean img | clean attr] [perturbation] [adv img | adv attr]
    outer = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # --- Column 0: Clean image + heatmap ---
    left = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.1)
    ax_c_img = fig.add_subplot(left[0])
    ax_c_heat = fig.add_subplot(left[1])

    clean_attr = load_array(result, "clean")
    # We don't have raw pixel inputs saved, so use the downsampled postprocessed heatmap
    clean_post = None
    attr_container = result.get("attribution", {})
    if isinstance(attr_container, dict):
        clean_obj = attr_container.get("clean", {})
        if isinstance(clean_obj, dict):
            post = clean_obj.get("postprocessed", {})
            if isinstance(post, dict) and post.get("heatmap") is not None:
                clean_post = np.asarray(post["heatmap"])

    heatmap_note = ""
    if clean_attr is not None:
        hm = _attribution_to_heatmap(clean_attr)
        # Detect if postprocessed heatmap was returned (smaller dims)
        if clean_post is not None and hm.shape != np.asarray(clean_post).shape:
            heatmap_note = "Downsampled"

    _plot_image_with_heatmap(ax_c_img, ax_c_heat, None, clean_attr,
                              "Clean Image\n(pixel data not stored)", heatmap_note)

    # --- Column 1: Amplified perturbation ---
    ax_pert = fig.add_subplot(outer[1])
    adv_attr = load_array(result, "adversarial")
    if clean_attr is not None and adv_attr is not None:
        clean_hm = _attribution_to_heatmap(clean_attr)
        adv_hm = _attribution_to_heatmap(adv_attr)
        diff = adv_hm - clean_hm
        amp_factor = 10
        amplified = _normalize_image(diff * amp_factor)
        ax_pert.imshow(amplified, cmap="hot", aspect="auto")
        ax_pert.set_title(f"Perturbation\n(×{amp_factor} amplified)", fontsize=8)
    else:
        show_attribution_unavailable(ax_pert, "Perturbation unavailable")
        ax_pert.set_title("Perturbation", fontsize=8)
    ax_pert.axis("off")

    # --- Column 2: Adversarial image + heatmap ---
    right = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], wspace=0.1)
    ax_a_img = fig.add_subplot(right[0])
    ax_a_heat = fig.add_subplot(right[1])
    _plot_image_with_heatmap(ax_a_img, ax_a_heat, None, adv_attr,
                              "Adversarial Image\n(pixel data not stored)", "")

    try:
        return save_fig(fig, output_dir, name)
    except Exception as exc:
        logger.warning("Failed to save image sample plot %s: %s", name, exc)
        plt.close(fig)
        return None


# =====================================================================
# Tabular plots
# =====================================================================

def plot_tabular_sample(result: Dict[str, Any], output_dir: Union[str, Path],
                         name: str) -> Optional[Path]:
    """
    Save a tabular attribution bar chart figure: top-k clean vs adversarial
    importances and standardized feature perturbation. Marks invalid features
    from validity_flags.

    Args:
        result: Explanation result dict with tabular attribution and validity_flags.
        output_dir: Directory for saving.
        name: Filename stem.

    Returns:
        Path to saved PNG, or None on failure.
    """
    attr_container = result.get("attribution", {})
    validity_flags = result.get("validity_flags", {})
    invalid_features = set()
    if isinstance(validity_flags, dict):
        for v in validity_flags.get("bounds_violations", []):
            invalid_features.add(str(v.get("feature", "")))
        for v in validity_flags.get("integer_violations", []):
            invalid_features.add(str(v.get("feature", "")))
        for v in validity_flags.get("immutable_violations", []):
            invalid_features.add(str(v.get("feature", "")))

    def _get_top_features(subkey: str) -> Tuple[List[str], List[float]]:
        """Extract feature names and importance values from postprocessed top_features."""
        sub = attr_container.get(subkey, {}) if isinstance(attr_container, dict) else {}
        if not isinstance(sub, dict):
            return [], []
        post = sub.get("postprocessed", {}) or {}
        tops = post.get("top_features", []) or sub.get("top_features", []) or []
        names: List[str] = []
        vals: List[float] = []
        for item in tops[:TOP_K_TABULAR]:
            n = str(item.get("feature", item.get("name", "?")))
            v = float(item.get("importance", 0.0))
            names.append(n)
            vals.append(v)
        return names, vals

    clean_names, clean_vals = _get_top_features("clean")
    adv_names, adv_vals = _get_top_features("adversarial")

    comparison_target = attr_container.get("comparison_target", "") if isinstance(attr_container, dict) else ""
    title = make_title(result)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{title}\nComparison target: {comparison_target}", fontsize=9, fontweight="bold")

    for ax, feat_names, feat_vals, label in [
        (axes[0], clean_names, clean_vals, "Clean"),
        (axes[1], adv_names, adv_vals, "Adversarial"),
    ]:
        if not feat_names:
            show_attribution_unavailable(ax, f"{label}: Attribution unavailable")
            ax.set_title(label, fontsize=9)
            continue

        colors = []
        for n, v in zip(feat_names, feat_vals):
            if n in invalid_features:
                colors.append("#FF4444")  # Red for invalid
            else:
                colors.append(CB_COLORS["blue"] if v >= 0 else CB_COLORS["orange"])

        y_pos = np.arange(len(feat_names))
        bars = ax.barh(y_pos, feat_vals, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feat_names, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Attribution Score", fontsize=8)
        ax.set_title(f"{label} — Top-{TOP_K_TABULAR} Features", fontsize=9)

        # Mark invalid features with an asterisk annotation
        for i, (n, v) in enumerate(zip(feat_names, feat_vals)):
            if n in invalid_features:
                ax.text(v, i, " *INVALID", va="center", fontsize=7, color="red")

    if invalid_features:
        fig.text(0.5, 0.01, "* Red / INVALID = violates domain validity constraints",
                 ha="center", fontsize=7, color="red")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    try:
        return save_fig(fig, output_dir, name)
    except Exception as exc:
        logger.warning("Failed to save tabular sample plot %s: %s", name, exc)
        plt.close(fig)
        return None


# =====================================================================
# Time-series plots
# =====================================================================

def plot_timeseries_sample(result: Dict[str, Any], output_dir: Union[str, Path],
                            name: str) -> Optional[Path]:
    """
    Save a time-series figure: attribution heatmap (variables vs. timesteps),
    per-sensor importance bar chart, and sensor signals for clean vs adversarial.

    Args:
        result: Explanation result dict with timeseries attribution.
        output_dir: Directory for saving.
        name: Filename stem.

    Returns:
        Path to saved PNG, or None on failure.
    """
    attr_container = result.get("attribution", {})

    def _get_var_importance(subkey: str) -> Dict[str, float]:
        """Extract variable_importance dict from attribution sub-dict."""
        sub = attr_container.get(subkey, {}) if isinstance(attr_container, dict) else {}
        if not isinstance(sub, dict):
            return {}
        post = sub.get("postprocessed", {}) or {}
        vi = post.get("variable_importance") or sub.get("variable_importance") or {}
        return {str(k): float(v) for k, v in vi.items()} if isinstance(vi, dict) else {}

    clean_vi = _get_var_importance("clean")
    adv_vi = _get_var_importance("adversarial")
    clean_attr = load_array(result, "clean")
    adv_attr = load_array(result, "adversarial")
    title = make_title(result)
    comparison_target = attr_container.get("comparison_target", "") if isinstance(attr_container, dict) else ""

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"{title}\nComparison target: {comparison_target}", fontsize=9, fontweight="bold")

    # --- Top-left: Attribution heatmap (clean) ---
    ax_hm = axes[0, 0]
    if clean_attr is not None:
        arr = np.asarray(clean_attr)
        # Collapse to 2D: (T, V) by squeezing batch
        while arr.ndim > 2:
            arr = arr.squeeze(0) if arr.shape[0] == 1 else arr[0]
        im = ax_hm.imshow(arr.T if arr.ndim == 2 else arr,
                          cmap=CMAP_SIGNED, aspect="auto")
        plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
        ax_hm.set_title("Clean Attribution Heatmap\n(sensors × timesteps)", fontsize=8)
        ax_hm.set_xlabel("Timestep", fontsize=7)
        ax_hm.set_ylabel("Sensor/Variable", fontsize=7)
    else:
        show_attribution_unavailable(ax_hm, "Attribution unavailable")

    # --- Top-right: Attribution heatmap (adversarial) ---
    ax_hm_a = axes[0, 1]
    if adv_attr is not None:
        arr_a = np.asarray(adv_attr)
        while arr_a.ndim > 2:
            arr_a = arr_a.squeeze(0) if arr_a.shape[0] == 1 else arr_a[0]
        im_a = ax_hm_a.imshow(arr_a.T if arr_a.ndim == 2 else arr_a,
                               cmap=CMAP_SIGNED, aspect="auto")
        plt.colorbar(im_a, ax=ax_hm_a, fraction=0.046, pad=0.04)
        ax_hm_a.set_title("Adversarial Attribution Heatmap\n(sensors × timesteps)", fontsize=8)
        ax_hm_a.set_xlabel("Timestep", fontsize=7)
        ax_hm_a.set_ylabel("Sensor/Variable", fontsize=7)
    else:
        show_attribution_unavailable(ax_hm_a, "Attribution unavailable")

    # --- Bottom-left: Per-sensor importance (clean) ---
    ax_bar = axes[1, 0]
    if clean_vi:
        vars_list = list(clean_vi.keys())[:20]
        vals_list = [clean_vi[v] for v in vars_list]
        colors = [CB_COLORS["blue"] if v >= 0 else CB_COLORS["orange"] for v in vals_list]
        ax_bar.bar(range(len(vars_list)), vals_list, color=colors)
        ax_bar.set_xticks(range(len(vars_list)))
        ax_bar.set_xticklabels(vars_list, rotation=45, ha="right", fontsize=6)
        ax_bar.set_title("Clean: Per-Sensor Importance (top 20)", fontsize=8)
        ax_bar.set_ylabel("Attribution", fontsize=7)
    else:
        show_attribution_unavailable(ax_bar, "Sensor importance unavailable")

    # --- Bottom-right: Per-sensor importance (adversarial) ---
    ax_bar_a = axes[1, 1]
    if adv_vi:
        vars_list_a = list(adv_vi.keys())[:20]
        vals_list_a = [adv_vi[v] for v in vars_list_a]
        colors_a = [CB_COLORS["blue"] if v >= 0 else CB_COLORS["orange"] for v in vals_list_a]
        ax_bar_a.bar(range(len(vars_list_a)), vals_list_a, color=colors_a)
        ax_bar_a.set_xticks(range(len(vars_list_a)))
        ax_bar_a.set_xticklabels(vars_list_a, rotation=45, ha="right", fontsize=6)
        ax_bar_a.set_title("Adversarial: Per-Sensor Importance (top 20)", fontsize=8)
        ax_bar_a.set_ylabel("Attribution", fontsize=7)
    else:
        show_attribution_unavailable(ax_bar_a, "Sensor importance unavailable")

    plt.tight_layout()
    try:
        return save_fig(fig, output_dir, name)
    except Exception as exc:
        logger.warning("Failed to save time-series sample plot %s: %s", name, exc)
        plt.close(fig)
        return None


# =====================================================================
# Text plots
# =====================================================================

def plot_text_sample(result: Dict[str, Any], output_dir: Union[str, Path],
                      name: str) -> Optional[Path]:
    """
    Save a text-modality figure showing failure analysis and attribution status.
    Token-level attributions are unavailable in current outputs (Layer IG errors);
    renders failure mode, attack reason, and top contributing features if available.

    Args:
        result: Explanation result dict with text modality.
        output_dir: Directory for saving.
        name: Filename stem.

    Returns:
        Path to saved PNG, or None on failure.
    """
    failure_analysis = result.get("failure_analysis", {})
    attr_container = result.get("attribution", {})
    title = make_title(result)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(title, fontsize=10, fontweight="bold")
    ax.axis("off")

    # Attribution status per side
    clean_status = "N/A"
    adv_status = "N/A"
    clean_msg = ""
    adv_msg = ""
    if isinstance(attr_container, dict):
        c = attr_container.get("clean", {})
        a = attr_container.get("adversarial", {})
        if isinstance(c, dict):
            clean_status = c.get("status", "N/A")
            clean_msg = c.get("message", "")
        if isinstance(a, dict):
            adv_status = a.get("status", "N/A")
            adv_msg = a.get("message", "")

    # Build text summary
    failure_mode = failure_analysis.get("failure_mode", "N/A") if isinstance(failure_analysis, dict) else "N/A"
    attack_reason = failure_analysis.get("attack_reason", "N/A") if isinstance(failure_analysis, dict) else "N/A"
    top_feats = failure_analysis.get("contributing_features", []) if isinstance(failure_analysis, dict) else []

    lines = [
        f"Clean Attribution:      {clean_status}",
        f"Adversarial Attribution:{adv_status}",
        "",
        f"Failure Mode: {failure_mode}",
        "",
        "Attack Reason:",
        f"  {attack_reason[:120]}",
        "",
    ]

    if top_feats:
        lines.append("Top Contributing Tokens/Features:")
        for feat in top_feats[:5]:
            n = feat.get("feature", feat.get("token", "?"))
            shift = feat.get("importance_shift", feat.get("importance", 0.0))
            lines.append(f"  #{feat.get('rank', '?')} {n}  shift={shift:+.4f}")
    else:
        lines.append("Note: Token-level attributions unavailable (Layer IG error).")
        if clean_msg:
            lines.append(f"  Clean error: {clean_msg[:80]}")

    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, va="top", ha="left", transform=ax.transAxes,
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", facecolor="#F5F5F5", edgecolor="#CCCCCC"))

    try:
        return save_fig(fig, output_dir, name)
    except Exception as exc:
        logger.warning("Failed to save text sample plot %s: %s", name, exc)
        plt.close(fig)
        return None


# =====================================================================
# Dispatch: plot_sample
# =====================================================================

def plot_sample(result: Dict[str, Any], output_dir: Union[str, Path],
                name: Optional[str] = None) -> Optional[Path]:
    """
    Select and render the correct per-sample plot based on modality.
    Dispatches to plot_image_sample, plot_tabular_sample,
    plot_timeseries_sample, or plot_text_sample.

    Args:
        result: Explanation result dictionary containing 'modality' key.
        output_dir: Directory for saved figures (figures/ subdir).
        name: Optional filename stem (auto-generated from attack + modality if None).

    Returns:
        Path to saved PNG file, or None if plotting fails.
    """
    modality = str(result.get("modality", "image")).lower()
    attack = str(result.get("attack_name", "unknown"))

    if name is None:
        name = f"sample_{modality}_{attack}"

    if modality == "image":
        return plot_image_sample(result, output_dir, name)
    elif modality == "tabular":
        return plot_tabular_sample(result, output_dir, name)
    elif modality in ("timeseries", "time-series", "time_series"):
        return plot_timeseries_sample(result, output_dir, name)
    elif modality == "text":
        return plot_text_sample(result, output_dir, name)
    else:
        logger.warning("Unknown modality '%s', attempting tabular plot.", modality)
        return plot_tabular_sample(result, output_dir, name)
