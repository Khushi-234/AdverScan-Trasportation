"""
Runtime and benchmark visualization plots for XAI evaluation results.
Reads benchmark.csv for speed/faithfulness data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.explainability.visualization.common import (
    CB_COLORS,
    CMAP_MAGNITUDE,
    save_fig,
    show_attribution_unavailable,
)

logger = logging.getLogger(__name__)

RANDOM_BASELINE_AUC: float = 0.5  # Reference deletion AUC for random


# =====================================================================
# plot_run_overview
# =====================================================================

def plot_run_overview(results: List[Dict[str, Any]],
                       output_dir: str, name: str = "run_overview") -> Optional[str]:
    """
    Plot run overview: technique usage counts, fallback/error rates,
    runtime histogram, and run status summary.

    Args:
        results: List of explanation result dicts.
        output_dir: Directory for saving figure.
        name: Filename stem.

    Returns:
        Path to saved PNG.
    """
    if not results:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("XAI Run Overview", fontsize=12, fontweight="bold")
        for ax in axes.flat:
            show_attribution_unavailable(ax, "no data")
        return str(save_fig(fig, output_dir, name))

    technique_counts: Dict[str, int] = {}
    fallback_count = 0
    error_count = 0
    runtimes: List[float] = []
    status_counts: Dict[str, int] = {"success": 0, "error": 0, "fallback": 0}

    for r in results:
        tech = str(r.get("technique", "unknown"))
        technique_counts[tech] = technique_counts.get(tech, 0) + 1

        xai_dec = r.get("xai_decision", {}) or {}
        if isinstance(xai_dec, dict):
            if xai_dec.get("fallback", False):
                fallback_count += 1

        attr_status = r.get("attribution_status", "success")
        if attr_status == "error":
            error_count += 1

        meta = r.get("metadata", {}) or {}
        if isinstance(meta, dict):
            secs = meta.get("explain_seconds")
            if secs is not None:
                runtimes.append(float(secs))
        if isinstance(xai_dec, dict):
            secs_xai = xai_dec.get("explain_seconds")
            if secs_xai is not None and not runtimes:
                runtimes.append(float(secs_xai))

        if attr_status == "error":
            status_counts["error"] += 1
        elif isinstance(xai_dec, dict) and xai_dec.get("fallback", False):
            status_counts["fallback"] += 1
        else:
            status_counts["success"] += 1

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("XAI Run Overview", fontsize=12, fontweight="bold")

    # --- Top-left: Technique counts ---
    ax_tech = axes[0, 0]
    if technique_counts:
        techs = sorted(technique_counts, key=technique_counts.get, reverse=True)
        vals = [technique_counts[t] for t in techs]
        bar_colors = list(CB_COLORS.values())[:len(techs)]
        ax_tech.barh(range(len(techs)), vals, color=bar_colors[:len(techs)], alpha=0.85)
        ax_tech.set_yticks(range(len(techs)))
        ax_tech.set_yticklabels(techs, fontsize=8)
        ax_tech.set_xlabel("Count", fontsize=8)
        ax_tech.set_title("Technique Usage", fontsize=9)
    else:
        show_attribution_unavailable(ax_tech, "No technique data")

    # --- Top-right: Status pie ---
    ax_status = axes[0, 1]
    labels = [k for k, v in status_counts.items() if v > 0]
    vals_s = [v for v in status_counts.values() if v > 0]
    pie_colors = [CB_COLORS["green"], CB_COLORS["orange"], CB_COLORS["blue"]]
    if vals_s:
        ax_status.pie(vals_s, labels=labels, colors=pie_colors[:len(labels)],
                       autopct="%1.0f%%", startangle=90, textprops={"fontsize": 8})
        ax_status.set_title("Run Status Distribution", fontsize=9)
    else:
        show_attribution_unavailable(ax_status, "No status data")

    # --- Bottom-left: Fallback + error bar ---
    ax_fb = axes[1, 0]
    n = len(results)
    fb_rate = fallback_count / n if n > 0 else 0
    err_rate = error_count / n if n > 0 else 0
    ax_fb.bar(["Fallback Rate", "Error Rate"], [fb_rate, err_rate],
               color=[CB_COLORS["yellow"], CB_COLORS["orange"]], alpha=0.85)
    ax_fb.set_ylabel("Rate", fontsize=8)
    ax_fb.set_ylim(0, 1.1)
    ax_fb.set_title(f"Fallback & Error Rates (n={n})", fontsize=9)
    for xi, val in enumerate([fb_rate, err_rate]):
        ax_fb.text(xi, val + 0.02, f"{val:.1%}", ha="center", fontsize=8)

    # --- Bottom-right: Runtime histogram ---
    ax_hist = axes[1, 1]
    if runtimes:
        ax_hist.hist(runtimes, bins=min(20, len(runtimes)), color=CB_COLORS["sky_blue"],
                      edgecolor="white", alpha=0.85)
        ax_hist.set_xlabel("Explanation Time (s)", fontsize=8)
        ax_hist.set_ylabel("Count", fontsize=8)
        ax_hist.set_title("Runtime Distribution", fontsize=9)
        med = np.median(runtimes)
        ax_hist.axvline(med, color=CB_COLORS["orange"], linestyle="--",
                         linewidth=1.2, label=f"Median {med:.3f}s")
        ax_hist.legend(fontsize=7)
    else:
        show_attribution_unavailable(ax_hist, "No runtime data")

    plt.tight_layout()
    return str(save_fig(fig, output_dir, name))


# =====================================================================
# plot_speed_vs_faithfulness
# =====================================================================

def plot_speed_vs_faithfulness(benchmark_csv: str,
                                output_dir: str,
                                name: str = "speed_vs_faithfulness") -> Optional[str]:
    """
    Scatter plot of seconds-per-sample vs. deletion AUC per technique.
    X-axis is log-scale. Annotates each technique. Shows random baseline.
    Returns None (and logs) if benchmark_csv does not exist.

    Args:
        benchmark_csv: Path to benchmark.csv with columns:
            technique, seconds_per_sample, deletion_auc.
        output_dir: Directory for saving figure.
        name: Filename stem.

    Returns:
        Path to saved PNG, or None if file missing.
    """
    csv_path = Path(benchmark_csv)
    if not csv_path.exists():
        logger.warning("plot_speed_vs_faithfulness: benchmark file not found at %s", benchmark_csv)
        return None

    try:
        import csv
        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as exc:
        logger.warning("Failed to read benchmark file %s: %s", benchmark_csv, exc)
        return None

    if not rows:
        logger.warning("plot_speed_vs_faithfulness: benchmark file is empty.")
        return None

    techniques = []
    speeds = []
    aucs = []
    for row in rows:
        try:
            t = str(row.get("technique", ""))
            s = float(row.get("seconds_per_sample", 0))
            a = float(row.get("deletion_auc", 0))
            if t and s > 0:
                techniques.append(t)
                speeds.append(s)
                aucs.append(a)
        except (ValueError, TypeError):
            continue

    if not techniques:
        logger.warning("plot_speed_vs_faithfulness: no valid rows in benchmark file.")
        return None

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle("Speed vs Faithfulness per Technique", fontsize=11, fontweight="bold")

    colors = list(CB_COLORS.values())
    for i, (t, s, a) in enumerate(zip(techniques, speeds, aucs)):
        ax.scatter(s, a, color=colors[i % len(colors)], s=100, zorder=5,
                    label=t, edgecolors="gray", linewidths=0.5)
        ax.annotate(t, (s, a), fontsize=7, xytext=(5, 3), textcoords="offset points")

    # Random baseline
    ax.axhline(RANDOM_BASELINE_AUC, color="gray", linestyle="--",
                linewidth=1.0, label="Random baseline")

    ax.set_xscale("log")
    ax.set_xlabel("Seconds per Sample (log scale)", fontsize=9)
    ax.set_ylabel("Deletion AUC (higher = more faithful)", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    return str(save_fig(fig, output_dir, name))


# =====================================================================
# plot_deletion_curves
# =====================================================================

def plot_deletion_curves(benchmark_csv: str,
                          output_dir: str,
                          name: str = "deletion_curves") -> Optional[str]:
    """
    Plot deletion curves from benchmark_csv if curve columns are available.
    Logs an info message and shows AUC annotation if only scalar AUC exists.
    Returns None if benchmark file is missing.

    Args:
        benchmark_csv: Path to benchmark.csv. Must exist. Expected columns:
            technique, deletion_auc, and optionally deletion_curve (JSON list).
        output_dir: Directory for saving figure.
        name: Filename stem.

    Returns:
        Path to saved PNG, or None if file missing.
    """
    csv_path = Path(benchmark_csv)
    if not csv_path.exists():
        logger.warning("plot_deletion_curves: benchmark file not found at %s", benchmark_csv)
        return None

    try:
        import csv
        import json as _json
        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as exc:
        logger.warning("Failed to read benchmark file %s: %s", benchmark_csv, exc)
        return None

    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle("Deletion Curves per Technique", fontsize=11, fontweight="bold")

    has_curves = False
    colors = list(CB_COLORS.values())

    for i, row in enumerate(rows):
        t = str(row.get("technique", f"technique_{i}"))
        auc = row.get("deletion_auc")
        curve_raw = row.get("deletion_curve")

        color = colors[i % len(colors)]

        if curve_raw:
            try:
                curve = _json.loads(curve_raw)
                xs = np.linspace(0, 1, len(curve))
                ax.plot(xs, curve, label=f"{t} (AUC={auc})", color=color, linewidth=1.5)
                has_curves = True
            except Exception:
                pass

        if not has_curves and auc is not None:
            try:
                auc_val = float(auc)
                ax.annotate(f"{t}: AUC={auc_val:.3f}",
                             xy=(0.5, 0.5 - 0.07 * i), xycoords="axes fraction",
                             fontsize=9, ha="center", color=color)
            except (ValueError, TypeError):
                pass

    if not has_curves:
        logger.info("plot_deletion_curves: only AUC scalar values found, not full curves.")
        show_attribution_unavailable(ax, "Deletion curves not stored; only AUC values available.\n"
                                         "See annotations above.")

    ax.axhline(RANDOM_BASELINE_AUC, color="gray", linestyle="--",
                linewidth=0.8, label="Random baseline")
    ax.set_xlabel("Feature Fraction Deleted", fontsize=9)
    ax.set_ylabel("Model Confidence / Score", fontsize=9)
    ax.set_xlim(0, 1)
    if has_curves:
        ax.legend(fontsize=7)
    plt.tight_layout()
    return str(save_fig(fig, output_dir, name))
