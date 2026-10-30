"""
Dataset-level summary visualizations for XAI results.
Includes attack success rates, failure modes, feature exploitation,
attribution drift, and confidence shift plots.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.explainability.visualization.common import (
    CB_COLORS,
    CMAP_MAGNITUDE,
    CMAP_SIGNED,
    save_fig,
    show_attribution_unavailable,
)

logger = logging.getLogger(__name__)

# Colorblind-safe palette for attack labels
_ATTACK_COLORS = [
    CB_COLORS["blue"], CB_COLORS["orange"], CB_COLORS["green"],
    CB_COLORS["sky_blue"], CB_COLORS["purple"], CB_COLORS["yellow"],
]


def _attack_color(attack: str, attack_list: List[str]) -> str:
    """Return a consistent colorblind-safe color for an attack label."""
    try:
        idx = attack_list.index(attack) % len(_ATTACK_COLORS)
    except ValueError:
        idx = 0
    return _ATTACK_COLORS[idx]


def _wilson_ci(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Compute 95% Wilson score confidence interval for a proportion.
    Args:
        successes: Number of successes.
        trials: Total trial count.
        z: Z-score for confidence level (1.96 for 95%).
    Returns:
        Tuple (lower, upper) confidence interval bounds.
    """
    if trials == 0:
        return 0.0, 0.0
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _no_data(ax: plt.Axes, msg: str = "no data") -> None:
    """Display a 'no data' message centered in the axes."""
    show_attribution_unavailable(ax, msg)


# =====================================================================
# plot_attack_summary
# =====================================================================

def plot_attack_summary(results: List[Dict[str, Any]],
                         output_dir: str, name: str = "attack_summary") -> Optional[str]:
    """
    Plot attack success rate with 95% Wilson CI per attack.
    Denominator is clean-correct samples; annotates successes/denominator.

    Args:
        results: List of explanation result dicts.
        output_dir: Directory to save figure.
        name: Filename stem.

    Returns:
        Path to saved PNG, or None on empty input.
    """
    if not results:
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.suptitle("Attack Success Rate", fontsize=11, fontweight="bold")
        _no_data(ax)
        return str(save_fig(fig, output_dir, name))

    # Aggregate by attack
    from collections import defaultdict
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"successes": 0, "clean_correct": 0})
    for r in results:
        atk = str(r.get("attack_name", "unknown"))
        fa = r.get("failure_analysis", {}) or {}
        clean_correct = bool(fa.get("clean_correct", False))
        attack_caused = bool(fa.get("attack_caused_failure", False))
        if clean_correct:
            counts[atk]["clean_correct"] += 1
            if attack_caused:
                counts[atk]["successes"] += 1

    if not counts:
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.suptitle("Attack Success Rate", fontsize=11, fontweight="bold")
        _no_data(ax, "No clean-correct samples found")
        return str(save_fig(fig, output_dir, name))

    attacks = sorted(counts.keys())
    attack_colors = {a: _attack_color(a, attacks) for a in attacks}
    asrs = [counts[a]["successes"] / counts[a]["clean_correct"]
            if counts[a]["clean_correct"] > 0 else 0.0 for a in attacks]
    ci_low = []
    ci_high = []
    for a in attacks:
        lo, hi = _wilson_ci(counts[a]["successes"], counts[a]["clean_correct"])
        ci_low.append(asrs[attacks.index(a)] - lo)
        ci_high.append(hi - asrs[attacks.index(a)])

    fig, ax = plt.subplots(figsize=(max(7, len(attacks) * 1.4), 5))
    fig.suptitle("Attack Success Rate (clean-correct denominator)", fontsize=11, fontweight="bold")

    x = np.arange(len(attacks))
    bars = ax.bar(x, asrs, color=[attack_colors[a] for a in attacks],
                  yerr=[ci_low, ci_high], capsize=5, alpha=0.85, ecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Attack Success Rate", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="50%")
    ax.legend(fontsize=8)

    for i, (a, asr) in enumerate(zip(attacks, asrs)):
        s = counts[a]["successes"]
        d = counts[a]["clean_correct"]
        ax.text(i, asr + 0.02, f"{s}/{d}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    return str(save_fig(fig, output_dir, name))


# =====================================================================
# plot_failure_modes
# =====================================================================

def plot_failure_modes(results: List[Dict[str, Any]],
                        output_dir: str, name: str = "failure_modes") -> Optional[str]:
    """
    Plot failure-mode counts per attack as a grouped bar chart.

    Args:
        results: List of explanation result dicts.
        output_dir: Directory to save figure.
        name: Filename stem.

    Returns:
        Path to saved PNG.
    """
    if not results:
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.suptitle("Failure Modes per Attack", fontsize=11, fontweight="bold")
        _no_data(ax)
        return str(save_fig(fig, output_dir, name))

    from collections import defaultdict
    mode_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    all_modes: set = set()

    for r in results:
        atk = str(r.get("attack_name", "unknown"))
        fa = r.get("failure_analysis", {}) or {}
        mode = str(fa.get("failure_mode", "unknown"))
        mode_counts[atk][mode] += 1
        all_modes.add(mode)

    attacks = sorted(mode_counts.keys())
    modes = sorted(all_modes)

    if not attacks or not modes:
        fig, ax = plt.subplots(figsize=(8, 4))
        _no_data(ax, "No failure mode data")
        return str(save_fig(fig, output_dir, name))

    x = np.arange(len(attacks))
    width = 0.8 / max(len(modes), 1)
    mode_colors = plt.cm.Set2(np.linspace(0, 1, len(modes)))

    fig, ax = plt.subplots(figsize=(max(8, len(attacks) * 1.5), 5))
    fig.suptitle("Failure Modes per Attack", fontsize=11, fontweight="bold")

    for mi, mode in enumerate(modes):
        vals = [mode_counts[a][mode] for a in attacks]
        ax.bar(x + mi * width, vals, width=width * 0.9,
               label=mode.replace("_", " "), color=mode_colors[mi], alpha=0.85)

    ax.set_xticks(x + width * (len(modes) - 1) / 2)
    ax.set_xticklabels(attacks, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    return str(save_fig(fig, output_dir, name))


# =====================================================================
# plot_exploited_features
# =====================================================================

def plot_exploited_features(results: List[Dict[str, Any]],
                              output_dir: str, name: str = "exploited_features",
                              top_k: int = 10) -> Optional[str]:
    """
    Plot top-k features in successful vs. failed attacks.
    Time-series results are aggregated by sensor name.
    Image comparisons are skipped with a log message.

    Args:
        results: List of explanation result dicts.
        output_dir: Directory to save figure.
        name: Filename stem.
        top_k: Number of top features to display.

    Returns:
        Path to saved PNG.
    """
    if not results:
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.suptitle("Exploited Features: Success vs Failure", fontsize=11, fontweight="bold")
        _no_data(ax)
        return str(save_fig(fig, output_dir, name))

    success_scores: Dict[str, float] = defaultdict(float)
    success_counts: Dict[str, int] = defaultdict(int)
    failure_scores: Dict[str, float] = defaultdict(float)
    failure_counts: Dict[str, int] = defaultdict(int)

    for r in results:
        modality = str(r.get("modality", "tabular")).lower()
        if modality in ("image",):
            logger.info("plot_exploited_features: skipping image result (unaligned spatial comparison).")
            continue

        fa = r.get("failure_analysis", {}) or {}
        is_success = bool(fa.get("attack_caused_failure", False))
        contributing = fa.get("contributing_features", []) or []

        for feat in contributing:
            feature_name = str(feat.get("feature", feat.get("sensor", "?")))
            # Aggregate time-series by sensor name
            if "sensor" in feat:
                feature_name = str(feat.get("sensor", feature_name))
            importance = abs(float(feat.get("importance_shift", feat.get("importance", 0.0))))

            if is_success:
                success_scores[feature_name] += importance
                success_counts[feature_name] += 1
            else:
                failure_scores[feature_name] += importance
                failure_counts[feature_name] += 1

    # Average by count
    avg_success = {k: v / success_counts[k] for k, v in success_scores.items()}
    avg_failure = {k: v / failure_counts[k] for k, v in failure_scores.items()}

    all_feats = sorted(
        set(avg_success.keys()) | set(avg_failure.keys()),
        key=lambda f: avg_success.get(f, 0) + avg_failure.get(f, 0),
        reverse=True,
    )[:top_k]

    if not all_feats:
        fig, ax = plt.subplots(figsize=(8, 4))
        _no_data(ax, "No feature exploitation data")
        return str(save_fig(fig, output_dir, name))

    y = np.arange(len(all_feats))
    width = 0.35
    s_vals = [avg_success.get(f, 0.0) for f in all_feats]
    f_vals = [avg_failure.get(f, 0.0) for f in all_feats]

    fig, ax = plt.subplots(figsize=(10, max(5, len(all_feats) * 0.45)))
    fig.suptitle(f"Top-{top_k} Exploited Features: Success vs Failed Attacks",
                 fontsize=11, fontweight="bold")

    ax.barh(y - width / 2, s_vals, width, label="Successful attacks",
            color=CB_COLORS["orange"], alpha=0.85)
    ax.barh(y + width / 2, f_vals, width, label="Failed attacks",
            color=CB_COLORS["blue"], alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(all_feats, fontsize=8)
    ax.set_xlabel("Avg Absolute Attribution / Shift", fontsize=9)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return str(save_fig(fig, output_dir, name))


# =====================================================================
# plot_attribution_drift
# =====================================================================

def plot_attribution_drift(results: List[Dict[str, Any]],
                            output_dir: str, name: str = "attribution_drift") -> Optional[str]:
    """
    Plot normalized L1 drift and cosine similarity by attack, from comparison field.

    Args:
        results: List of explanation result dicts.
        output_dir: Directory to save figure.
        name: Filename stem.

    Returns:
        Path to saved PNG.
    """
    if not results:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle("Attribution Drift by Attack", fontsize=11, fontweight="bold")
        for ax in axes:
            _no_data(ax)
        return str(save_fig(fig, output_dir, name))

    l1_by_attack: Dict[str, List[float]] = defaultdict(list)
    cos_by_attack: Dict[str, List[float]] = defaultdict(list)

    for r in results:
        atk = str(r.get("attack_name", "unknown"))
        cmp = r.get("comparison", {}) or {}
        if not isinstance(cmp, dict):
            continue
        l1 = cmp.get("normalized_attribution_l1")
        cos = cmp.get("attribution_cosine_similarity")
        if l1 is not None:
            l1_by_attack[atk].append(float(l1))
        if cos is not None:
            cos_by_attack[atk].append(float(cos))

    attacks = sorted(set(l1_by_attack.keys()) | set(cos_by_attack.keys()))

    if not attacks:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax in axes:
            _no_data(ax, "No comparison drift data")
        return str(save_fig(fig, output_dir, name))

    attack_colors = {a: _attack_color(a, attacks) for a in attacks}

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(attacks) * 2), 5))
    fig.suptitle("Attribution Drift by Attack", fontsize=11, fontweight="bold")

    # L1 drift
    ax_l1 = axes[0]
    l1_means = [np.mean(l1_by_attack[a]) if a in l1_by_attack else 0.0 for a in attacks]
    l1_stds = [np.std(l1_by_attack[a]) if a in l1_by_attack else 0.0 for a in attacks]
    ax_l1.bar(attacks, l1_means, yerr=l1_stds, capsize=5,
              color=[attack_colors[a] for a in attacks], alpha=0.85, ecolor="black")
    ax_l1.set_title("Normalized L1 Drift", fontsize=9)
    ax_l1.set_ylabel("Normalized L1 (clean vs adv attribution)", fontsize=8)
    ax_l1.set_xticklabels(attacks, rotation=20, ha="right", fontsize=8)

    # Cosine similarity
    ax_cos = axes[1]
    cos_means = [np.mean(cos_by_attack[a]) if a in cos_by_attack else 0.0 for a in attacks]
    cos_stds = [np.std(cos_by_attack[a]) if a in cos_by_attack else 0.0 for a in attacks]
    ax_cos.bar(attacks, cos_means, yerr=cos_stds, capsize=5,
               color=[attack_colors[a] for a in attacks], alpha=0.85, ecolor="black")
    ax_cos.set_title("Cosine Similarity", fontsize=9)
    ax_cos.set_ylabel("Attribution Cosine Similarity", fontsize=8)
    ax_cos.set_ylim(0, 1.15)
    ax_cos.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax_cos.set_xticklabels(attacks, rotation=20, ha="right", fontsize=8)

    plt.tight_layout()
    return str(save_fig(fig, output_dir, name))


# =====================================================================
# plot_confidence_shift
# =====================================================================

def plot_confidence_shift(results: List[Dict[str, Any]],
                           output_dir: str, name: str = "confidence_shift") -> Optional[str]:
    """
    Scatter plot: clean vs. adversarial confidence for the clean class,
    colored by attack success (red=success, blue=failure).

    Args:
        results: List of explanation result dicts.
        output_dir: Directory to save figure.
        name: Filename stem.

    Returns:
        Path to saved PNG.
    """
    if not results:
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.suptitle("Clean vs Adversarial Confidence Shift", fontsize=11, fontweight="bold")
        _no_data(ax)
        return str(save_fig(fig, output_dir, name))

    clean_confs = []
    adv_confs = []
    attack_success = []

    for r in results:
        cc = r.get("clean_confidence")
        ac = r.get("adversarial_confidence")
        fa = r.get("failure_analysis", {}) or {}
        success = bool(fa.get("attack_caused_failure", False))
        if cc is not None and ac is not None:
            clean_confs.append(float(cc))
            adv_confs.append(float(ac))
            attack_success.append(success)

    if not clean_confs:
        fig, ax = plt.subplots(figsize=(7, 6))
        _no_data(ax, "No confidence data")
        return str(save_fig(fig, output_dir, name))

    colors = [CB_COLORS["orange"] if s else CB_COLORS["blue"] for s in attack_success]

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("Clean vs Adversarial Confidence (clean class)",
                 fontsize=11, fontweight="bold")

    sc = ax.scatter(clean_confs, adv_confs, c=colors, alpha=0.7, s=60, edgecolors="gray", linewidths=0.5)

    diag = np.linspace(0, 1, 100)
    ax.plot(diag, diag, "k--", linewidth=0.8, label="No change")
    ax.set_xlabel("Clean Confidence", fontsize=9)
    ax.set_ylabel("Adversarial Confidence", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=CB_COLORS["orange"], label="Attack succeeded"),
        Patch(facecolor=CB_COLORS["blue"], label="Attack failed"),
    ]
    ax.legend(handles=legend_elems, fontsize=8)
    plt.tight_layout()
    return str(save_fig(fig, output_dir, name))
