"""
Build all XAI visualization figures from saved evaluation outputs.
Entry point: build_all_figures(output_dir, max_samples_per_attack).
CLI: python -m app.explainability.visualization.build_figures --output-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.explainability.visualization.common import load_results
from app.explainability.visualization.sample_plots import plot_sample
from app.explainability.visualization.summary_plots import (
    plot_attack_summary,
    plot_attribution_drift,
    plot_confidence_shift,
    plot_exploited_features,
    plot_failure_modes,
)
from app.explainability.visualization.run_plots import (
    plot_deletion_curves,
    plot_run_overview,
    plot_speed_vs_faithfulness,
)

logger = logging.getLogger(__name__)

FIXED_SEED: int = 42


# =====================================================================
# Sample selection helpers
# =====================================================================

def _select_representative_samples(results: List[Dict[str, Any]],
                                    max_per_attack: int,
                                    seed: int = FIXED_SEED) -> List[Dict[str, Any]]:
    """
    Select representative samples per (attack, modality): prefer successful attacks with
    high adversarial confidence, plus one failed attack for contrast. Uses a fixed seed.

    Args:
        results: All explanation result dicts.
        max_per_attack: Maximum samples to select per (attack, modality) group.
        seed: Random seed for reproducibility.

    Returns:
        List of selected result dicts.
    """
    from collections import defaultdict

    rng = random.Random(seed)

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        atk = str(r.get("attack_name", "unknown"))
        mod = str(r.get("modality", "image"))
        groups[f"{atk}__{mod}"].append(r)

    selected: List[Dict[str, Any]] = []
    for group_key, group_results in groups.items():
        fa_map = {
            id(r): (r.get("failure_analysis") or {}).get("attack_caused_failure", False)
            for r in group_results
        }
        success_rs = [r for r in group_results if fa_map[id(r)]]
        failed_rs = [r for r in group_results if not fa_map[id(r)]]

        # Sort successes by adversarial confidence descending
        success_rs.sort(key=lambda r: -float(r.get("adversarial_confidence", 0.0)))
        take_success = success_rs[: max(1, max_per_attack - 1)]

        # Take one failed for contrast
        take_failed = rng.sample(failed_rs, min(1, len(failed_rs)))

        selected.extend(take_success)
        selected.extend(take_failed)

    return selected


# =====================================================================
# Index management
# =====================================================================

def _append_index_entry(
    index: List[Dict[str, Any]],
    path: Optional[str],
    caption: str,
    modality: str,
    attack: str,
    technique: str,
) -> None:
    """Append a figure metadata entry to the index list."""
    if path is None:
        return
    index.append({
        "path": str(path),
        "caption": caption,
        "modality": modality,
        "attack": attack,
        "technique": technique,
    })


def _load_status(output_dir: Path) -> Dict[str, Any]:
    """
    Load xai_status.json from output_dir if present.
    Returns empty dict if file is absent.
    """
    status_path = output_dir / "xai_status.json"
    if status_path.exists():
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# =====================================================================
# build_all_figures
# =====================================================================

def build_all_figures(output_dir: str, max_samples_per_attack: int = 5) -> int:
    """
    Load saved XAI results and build all visualization figures.

    Steps:
      1. Load results and status.
      2. Select representative samples per attack/modality (fixed seed).
      3. Save per-sample, summary, and runtime plots.
      4. Write figures/figures_index.json.
      5. Mark index partial=true if status is pending/running.
      6. Log individual failures and continue.

    Args:
        output_dir: Directory containing XAI result files.
        max_samples_per_attack: Maximum samples plotted per (attack, modality).

    Returns:
        Number of successfully saved figures.
    """
    out_path = Path(output_dir)
    fig_dir = out_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load results and status
    results = load_results(out_path)
    status = _load_status(out_path)

    is_partial = status.get("status") in ("pending", "running") or len(results) == 0
    logger.info("Loaded %d results from %s (partial=%s)", len(results), output_dir, is_partial)

    index: List[Dict[str, Any]] = []
    saved_count = 0

    # 2. Select representative samples
    samples = _select_representative_samples(results, max_samples_per_attack)

    # 3a. Per-sample plots
    for i, sample in enumerate(samples):
        atk = str(sample.get("attack_name", "unknown"))
        mod = str(sample.get("modality", "image"))
        tech = str(sample.get("technique", "default"))
        fig_name = f"sample_{mod}_{atk}_{i:03d}"
        try:
            path = plot_sample(sample, out_path, fig_name)
            if path is not None:
                _append_index_entry(
                    index, str(path),
                    caption=f"{mod.capitalize()} sample | attack={atk} | tech={tech}",
                    modality=mod, attack=atk, technique=tech,
                )
                saved_count += 1
        except Exception as exc:
            logger.warning("Failed sample plot %s: %s", fig_name, exc)

    # 3b. Summary plots
    summary_fns = [
        ("attack_summary", lambda: plot_attack_summary(results, str(out_path))),
        ("failure_modes", lambda: plot_failure_modes(results, str(out_path))),
        ("exploited_features", lambda: plot_exploited_features(results, str(out_path))),
        ("attribution_drift", lambda: plot_attribution_drift(results, str(out_path))),
        ("confidence_shift", lambda: plot_confidence_shift(results, str(out_path))),
    ]
    for fig_id, fn in summary_fns:
        try:
            path = fn()
            if path is not None:
                _append_index_entry(
                    index, path,
                    caption=fig_id.replace("_", " ").title(),
                    modality="all", attack="all", technique="all",
                )
                saved_count += 1
        except Exception as exc:
            logger.warning("Failed summary plot %s: %s", fig_id, exc)

    # 3c. Runtime plots
    try:
        path = plot_run_overview(results, str(out_path))
        if path is not None:
            _append_index_entry(index, path, caption="Run Overview",
                                 modality="all", attack="all", technique="all")
            saved_count += 1
    except Exception as exc:
        logger.warning("Failed run overview plot: %s", exc)

    benchmark_csv = str(out_path / "benchmark.csv")
    for fn_name, fn in [
        ("speed_vs_faithfulness", lambda: plot_speed_vs_faithfulness(benchmark_csv, str(out_path))),
        ("deletion_curves", lambda: plot_deletion_curves(benchmark_csv, str(out_path))),
    ]:
        try:
            path = fn()
            if path is not None:
                _append_index_entry(index, path, caption=fn_name.replace("_", " ").title(),
                                     modality="all", attack="all", technique="all")
                saved_count += 1
        except Exception as exc:
            logger.warning("Failed benchmark plot %s: %s", fn_name, exc)

    # 4. Write index
    index_data: Dict[str, Any] = {
        "figures": index,
        "total": saved_count,
        "partial": is_partial,
        "output_dir": str(out_path),
    }
    index_path = fig_dir / "figures_index.json"
    try:
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)
        logger.info("Figures index written to %s", index_path)
    except Exception as exc:
        logger.warning("Failed to write figures index: %s", exc)

    return saved_count


# =====================================================================
# CLI entry point
# =====================================================================

def _main() -> None:
    """CLI entry: python -m app.explainability.visualization.build_figures --output-dir <dir>"""
    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Build XAI visualization figures from saved results."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory containing XAI explanation result files."
    )
    parser.add_argument(
        "--max-samples", type=int, default=5,
        help="Maximum samples per (attack, modality) group (default: 5)."
    )
    args = parser.parse_args()

    n = build_all_figures(args.output_dir, max_samples_per_attack=args.max_samples)
    print(f"[XAI FIGURES] {n} figures saved to {args.output_dir}")


if __name__ == "__main__":
    _main()
