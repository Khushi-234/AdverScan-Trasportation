"""
ReportWriter — Handles persisting an AdverScan ReportResult to disk.

Writes the complete report to a scan-specific output folder inside the
project-level ``reports/`` directory:

    reports/<scan_id>/adverscan_report_<attacks>_<scan_id>.md     ← Markdown
    reports/<scan_id>/adverscan_report_<attacks>_<scan_id>.json   ← JSON
    reports/<scan_id>/adverscan_report_<attacks>_<scan_id>_timing.csv  ← CSV

File stem is built from the actual attacks performed:
  - Only PGD       →  adverscan_report_pgd_<scan_id>.md
  - FGSM + PGD     →  adverscan_report_fgsm_pgd_<scan_id>.md
  - FGSM+PGD+DF    →  adverscan_report_fgsm_pgd_deepfool_<scan_id>.md

All output methods are optional — callers pick which formats to persist.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .report_result import ReportResult


# Project-level reports/ directory: two levels up from this file
# (app/report_generator/report_writer.py → app/ → project root)
_DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"


class ReportWriter:
    """
    Persists a ReportResult to a structured output directory.

    By default, reports land in the project-level ``reports/`` folder:

        reports/<scan_id>/adverscan_report_<attacks>_<scan_id>.md
        reports/<scan_id>/adverscan_report_<attacks>_<scan_id>.json
        reports/<scan_id>/adverscan_report_<attacks>_<scan_id>_timing.csv

    The ``<attacks>`` segment is built from the actual attack names present
    in the report result, preserving their insertion order:

        # Only PGD ran:
        writer.write(result)  →  reports/SCAN-XYZ/adverscan_report_pgd_scan-xyz.md

        # FGSM + PGD ran:
        writer.write(result)  →  reports/SCAN-XYZ/adverscan_report_fgsm_pgd_scan-xyz.md

        # FGSM + PGD + DeepFool ran:
        writer.write(result)  →  reports/SCAN-XYZ/adverscan_report_fgsm_pgd_deepfool_scan-xyz.md
    """

    def __init__(self, output_root: Union[str, Path, None] = None) -> None:
        """
        Args:
            output_root: Root folder for all reports.
                         Defaults to the project-level ``reports/`` directory.
        """
        self.output_root = Path(output_root) if output_root is not None else _DEFAULT_REPORTS_DIR

    # ── Public API ─────────────────────────────────────────────────────────────

    def write(
        self,
        result: ReportResult,
        formats: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Write the report to disk in the requested formats.

        Args:
            result: Completed ReportResult from ReportGenerator.generate().
            formats: List of output formats. Defaults to ["md", "json"].
                     Supported: "md", "json", "csv", "txt".

        Returns:
            Dict mapping format name → absolute file path.
        """
        if formats is None:
            formats = ["md", "json"]

        # ── Output directory: reports/<scan_id>/ ───────────────────────────────
        scan_id = result.scan_id or result.report_id
        out_dir = self.output_root / scan_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── File stem: adverscan_report_<attacks>_<scan_id_lower> ─────────────
        # Built purely from the actual attack names in the result — no hardcoding.
        # Preserves the insertion order of attacks as they were run.
        attack_names = list((result.attack_results or {}).keys())
        attacks_segment = "_".join(name.lower() for name in attack_names) if attack_names else "scan"
        scan_suffix = scan_id.lower().replace("-", "_")
        stem = f"adverscan_report_{attacks_segment}_{scan_suffix}"

        written: Dict[str, str] = {}

        for fmt in formats:
            fmt = fmt.lower().strip()
            try:
                if fmt in ("md", "markdown"):
                    path = out_dir / f"{stem}.md"
                    self._write_markdown(result, path)
                    written["md"] = str(path.resolve())
                elif fmt == "json":
                    path = out_dir / f"{stem}.json"
                    self._write_json(result, path)
                    written["json"] = str(path.resolve())
                elif fmt == "csv":
                    path = out_dir / f"{stem}_timing.csv"
                    self._write_csv(result, path)
                    written["csv"] = str(path.resolve())
                elif fmt in ("txt", "text"):
                    path = out_dir / f"{stem}.txt"
                    self._write_text(result, path)
                    written["txt"] = str(path.resolve())
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠ ReportWriter: Failed to write '{fmt}' — {exc}")

        if written:
            print(f"\n📁 Report saved → {out_dir.resolve()}/")
            for fmt, p in written.items():
                print(f"   [{fmt.upper():>4}] {Path(p).name}")

        return written

    # ── Format Writers ─────────────────────────────────────────────────────────

    def _write_markdown(self, result: ReportResult, path: Path) -> None:
        """Write the formatted Markdown / text report."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(result.formatted_report)

    def _write_text(self, result: ReportResult, path: Path) -> None:
        """Write the plain text report (same content as Markdown)."""
        self._write_markdown(result, path)

    def _write_json(self, result: ReportResult, path: Path) -> None:
        """Write the full report as a structured JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

    def _write_csv(self, result: ReportResult, path: Path) -> None:
        """Write a per-module execution timing CSV."""
        path.parent.mkdir(parents=True, exist_ok=True)

        es = result.execution_summary
        if es is None or not es.modules:
            # Write a minimal placeholder
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["module_id", "module_name", "status", "elapsed_seconds"])
                writer.writerow(["N/A", "No execution data captured", "UNKNOWN", 0.0])
            return

        # Collect all possible metric column names across modules
        metric_keys: List[str] = []
        for rec in es.modules:
            for k in (rec.metrics or {}):
                col = f"metric_{k}"
                if col not in metric_keys:
                    metric_keys.append(col)

        header = ["module_id", "module_name", "status", "elapsed_seconds"] + metric_keys

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            for rec in es.modules:
                row: Dict[str, Any] = {
                    "module_id": rec.module_id,
                    "module_name": rec.module_name,
                    "status": rec.status,
                    "elapsed_seconds": rec.elapsed_seconds,
                }
                for mk, mv in (rec.metrics or {}).items():
                    row[f"metric_{mk}"] = mv
                writer.writerow(row)

        # Append summary footer row
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([])
            writer.writerow(
                ["TOTAL", es.run_label, es.overall_status, es.total_elapsed_seconds]
            )
