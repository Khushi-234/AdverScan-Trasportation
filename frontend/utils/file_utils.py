"""
File Utilities for AdverScan Frontend.

Safely reads scan results, reports, and logs from disk without mutating
backend behavior or inventing historical scan data.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def find_reports_directory() -> Path:
    """Locate the reports directory relative to repo root or working directory."""
    # Try current working directory or relative to this file
    cwd_reports = Path.cwd() / "reports"
    if cwd_reports.is_dir():
        return cwd_reports

    module_root = Path(__file__).resolve().parent.parent.parent / "reports"
    if module_root.is_dir():
        return module_root

    return cwd_reports


def get_all_scans() -> List[Dict[str, Any]]:
    """
    Safely inspect the reports directory for completed or ongoing scan reports.
    Returns a sorted list of scan summary dictionaries, newest first.
    If no valid scans are found, returns an empty list.
    """
    reports_dir = find_reports_directory()
    if not reports_dir.exists():
        return []

    scans: List[Dict[str, Any]] = []
    # Search for all adverscan_report*.json files within subdirectories of reports
    pattern = str(reports_dir / "SCAN-*" / "*.json")
    for filepath in glob.glob(pattern):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                continue

            exec_summary = data.get("executive_summary", {})
            model_info = data.get("model_info", {})

            scan_id = (
                data.get("scan_id")
                or exec_summary.get("scan_id")
                or Path(filepath).parent.name
            )
            timestamp = data.get("timestamp") or exec_summary.get("timestamp") or "N/A"
            status = (
                data.get("status") or exec_summary.get("execution_status") or "UNKNOWN"
            )
            risk_level = exec_summary.get("risk_level") or "UNKNOWN"
            vuln_score = exec_summary.get("vulnerability_score")
            attacks = exec_summary.get("attacks_evaluated", [])
            model_name = model_info.get("model_name") or "Unknown Model"
            framework = model_info.get("framework") or "Unknown"

            scans.append(
                {
                    "scan_id": scan_id,
                    "timestamp": timestamp,
                    "status": status,
                    "risk_level": str(risk_level).upper(),
                    "vulnerability_score": vuln_score,
                    "attacks": attacks,
                    "num_attacks": len(attacks),
                    "model_name": model_name,
                    "framework": framework,
                    "file_path": filepath,
                    "hardening_applied": bool(exec_summary.get("hardening_applied")),
                    "retest_conducted": bool(exec_summary.get("retest_conducted")),
                }
            )
        except Exception:
            # Silently skip corrupted or unreadable individual report files
            continue

    # Sort descending by timestamp (or scan_id if timestamp not standard)
    scans.sort(key=lambda s: str(s.get("timestamp", "")), reverse=True)
    return scans


def get_scan_report(scan_id: str) -> Optional[Dict[str, Any]]:
    """Safely load full JSON report for a given scan_id."""
    reports_dir = find_reports_directory()
    scan_dir = reports_dir / scan_id
    if not scan_dir.is_dir():
        # Fallback search by scan_id in all scan directories
        pattern = str(reports_dir / f"*{scan_id}*" / "*.json")
        matches = glob.glob(pattern)
        if matches:
            filepath = matches[0]
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    json_files = list(scan_dir.glob("*.json"))
    if not json_files:
        return None

    try:
        with open(json_files[0], "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_dashboard_summary_metrics() -> Dict[str, Any]:
    """
    Calculate summary counts from actual scans on disk.
    If no scans exist, returns all zeroes to enable explicit empty state handling.
    """
    scans = get_all_scans()
    total_scans = len(scans)
    if total_scans == 0:
        return {
            "total_scans": 0,
            "models_evaluated": 0,
            "high_risk_findings": 0,
            "completed_scans": 0,
            "risk_distribution": {},
            "recent_scans": [],
        }

    unique_models = set(
        s["model_name"] for s in scans if s["model_name"] != "Unknown Model"
    )
    high_risk_count = sum(1 for s in scans if s["risk_level"] in ("HIGH", "CRITICAL"))
    completed_count = sum(
        1 for s in scans if s["status"].upper() in ("SUCCESS", "COMPLETED")
    )

    risk_dist: Dict[str, int] = {
        "CRITICAL": 0,
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0,
        "NEGLIGIBLE": 0,
        "UNKNOWN": 0,
    }
    for s in scans:
        r = s["risk_level"]
        if r in risk_dist:
            risk_dist[r] += 1
        else:
            risk_dist["UNKNOWN"] += 1

    # Remove zero counts from distribution
    filtered_dist = {k: v for k, v in risk_dist.items() if v > 0}

    return {
        "total_scans": total_scans,
        "models_evaluated": len(unique_models) if unique_models else total_scans,
        "high_risk_findings": high_risk_count,
        "completed_scans": completed_count,
        "risk_distribution": filtered_dist,
        "recent_scans": scans[:10],
    }
