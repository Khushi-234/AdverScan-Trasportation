"""
Assessment History Service for AdverScan Frontend.

Interfaces the Streamlit frontend with the underlying SQLite StorageRepository.
Provides methods for querying historical scan records, filtering by model/modality/status,
fetching complete hierarchical scan details, and deleting records.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path

from app.storage.storage import StorageRepository, DEFAULT_ARTIFACT_DIR


_REPO_INSTANCE: Optional[StorageRepository] = None


def get_storage_repository() -> StorageRepository:
    """Retrieve or initialize the global StorageRepository singleton."""
    global _REPO_INSTANCE
    if _REPO_INSTANCE is None:
        _REPO_INSTANCE = StorageRepository()
    return _REPO_INSTANCE


def get_scan_history(
    status: Optional[str] = None,
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    modality: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Retrieve filtered list of historical assessment records.

    Args:
        status: 'running', 'completed', 'partially_completed', 'failed'.
        model_name: Substring filter on model name.
        dataset_name: Substring filter on dataset name.
        modality: Domain filter ('image', 'text', 'tabular', 'time_series').
        limit: Max records to return.
        offset: Pagination offset.

    Returns:
        List of serialized assessment dictionaries.
    """
    repo = get_storage_repository()
    return repo.list_assessments(
        status=status,
        model_name=model_name,
        dataset_name=dataset_name,
        modality=modality,
        limit=limit,
        offset=offset,
    )


def get_scan_detail(scan_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve full hierarchical assessment record by scan_id, including
    model, dataset, attack runs, defense runs, XAI results, vulnerabilities,
    and report links.
    """
    if not scan_id:
        return None
    repo = get_storage_repository()
    return repo.get_assessment(scan_id)


def delete_scan(scan_id: str) -> bool:
    """
    Permanently delete an assessment and all its child execution records.
    """
    if not scan_id:
        return False
    repo = get_storage_repository()
    return repo.delete_assessment(scan_id)


def get_database_status() -> Dict[str, Any]:
    """
    Retrieve diagnostic information about the SQLite database, WAL state,
    and total stored evaluation records.
    """
    repo = get_storage_repository()
    return repo.get_storage_stats()


def get_scans_for_selection() -> List[Dict[str, str]]:
    """
    Retrieve a lightweight list of all available completed/scanned assessments
    formatted for UI dropdown select boxes.
    """
    scans = get_scan_history(limit=100)
    out = []
    for s in scans:
        s_id = s.get("scan_id", "")
        m_name = s.get("model_name") or "Unknown Model"
        dom = (s.get("modality") or "general").capitalize()
        status = s.get("status", "unknown").capitalize()
        created = s.get("created_at")
        date_str = str(created)[:16] if created else "N/A"
        
        attacks_count = len(s.get("attacks", []))
        label = f"{s_id} | {m_name} ({dom}) | {attacks_count} Attacks | {date_str} [{status}]"
        
        out.append({
            "scan_id": s_id,
            "display_label": label,
            "model_name": m_name,
            "modality": s.get("modality") or "image",
            "status": s.get("status", "completed"),
            "date": date_str,
        })
    return out

