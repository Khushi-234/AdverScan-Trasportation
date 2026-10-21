"""
Ingestion logging utilities for AdverScan.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Union


def log_ingestion_event(
    event_data: Dict[str, Any],
    log_filepath: Union[str, Path] = "results/ingestion_log.json",
) -> str:
    """
    Record an ingestion or environment setup event to a structured JSON log.

    Args:
        event_data: Metadata dictionary describing the model and setup.
        log_filepath: Target output file path.

    Returns:
        Path string to the written log file.
    """
    log_path = Path(log_filepath)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    if log_path.exists() and log_path.stat().st_size > 0:
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                history = json.load(f)
                if not isinstance(history, list):
                    history = [history]
        except Exception:
            history = []

    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "event_type": "MODEL_INGESTION_AND_ENVIRONMENT_SETUP",
        **event_data,
    }
    history.append(entry)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return str(log_path)
