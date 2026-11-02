"""
Session State Management for AdverScan Frontend.

Manages the assessment workflow stages:
Dashboard -> Model Ingestion -> Attack Configuration -> Attack Results
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

DEFAULT_SESSION_STATE: Dict[str, Any] = {
    "workflow_stage": "dashboard",  # dashboard, ingestion, attack_config, attack_results
    "theme": "Dark",
    "is_assessment_active": False,
    "active_model_name": None,
    "active_model_domain": None,
    "active_model_framework": None,
    "active_metadata_dict": None,
    "active_attack_name": None,
    "last_error": None,
    # Internal in-memory handles for the active assessment run
    "_active_assessment_data": None,
    "_active_adapter": None,
    "_active_metadata": None,
    "_test_inputs": None,
    "_test_labels": None,
    "_latest_attack_result": None,
}


def init_session_state() -> None:
    """Initialize default session state keys if not already present."""
    for key, val in DEFAULT_SESSION_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_state(key: str, default: Any = None) -> Any:
    """Safely get a value from session state."""
    init_session_state()
    return st.session_state.get(key, default)


def set_state(key: str, value: Any) -> None:
    """Safely set a value in session state."""
    init_session_state()
    # Guard against accidental storage of arbitrary un-managed tensors in public keys
    if not key.startswith("_"):
        val_type = type(value).__name__
        if val_type in ("Tensor", "ndarray", "Module") and key not in (
            "_test_inputs",
            "_test_labels",
        ):
            raise ValueError(
                f"Cannot store {val_type} directly in public session state key '{key}'. "
                "Use internal prefixed keys."
            )
    st.session_state[key] = value


def get_workflow_stage() -> str:
    """Get the current assessment workflow stage."""
    init_session_state()
    return st.session_state.get("workflow_stage", "dashboard")


def set_workflow_stage(stage: str) -> None:
    """Transition to a new assessment workflow stage."""
    init_session_state()
    valid_stages = ("dashboard", "ingestion", "attack_config", "attack_results")
    if stage not in valid_stages:
        raise ValueError(
            f"Invalid workflow stage '{stage}'. Must be one of {valid_stages}"
        )
    st.session_state["workflow_stage"] = stage


def set_active_assessment(data: Dict[str, Any]) -> None:
    """Store loaded model and dataset context into active session state."""
    init_session_state()
    st.session_state["is_assessment_active"] = True
    st.session_state["_active_assessment_data"] = data
    st.session_state["_active_adapter"] = data.get("adapter")
    st.session_state["_active_metadata"] = data.get("metadata")
    st.session_state["_test_inputs"] = data.get("test_inputs")
    st.session_state["_test_labels"] = data.get("test_labels")
    st.session_state["active_metadata_dict"] = data.get("metadata_dict") or {}
    meta = data.get("metadata_dict") or {}
    st.session_state["active_model_name"] = meta.get("Model Name") or data.get(
        "model_identifier"
    )
    st.session_state["active_model_domain"] = data.get("domain", "image")
    st.session_state["active_model_framework"] = meta.get("Framework") or data.get(
        "framework"
    )
    st.session_state["last_error"] = None


def get_active_assessment() -> Optional[Dict[str, Any]]:
    """Retrieve active model and dataset context."""
    init_session_state()
    if not st.session_state.get("is_assessment_active"):
        return None

    stored = st.session_state.get("_active_assessment_data")
    if isinstance(stored, dict) and stored.get("adapter") is not None:
        return stored

    return {
        "adapter": st.session_state.get("_active_adapter"),
        "metadata": st.session_state.get("_active_metadata"),
        "test_inputs": st.session_state.get("_test_inputs"),
        "test_labels": st.session_state.get("_test_labels"),
        "metadata_dict": st.session_state.get("active_metadata_dict", {}),
        "domain": st.session_state.get("active_model_domain", "image"),
    }


def set_attack_result(result: Dict[str, Any]) -> None:
    """Save latest attack execution result."""
    init_session_state()
    st.session_state["_latest_attack_result"] = result
    st.session_state["active_attack_name"] = result.get("attack_name")


def get_attack_result() -> Optional[Dict[str, Any]]:
    """Get latest attack execution result."""
    init_session_state()
    return st.session_state.get("_latest_attack_result")


def reset_workflow_state() -> None:
    """Reset the entire workflow back to the idle dashboard state."""
    init_session_state()
    saved_theme = st.session_state.get("theme", "Dark")
    for key, val in DEFAULT_SESSION_STATE.items():
        st.session_state[key] = val
    st.session_state["theme"] = saved_theme
