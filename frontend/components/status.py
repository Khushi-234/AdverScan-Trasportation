"""
Status Badges, Empty States, and Phase Placeholder Components for AdverScan.
"""

from __future__ import annotations

from typing import List, Optional

import streamlit as st


def render_status_badge(status: str) -> str:
    """Return an HTML snippet for a styled execution status badge."""
    st_upper = status.upper()
    if st_upper in ("SUCCESS", "COMPLETED"):
        css_class = "badge-success"
    elif st_upper in ("FAILED", "ERROR"):
        css_class = "badge-failed"
    elif st_upper in ("RUNNING", "IN_PROGRESS"):
        css_class = "badge-running"
    else:
        css_class = "badge-pending"

    return f'<span class="badge-status {css_class}">{st_upper}</span>'


def render_risk_badge(risk_level: str) -> str:
    """Return an HTML snippet for a styled risk severity badge."""
    r_upper = str(risk_level).upper()
    if r_upper == "CRITICAL":
        css_class = "badge-critical"
    elif r_upper == "HIGH":
        css_class = "badge-high"
    elif r_upper == "MEDIUM":
        css_class = "badge-medium"
    elif r_upper == "LOW":
        css_class = "badge-low"
    else:
        css_class = "badge-pending"

    return f'<span class="badge-status {css_class}">{r_upper}</span>'


def render_empty_state(
    title: str = "No Data Recorded",
    description: str = "No scans or evaluation records have been generated yet.",
    action_label: Optional[str] = "Start New Assessment",
    action_page: Optional[str] = "pages/2_Model_Ingestion.py",
) -> None:
    """Render an explicit empty state box with an optional action button."""
    st.markdown(
        f"""
        <div class="empty-state-box">
            <div class="empty-state-icon">🛡️</div>
            <div class="empty-state-title">{title}</div>
            <div class="empty-state-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if action_label and action_page:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                action_label,
                use_container_width=True,
                type="primary",
                key=f"empty_action_{title.replace(' ', '_').lower()}",
            ):
                st.switch_page(action_page)


def render_phase_placeholder(
    phase_num: int,
    phase_name: str,
    description: str,
    next_phase_integration_notes: str,
    planned_inputs: Optional[List[str]] = None,
    planned_outputs: Optional[List[str]] = None,
) -> None:
    """
    Render a standardized, informative placeholder for pipeline phases that
    are awaiting backend engine integration in subsequent phases.
    """
    st.markdown(
        f"""
        <div class="phase-card">
            <div class="phase-header">
                <span class="phase-number">PIPELINE PHASE {phase_num:02d}</span>
                <span class="badge-status badge-pending">INTEGRATION PLANNED</span>
            </div>
            <h3 style="margin: 0 0 0.5rem 0; color: var(--text-primary);">{phase_name}</h3>
            <p style="color: var(--text-muted); font-size: 0.95rem; margin-bottom: 1rem;">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Planned Inputs & Parameters")
        if planned_inputs:
            for item in planned_inputs:
                st.markdown(f"- **{item}**")
        else:
            st.markdown("_No direct user inputs required._")

    with col2:
        st.subheader("📊 Planned Artifacts & Outputs")
        if planned_outputs:
            for item in planned_outputs:
                st.markdown(f"- **{item}**")
        else:
            st.markdown("_Direct pipeline output._")

    st.markdown("---")
    st.info(f"**Integration Blueprint**: {next_phase_integration_notes}")
