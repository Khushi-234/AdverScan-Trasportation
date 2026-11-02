"""
Sidebar Navigation and Status Component for AdverScan.
"""

from __future__ import annotations

import streamlit as st

from frontend.utils.session_state import (
    get_state,
    get_workflow_stage,
    reset_workflow_state,
    set_state,
)


def render_sidebar_branding() -> None:
    """Render AdverScan branding in sidebar header."""
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand" style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.5rem;">
                <span class="brand-icon" style="font-size: 2rem;">🛡️</span>
                <div>
                    <h2 style="margin: 0; font-size: 1.35rem; font-weight: 700;">AdverScan</h2>
                    <span class="brand-subtitle" style="font-size: 0.72rem; letter-spacing: 1px; font-weight: 700;">ROBUSTNESS PLATFORM</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Adversarial Robustness & AI Security Assessment")


def render_sidebar_session() -> None:
    """Render active assessment session details and controls in the sidebar."""
    with st.sidebar:
        st.markdown("---")
        is_active = get_state("is_assessment_active", False)
        stage = get_workflow_stage()

        st.markdown(
            "<div class='sidebar-section-title' style='margin-bottom: 0.5rem;'>Assessment Session</div>",
            unsafe_allow_html=True,
        )

        if not is_active:
            st.markdown(
                """
                <div class="assessment-session">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-size: 0.85rem; font-weight: 600;">Status</span>
                        <span class="status-idle">Idle</span>
                    </div>
                    <div style="font-size: 0.82rem; color: var(--text-secondary);">No active assessment in progress.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            model_name = get_state("active_model_name") or "Ingested Model"
            domain = get_state("active_model_domain", "Unknown")

            stage_labels = {
                "ingestion": "1. Model Ingestion",
                "attack_config": "2. Attack Discovery & Config",
                "attack_results": "3. Attack Results",
            }
            stage_str = stage_labels.get(stage, "Active")

            st.markdown(
                f"""
                <div class="modern-card card-accent-cyan" style="padding: 0.75rem 1rem; margin-bottom: 0.75rem;">
                    <div class="stat-label">Active Model</div>
                    <div style="font-size: 0.95rem; font-weight: 700; color: var(--text-primary); margin: 0.2rem 0;">{model_name}</div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary);">Domain: <strong style="color: var(--text-primary);">{str(domain).capitalize()}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(f"**Current Stage**: `{stage_str}`")

            st.markdown("---")
            if st.button(
                "Reset Assessment",
                use_container_width=True,
                help="Discard active assessment and return to Dashboard",
            ):
                reset_workflow_state()
                st.rerun()

        st.markdown(
            """
            <div class="sidebar-version" style="margin-top: 1rem; font-size: 0.78rem;">
                AdverScan v0.1.0 • Pipeline Mode
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sidebar() -> None:
    """
    Render clean sidebar containing AdverScan branding and session info.
    Theme selector is handled by theme_manager.initialize_theme().
    """
    render_sidebar_branding()
    render_sidebar_session()
