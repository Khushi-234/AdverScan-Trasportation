"""
Reusable Metric Card Components for AdverScan.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st


def render_metric_card(
    title: str,
    value: Any,
    delta: Optional[str] = None,
    delta_color: str = "normal",  # "normal", "inverse", "off"
    help_text: Optional[str] = None,
    border_accent: Optional[str] = None,
) -> None:
    """Render a styled metric card using Streamlit native or container markup."""
    # Streamlit st.metric is clean, responsive, and native
    st.metric(
        label=title, value=value, delta=delta, delta_color=delta_color, help=help_text
    )


def render_summary_metrics(summary: Dict[str, Any]) -> None:
    """Render the standard executive summary row with 4 primary metrics."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_metric_card(
            title="Total Scans",
            value=summary.get("total_scans", 0),
            help_text="Cumulative number of scans evaluated across all models",
        )

    with col2:
        render_metric_card(
            title="Models Evaluated",
            value=summary.get("models_evaluated", 0),
            help_text="Distinct models tested for adversarial robustness",
        )

    with col3:
        high_risk = summary.get("high_risk_findings", 0)
        render_metric_card(
            title="High-Risk Findings",
            value=high_risk,
            delta=f"{high_risk} flagged" if high_risk > 0 else "0 flagged",
            delta_color="inverse" if high_risk > 0 else "normal",
            help_text="Scans with HIGH or CRITICAL risk severity classification",
        )

    with col4:
        completed = summary.get("completed_scans", 0)
        total = summary.get("total_scans", 0)
        rate = f"{(completed / total * 100):.0f}%" if total > 0 else "0%"
        render_metric_card(
            title="Completed Scans",
            value=completed,
            delta=f"{rate} success rate" if total > 0 else None,
            help_text="Successfully finished end-to-end security evaluations",
        )
