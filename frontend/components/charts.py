"""
Interactive Plotly Chart Components for AdverScan.
"""

from __future__ import annotations

from typing import Dict, Optional

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Color mappings for risk levels
RISK_COLOR_MAP = {
    "CRITICAL": "#ef4444",
    "HIGH": "#f97316",
    "MEDIUM": "#eab308",
    "LOW": "#10b981",
    "NEGLIGIBLE": "#06b6d4",
    "UNKNOWN": "#64748b",
}


def render_risk_distribution_chart(risk_counts: Dict[str, int]) -> None:
    """Render a donut chart displaying risk severity breakdown."""
    if not risk_counts or sum(risk_counts.values()) == 0:
        st.info("No risk distribution data available.")
        return

    labels = list(risk_counts.keys())
    values = list(risk_counts.values())
    colors = [RISK_COLOR_MAP.get(label.upper(), "#94a3b8") for label in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.6,
                marker=dict(colors=colors, line=dict(color="#1e293b", width=2)),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text="Risk Severity Distribution", font=dict(size=16, color="#f8fafc")
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_vulnerability_score_gauge(
    score: float, title: str = "Vulnerability Score"
) -> None:
    """Render a gauge indicator for vulnerability score (0-10 or 0-100)."""
    max_val = 10.0 if score <= 10.0 else 100.0

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title, "font": {"size": 18, "color": "#f8fafc"}},
            gauge={
                "axis": {"range": [0, max_val], "tickcolor": "#94a3b8"},
                "bar": {"color": "#38bdf8"},
                "bgcolor": "rgba(30, 41, 59, 0.5)",
                "borderwidth": 1,
                "bordercolor": "#475569",
                "steps": [
                    {"range": [0, max_val * 0.3], "color": "rgba(16, 185, 129, 0.3)"},
                    {
                        "range": [max_val * 0.3, max_val * 0.7],
                        "color": "rgba(234, 179, 8, 0.3)",
                    },
                    {
                        "range": [max_val * 0.7, max_val],
                        "color": "rgba(239, 68, 68, 0.3)",
                    },
                ],
            },
        )
    )

    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_attacks_frequency_chart(scans_list: list) -> None:
    """Render a horizontal bar chart showing frequency of evaluated attack algorithms."""
    attack_counts: Dict[str, int] = {}
    for s in scans_list:
        for att in s.get("attacks", []):
            attack_counts[att] = attack_counts.get(att, 0) + 1

    if not attack_counts:
        return

    sorted_attacks = sorted(attack_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [x[0].upper() for x in sorted_attacks]
    values = [x[1] for x in sorted_attacks]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color="#0284c7"),
        )
    )

    fig.update_layout(
        title=dict(
            text="Evaluated Attack Methods", font=dict(size=16, color="#f8fafc")
        ),
        xaxis_title="Scan Count",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
    )

    st.plotly_chart(fig, use_container_width=True)
