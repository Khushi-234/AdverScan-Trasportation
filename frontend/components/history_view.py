"""
Assessment History View Component for AdverScan Frontend.

Renders:
1. Database diagnostics banner & summary metrics
2. Search, filter, and pagination controls
3. History data table with status badges and action triggers
4. Drill-down assessment inspection drawer/tabs for past evaluations
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import streamlit as st

from frontend.services.assessment_history_service import (
    delete_scan,
    get_database_status,
    get_scan_detail,
    get_scan_history,
    get_storage_repository,
)
from app.storage.db import get_db_session
from app.storage.models import ModelRecord, DatasetRecord
from frontend.services.ingestion_service import MODEL_PRESETS
from frontend.utils.session_state import set_state, get_state, set_workflow_stage


def render_history_view() -> None:
    """Render the primary Assessment History and Database Management UI."""
    st.markdown(
        """
        <div class="hero-banner" style="margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 16px;">
                <span style="font-size: 2.5rem;">📜</span>
                <div>
                    <h1 style="margin: 0; font-size: 1.8rem; font-weight: 700;">Assessment History & Central Records</h1>
                    <p style="margin: 0.3rem 0 0 0; font-size: 0.95rem;">
                        Audit, inspect, and track historical robustness evaluations stored in the persistent SQLite database.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Database Diagnostics & System Telemetry
    stats = get_database_status()
    db_exists = stats.get("database_exists", False)
    db_size = stats.get("database_size_mb", 0.0)
    total_scans = stats.get("total_assessments", 0)
    completed_scans = stats.get("completed_assessments", 0)
    failed_scans = stats.get("failed_assessments", 0)
    running_scans = stats.get("running_assessments", 0)
    total_attacks = stats.get("total_attacks_recorded", 0)

    col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
    with col_stat1:
        st.markdown(
            f"""
            <div class="modern-card card-accent-green" style="padding: 0.75rem 1rem;">
                <div class="stat-label">Database Status</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: var(--stat-val-green);">
                    {'🟢 Connected' if db_exists else '🔴 Offline'}
                </div>
                <div style="font-size: 0.75rem; color: var(--text-secondary);">WAL Mode • {db_size} MB</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_stat2:
        st.markdown(
            f"""
            <div class="modern-card card-accent-cyan" style="padding: 0.75rem 1rem;">
                <div class="stat-label">Total Scans</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: var(--stat-val-cyan);">{total_scans}</div>
                <div style="font-size: 0.75rem; color: var(--text-secondary);">Historical Runs</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_stat3:
        st.markdown(
            f"""
            <div class="modern-card card-accent-blue" style="padding: 0.75rem 1rem;">
                <div class="stat-label">Completed</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: var(--stat-val-blue);">{completed_scans}</div>
                <div style="font-size: 0.75rem; color: var(--text-secondary);">{running_scans} active/running</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_stat4:
        st.markdown(
            f"""
            <div class="modern-card card-accent-purple" style="padding: 0.75rem 1rem;">
                <div class="stat-label">Attacks Recorded</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: var(--stat-val-purple);">{total_attacks}</div>
                <div style="font-size: 0.75rem; color: var(--text-secondary);">Evaluated vectors</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_stat5:
        st.markdown(
            f"""
            <div class="modern-card {'card-accent-red' if failed_scans > 0 else 'card-accent-blue'}" style="padding: 0.75rem 1rem;">
                <div class="stat-label">Failed / Errored</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: {'var(--stat-val-red)' if failed_scans > 0 else 'var(--text-secondary)'};">{failed_scans}</div>
                <div style="font-size: 0.75rem; color: var(--text-secondary);">Unfinished assessments</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # 2. Search & Filter Bar
    col_search, col_mod, col_stat = st.columns([3, 1, 1])
    with col_search:
        search_query = st.text_input(
            "Search Scans",
            placeholder="Search by Model Name (e.g. ResNet, BERT, PatchTSMixer)...",
            label_visibility="collapsed",
        )
    with col_mod:
        modality_filter = st.selectbox(
            "Modality",
            ["All Modalities", "Image", "Text", "Tabular", "Time-Series"],
            label_visibility="collapsed",
        )
    with col_stat:
        status_filter = st.selectbox(
            "Status",
            ["All Statuses", "Completed", "Running", "Failed"],
            label_visibility="collapsed",
        )

    # Resolve query parameters
    mod_param = None
    if modality_filter != "All Modalities":
        mod_param = modality_filter.lower().replace("-", "_")

    stat_param = None
    if status_filter != "All Statuses":
        stat_param = status_filter.lower()

    # Fetch records
    scans = get_scan_history(
        status=stat_param,
        model_name=search_query.strip() if search_query.strip() else None,
        modality=mod_param,
        limit=50,
    )

    if not scans:
        st.markdown(
            """
            <div class="modern-card" style="text-align: center; padding: 2.5rem 1rem; margin-top: 1rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📭</div>
                <div style="font-size: 1.1rem; font-weight: 600;">No Historical Assessments Found</div>
                <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.25rem;">
                    Completed assessments will automatically appear here once evaluated through the pipeline.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # 3. Assessment Table
    st.markdown("### 📋 Evaluated Scans")

    selected_scan_id = get_state("history_inspected_scan_id")

    for scan in scans:
        s_id = scan.get("scan_id", "")
        s_name = scan.get("assessment_name") or s_id
        m_name = scan.get("model_name") or "Unknown Model"
        d_name = scan.get("dataset_name") or "N/A"
        modality = (scan.get("modality") or "general").capitalize()
        status = scan.get("status", "completed")
        created = str(scan.get("created_at"))[:19]
        attacks = scan.get("attacks", [])
        defenses = scan.get("defenses", [])
        vulns = scan.get("vulnerabilities", [])
        xai_res = scan.get("xai_results", [])
        reports = scan.get("reports", [])

        status_color = {
            "completed": "var(--stat-val-green)",
            "running": "var(--stat-val-cyan)",
            "failed": "var(--stat-val-red)",
            "partially_completed": "var(--stat-val-orange, #f59e0b)",
        }.get(status, "var(--text-secondary)")

        with st.container():
            c_icon, c_info, c_metrics, c_actions = st.columns([1, 4, 3, 2])
            with c_icon:
                dom_icons = {"Image": "🖼️", "Text": "📝", "Tabular": "📊", "Time_series": "📈"}
                st.markdown(
                    f"<div style='font-size: 2rem; text-align: center; padding-top: 0.5rem;'>{dom_icons.get(modality, '🛡️')}</div>",
                    unsafe_allow_html=True,
                )
            with c_info:
                st.markdown(
                    f"""
                    <div style="font-weight: 700; font-size: 1.05rem; color: var(--text-primary);">
                        {m_name} <span style="font-size: 0.8rem; font-weight: 600; color: var(--text-secondary);">({modality})</span>
                    </div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary);">
                        Scan ID: <code style="font-weight: 600;">{s_id}</code> • Dataset: <strong>{d_name}</strong> • {created}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c_metrics:
                st.markdown(
                    f"""
                    <div style="font-size: 0.85rem; margin-top: 0.2rem;">
                        Status: <strong style="color: {status_color};">{status.capitalize()}</strong>
                    </div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary);">
                        Attacks: <strong>{len(attacks)}</strong> | Defenses: <strong>{len(defenses)}</strong> | Vulns: <strong>{len(vulns)}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c_actions:
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    is_current_inspect = (selected_scan_id == s_id)
                    btn_label = "Hide" if is_current_inspect else "Inspect"
                    if st.button(btn_label, key=f"inspect_{s_id}", use_container_width=True):
                        set_state("history_inspected_scan_id", None if is_current_inspect else s_id)
                        st.rerun()
                with col_btn2:
                    if st.button("Compare", key=f"comp_btn_{s_id}", use_container_width=True):
                        set_state("comparison_baseline_scan_id", s_id)
                        set_workflow_stage("comparison")
                        st.rerun()

            st.markdown("---")

    # 4. Detailed Drill-Down Drawer (if an assessment is selected)
    if selected_scan_id:
        render_assessment_detail_drawer(selected_scan_id)


def render_assessment_detail_drawer(scan_id: str) -> None:
    """Render full hierarchical breakdown tabs for a selected assessment."""
    detail = get_scan_detail(scan_id)
    if not detail:
        st.error(f"Could not load scan details for '{scan_id}'.")
        return

    st.markdown(f"## 🔍 Assessment Drill-Down: `{scan_id}`")
    st.caption(f"Full historical audit for {detail.get('model_name', 'Model')} on {detail.get('dataset_name', 'Dataset')}")

    tab_overview, tab_attacks, tab_defenses, tab_xai, tab_vulns, tab_report = st.tabs([
        "Model & Dataset",
        f"Attacks ({len(detail.get('attacks', []))})",
        f"Defenses ({len(detail.get('defenses', []))})",
        f"XAI ({len(detail.get('xai_results', []))})",
        f"Vulnerabilities ({len(detail.get('vulnerabilities', []))})",
        "Report",
    ])

    with tab_overview:
        col_m, col_d = st.columns(2)
        with col_m:
            st.markdown("#### Model Specification")
            m_rec = detail.get("model") or {}
            if not m_rec and detail.get("model_id"):
                try:
                    with get_db_session(get_storage_repository().db_path) as s:
                        m_row = s.query(ModelRecord).filter_by(model_id=detail.get("model_id")).first()
                        if m_row:
                            m_rec = m_row.to_dict()
                except Exception:
                    pass

            meta_json = m_rec.get("metadata_json") or {}
            cfg_model = detail.get("configuration", {}).get("model_info", {})
            preset_m = MODEL_PRESETS.get(detail.get("model_id"), {}) or MODEL_PRESETS.get(detail.get("model_name"), {})
            arch = (
                m_rec.get("architecture")
                or meta_json.get("Architecture")
                or meta_json.get("architecture")
                or cfg_model.get("architecture")
                or preset_m.get("model_type")
                or "N/A"
            )
            framework = (
                m_rec.get("framework")
                or meta_json.get("Framework")
                or meta_json.get("framework")
                or cfg_model.get("framework")
                or preset_m.get("framework")
                or "N/A"
            )
            total_params = (
                m_rec.get("total_parameters")
                or meta_json.get("Total Parameters")
                or meta_json.get("total_parameters")
                or cfg_model.get("total_parameters")
                or preset_m.get("parameters")
            )
            classes_count = (
                m_rec.get("num_classes")
                or meta_json.get("Total Classes")
                or meta_json.get("num_classes")
                or cfg_model.get("num_classes")
                or preset_m.get("num_classes")
                or "N/A"
            )
            param_str = (
                f"{total_params:,}"
                if isinstance(total_params, (int, float))
                else (str(total_params) if total_params is not None else "N/A")
            )

            st.write(f"**Model ID:** `{detail.get('model_id', 'N/A')}`")
            st.write(f"**Model Name:** {detail.get('model_name', 'N/A')}")
            st.write(f"**Architecture:** {arch}")
            st.write(f"**Framework:** {framework}")
            st.write(f"**Total Parameters:** {param_str}")
            st.write(f"**Classes Count:** {classes_count}")

        with col_d:
            st.markdown("#### Dataset Specification")
            d_rec = detail.get("dataset") or {}
            if not d_rec and detail.get("dataset_id"):
                try:
                    with get_db_session(get_storage_repository().db_path) as s:
                        d_row = s.query(DatasetRecord).filter_by(dataset_id=detail.get("dataset_id")).first()
                        if d_row:
                            d_rec = d_row.to_dict()
                except Exception:
                    pass

            d_meta = d_rec.get("metadata_json") or {}
            cfg_data = detail.get("configuration", {}).get("dataset_info", {})
            tot_samples = (
                d_rec.get("number_of_samples")
                or d_meta.get("Total Dataset Samples")
                or d_meta.get("total_samples")
                or cfg_data.get("number_of_samples")
            )
            sample_str = (
                f"{tot_samples:,}"
                if isinstance(tot_samples, (int, float))
                else (str(tot_samples) if tot_samples is not None else "N/A")
            )
            split = d_rec.get("data_split") or d_meta.get("split") or cfg_data.get("data_split") or "Test"

            st.write(f"**Dataset ID:** `{detail.get('dataset_id', 'N/A')}`")
            st.write(f"**Dataset Name:** {detail.get('dataset_name', 'N/A')}")
            st.write(f"**Modality:** {detail.get('modality', 'N/A')}")
            st.write(f"**Total Samples:** {sample_str}")
            st.write(f"**Split:** {split}")

    with tab_attacks:
        attacks = detail.get("attacks", [])
        if not attacks:
            st.info("No attacks recorded for this assessment.")
        else:
            for atk in attacks:
                atk_name = atk.get("attack_name", "Attack")
                with st.expander(f"⚔️ {atk_name} (ASR: {atk.get('attack_success_rate', 0.0)}%)", expanded=True):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Clean Accuracy", f"{atk.get('clean_accuracy', 0.0):.1f}%")
                    c2.metric("Adversarial Accuracy", f"{atk.get('adversarial_accuracy', 0.0):.1f}%")
                    c3.metric("Accuracy Drop", f"{atk.get('accuracy_drop', 0.0):.1f}%")
                    c4.metric("Attack Success Rate", f"{atk.get('attack_success_rate', 0.0):.1f}%")

                    # Configured Attack Parameters
                    atk_params = atk.get("attack_parameters") or {}
                    st.markdown("**⚙️ Configured Attack Parameters:**")
                    if atk_params:
                        param_pills = " • ".join([f"<code>{k}: {v}</code>" for k, v in atk_params.items()])
                        st.markdown(f"<div style='font-size: 0.85rem; margin-bottom: 0.5rem;'>{param_pills}</div>", unsafe_allow_html=True)
                    else:
                        st.caption("Standard / Default hyperparameters")

                    # Execution Time & Sample Telemetry
                    exec_s = atk.get('execution_time_seconds', 0.0)
                    tot_s = atk.get('total_samples', 0)
                    ms_sample = f"{(exec_s * 1000.0 / tot_s):.1f} ms/sample" if tot_s and exec_s else "N/A"

                    st.markdown("**⏱️ Performance & Distortion Telemetry:**")
                    st.write(
                        f"• Execution Time: `{exec_s:.2f}s` (`{ms_sample}`) | "
                        f"Samples Evaluated: `{tot_s}` | "
                        f"$L_\\infty$: `{atk.get('l_inf_norm')}` | $L_2$: `{atk.get('l2_norm')}` | "
                        f"$L_0$: `{atk.get('l0_norm')}` | $\\text{{MAE}}$: `{atk.get('mae')}`"
                    )
                    if atk.get("adversarial_samples_path"):
                        st.caption(f"💾 Adversarial Tensor Artifact: `{atk.get('adversarial_samples_path')}`")

    with tab_defenses:
        defenses = detail.get("defenses", [])
        if not defenses:
            st.info("No defenses applied for this assessment.")
        else:
            for d in defenses:
                raw_name = d.get("defense_name", "defense")
                formatted_name = raw_name.replace("_", " ").title()
                family = d.get("defense_family") or "Preprocessing"
                stage_str = d.get("defense_stage") or "Inference Transformation"
                rob_status = d.get("robustness_status") or ("Improved" if d.get("is_improved") else "Evaluated")
                val_status = str(d.get("validation_status") or "Accepted").upper()

                with st.expander(f"🛡️ {formatted_name} — [{family}] ({rob_status})", expanded=True):
                    # Header badges
                    st.markdown(
                        f"""
                        <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.75rem;">
                            <span class="info-badge">Applied Technique: <strong>{formatted_name}</strong></span>
                            <span class="info-badge">Family: <strong>{family}</strong></span>
                            <span class="info-badge">Stage: <strong>{stage_str}</strong></span>
                            <span class="info-badge">Status: <strong>{val_status}</strong></span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Diagnostic alerts for harmful / robust status
                    if d.get("clean_accuracy_drop", 0.0) >= 10.0 or "HARMFUL" in str(rob_status).upper():
                        st.error("⚠️ **Notice: Harmful Defense Outcome.** Clean accuracy dropped significantly (≥10.0%), severely degrading standard model utility.")
                    elif "ALREADY ROBUST" in formatted_name.upper() or val_status == "NOT_NEEDED":
                        st.info("🛡️ **Notice: Model Already Robust.** Baseline ASR was 0.0%, meaning the attack was ineffective. No defense transformation was necessary.")

                    # 4-column metric grid
                    c1, c2, c3, c4 = st.columns(4)
                    c_before = d.get("clean_accuracy_before", 0.0)
                    c_after = d.get("clean_accuracy_after", 0.0)
                    c_drop = d.get("clean_accuracy_drop", 0.0)
                    c1.metric(
                        "Clean Accuracy",
                        f"{c_before:.1f}% → {c_after:.1f}%",
                        f"-{c_drop:.1f}% drop" if c_drop else None,
                        delta_color="inverse",
                    )

                    a_before = d.get("adv_accuracy_before", 0.0)
                    a_after = d.get("adv_accuracy_after", 0.0)
                    a_rec = d.get("accuracy_recovery", 0.0)
                    c2.metric(
                        "Adversarial Accuracy",
                        f"{a_before:.1f}% → {a_after:.1f}%",
                        f"{a_rec:+.1f}%" if a_rec else None,
                    )

                    asr_b = d.get("asr_before", 0.0)
                    asr_a = d.get("asr_after", 0.0)
                    asr_red = d.get("asr_reduction", 0.0)
                    c3.metric(
                        "Attack Success Rate (ASR)",
                        f"{asr_b:.1f}% → {asr_a:.1f}%",
                        f"-{asr_red:.1f}%" if asr_red else None,
                        delta_color="inverse",
                    )

                    c4.metric(
                        "ASR Suppression",
                        f"{asr_red:.1f}%",
                        f"Recovery: {a_rec:+.1f}%",
                    )

                    # Configuration Hyperparameters
                    params = d.get("defense_configuration") or {}
                    st.markdown("**⚙️ Configured Defense Hyperparameters:**")
                    if params:
                        p_items = " • ".join([f"<code>{k}: {v}</code>" for k, v in params.items()])
                        st.markdown(f"<div style='font-size: 0.85rem; margin-bottom: 0.5rem;'>{p_items}</div>", unsafe_allow_html=True)
                    else:
                        st.caption("Standard / Default hyperparameters applied")

                    # Performance telemetry
                    exec_s = d.get("execution_time_seconds", 0.0)
                    if exec_s:
                        st.caption(f"⏱️ Defense Execution Time: `{exec_s:.2f}s`")
                    if d.get("hardened_samples_path"):
                        st.caption(f"💾 Hardened Artifact: `{d.get('hardened_samples_path')}`")

    with tab_xai:
        xais = detail.get("xai_results", [])
        if not xais:
            st.info("No XAI explanations recorded for this assessment.")
        else:
            for x in xais:
                tech = x.get("explainer_technique", "Explainer").upper()
                with st.expander(f"💡 {tech} Analysis", expanded=True):
                    st.write(f"**Clean Prediction:** `{x.get('clean_prediction')}` (Conf: {x.get('clean_confidence')})")
                    st.write(f"**Adversarial Prediction:** `{x.get('adversarial_prediction')}` (Conf: {x.get('adversarial_confidence')})")
                    st.write(f"**Attribution Drift:** `{x.get('attribution_drift')}`")
                    if x.get("root_cause_narrative"):
                        st.info(x.get("root_cause_narrative"))

    with tab_vulns:
        vulns = detail.get("vulnerabilities", [])
        recs = detail.get("recommendations", [])
        if not vulns and not recs:
            st.info("No vulnerability findings recorded.")
        else:
            if vulns:
                st.markdown("#### 🚨 Diagnostic Vulnerability Findings")
                for v in vulns:
                    sev = str(v.get("severity", "Medium")).upper()
                    sev_colors = {"CRITICAL": "#ef4444", "HIGH": "#f97316", "MEDIUM": "#eab308", "LOW": "#22c55e"}
                    col_bar = sev_colors.get(sev, "#3b82f6")
                    v_score = v.get("vulnerability_score", 0.0)

                    st.markdown(
                        f"""
                        <div style="border-left: 4px solid {col_bar}; padding: 0.6rem 0.8rem; background: var(--card-bg, rgba(255,255,255,0.03)); border-radius: 4px; margin-bottom: 0.6rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <strong style="font-size: 0.95rem;">{v.get('vulnerability_name')}</strong>
                                <span style="font-size: 0.8rem; padding: 2px 8px; border-radius: 4px; background: {col_bar}22; color: {col_bar}; font-weight: 700;">{sev} • {v_score:.1f}/100</span>
                            </div>
                            <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.25rem;">
                                {v.get('description') or 'Diagnostic vulnerability finding identified during adversarial testing.'}
                            </div>
                            {f"<div style='font-size: 0.8rem; color: var(--text-muted); margin-top: 0.2rem;'>Region: <code>{v.get('affected_feature_or_region')}</code></div>" if v.get('affected_feature_or_region') else ''}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            if recs:
                st.markdown("#### 🛡️ Actionable Hardening Recommendations")
                for r in recs:
                    fam = r.get("defense_family") or "Mitigation"
                    conf = r.get("confidence")
                    conf_str = f" • Confidence: {conf * 100:.0f}%" if isinstance(conf, float) and 0 < conf <= 1.0 else (f" • Confidence: {conf}%" if conf else "")
                    st.markdown(
                        f"""
                        <div style="border-left: 4px solid #10b981; padding: 0.6rem 0.8rem; background: var(--card-bg, rgba(255,255,255,0.03)); border-radius: 4px; margin-bottom: 0.6rem;">
                            <div style="font-weight: 700; font-size: 0.95rem; color: var(--text-primary);">
                                💡 {r.get('recommended_technique', 'Defense Hardening')} <span style="font-size: 0.8rem; font-weight: 500; color: var(--text-secondary);">[{fam}{conf_str}]</span>
                            </div>
                            <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.25rem;">
                                {r.get('rationale') or 'Apply this mitigation to restore adversarial robustness.'}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    with tab_report:
        reports = detail.get("reports", [])
        if not reports:
            st.info("No compiled report generated for this scan.")
        else:
            for rep in reports:
                r_name = rep.get("report_name", "Report")
                r_path = rep.get("report_path", "")
                st.write(f"📄 **{r_name}** ({rep.get('report_format', 'pdf').upper()})")
                st.caption(f"Path: `{r_path}` • Generated: {rep.get('generated_at')}")
                if Path(r_path).is_file():
                    with open(r_path, "rb") as f:
                        st.download_button(
                            label=f"⬇️ Download {r_name}",
                            data=f.read(),
                            file_name=Path(r_path).name,
                            key=f"dl_rep_{rep.get('report_id')}",
                        )

