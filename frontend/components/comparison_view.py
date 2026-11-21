"""
Assessment Comparison View Component for AdverScan Frontend.

Renders:
1. Dual scan selectors (Baseline vs Comparison)
2. Compatibility verification & context matching
3. Executive verdict cards with delta metrics (Δ ASR, Δ Robust Accuracy)
4. Side-by-side comparative tables for attacks, defenses, and vulnerability scores
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import streamlit as st

from frontend.services.assessment_comparison_service import compare_assessments
from frontend.services.assessment_history_service import get_scans_for_selection
from frontend.utils.session_state import get_state, set_state


def render_comparison_view() -> None:
    """Render the primary Assessment Comparison UI."""
    st.markdown(
        """
        <div class="hero-banner" style="margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 16px;">
                <span style="font-size: 2.5rem;">⚖️</span>
                <div>
                    <h1 style="margin: 0; font-size: 1.8rem; font-weight: 700;">Assessment Comparison Engine</h1>
                    <p style="margin: 0.3rem 0 0 0; font-size: 0.95rem;">
                        Contrast adversarial robustness, defense recovery, and vulnerability posture between two assessments.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    available_scans = get_scans_for_selection()

    if len(available_scans) < 2:
        st.markdown(
            """
            <div class="modern-card" style="text-align: center; padding: 2.5rem 1rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                <div style="font-size: 1.1rem; font-weight: 600;">At Least Two Assessments Required</div>
                <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.25rem;">
                    You need at least 2 completed assessments in the database to run a comparative evaluation.
                    Execute another attack assessment or defense run to compare results.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    scan_ids = [s["scan_id"] for s in available_scans]
    scan_labels = {s["scan_id"]: s["display_label"] for s in available_scans}

    # Pre-selection from session state if set from History page
    default_base_id = get_state("comparison_baseline_scan_id") or scan_ids[0]
    if default_base_id not in scan_ids:
        default_base_id = scan_ids[0]

    default_base_idx = scan_ids.index(default_base_id)
    default_comp_idx = 1 if len(scan_ids) > 1 and default_base_idx == 0 else 0

    col_base, col_comp = st.columns(2)
    with col_base:
        st.markdown("#### 1️⃣ Baseline Assessment (Reference)")
        selected_base_id = st.selectbox(
            "Select Baseline Assessment",
            scan_ids,
            index=default_base_idx,
            format_func=lambda s: scan_labels.get(s, s),
            key="select_comp_base",
            label_visibility="collapsed",
        )

    with col_comp:
        st.markdown("#### 2️⃣ Comparison Assessment (Candidate / Defended)")
        # Filter out identical scan if possible, or allow user choice
        selected_comp_id = st.selectbox(
            "Select Comparison Assessment",
            scan_ids,
            index=default_comp_idx,
            format_func=lambda s: scan_labels.get(s, s),
            key="select_comp_cand",
            label_visibility="collapsed",
        )

    if selected_base_id == selected_comp_id:
        st.warning("⚠️ You have selected the same scan for both baseline and comparison. Please select two distinct assessments.")
        return

    # Run Comparison Engine
    res = compare_assessments(selected_base_id, selected_comp_id)
    if not res.get("success"):
        st.error(res.get("error", "Error executing comparison."))
        return

    models = res.get("models", {})
    summary = res.get("summary", {})
    attack_comps = res.get("attack_comparisons", [])
    defense_comp = res.get("defense_comparison", {})
    vuln_comp = res.get("vulnerability_comparison", {})

    # Compatibility Warning
    if not models.get("modality_match"):
        st.warning(
            f"⚠️ **Modality Mismatch**: Baseline is `{models['baseline']['modality']}` "
            f"while Comparison is `{models['comparison']['modality']}`. Direct quantitative comparison may be invalid."
        )

    # 1. Executive Verdict Banner
    verdict_badge = summary.get("verdict_badge", "neutral")
    accent_class = {
        "success": "card-accent-green",
        "danger": "card-accent-red",
        "neutral": "card-accent-cyan",
    }.get(verdict_badge, "card-accent-cyan")

    st.markdown(
        f"""
        <div class="modern-card {accent_class}" style="padding: 1.25rem 1.5rem; margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div class="stat-label">Executive Comparison Verdict</div>
                    <div style="font-size: 1.4rem; font-weight: 700; margin: 0.2rem 0;">
                        {summary.get('overall_verdict')}
                    </div>
                </div>
            </div>
            <div style="font-size: 0.92rem; color: var(--text-secondary); margin-top: 0.5rem;">
                {summary.get('verdict_summary')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Key Delta Metrics
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    avg_asr_delta = summary.get("avg_asr_delta", 0.0)
    avg_adv_acc_delta = summary.get("avg_adv_acc_delta", 0.0)

    with c_m1:
        st.metric(
            "Average ASR Delta",
            f"{avg_asr_delta:+.1f}%",
            delta=f"{avg_asr_delta:+.1f}%",
            delta_color="inverse",  # Lower ASR is better
            help="Difference in Attack Success Rate: negative indicates attack is less successful on comparison",
        )
    with c_m2:
        st.metric(
            "Robust Accuracy Delta",
            f"{avg_adv_acc_delta:+.1f}%",
            delta=f"{avg_adv_acc_delta:+.1f}%",
            delta_color="normal",  # Higher accuracy is better
            help="Difference in Adversarial Accuracy under attack: positive indicates higher accuracy",
        )
    with c_m3:
        matched_cnt = summary.get("matched_attacks_count", 0)
        tot_cnt = summary.get("total_attacks_count", 0)
        st.metric("Matched Attacks", f"{matched_cnt} of {tot_cnt}")
    with c_m4:
        vuln_delta = vuln_comp.get("score_delta", 0.0)
        st.metric(
            "Vulnerability Score Delta",
            f"{vuln_delta:+.1f}",
            delta=f"{vuln_delta:+.1f}",
            delta_color="inverse",
            help="Difference in mean vulnerability score",
        )

    # 3. Model Metadata Comparison
    with st.expander("📌 Model Architecture & Metadata Comparison", expanded=False):
        c_base_info, c_comp_info = st.columns(2)
        with c_base_info:
            st.markdown(f"**Baseline (`{selected_base_id}`)**")
            st.write(f"• Model: `{models['baseline']['model_name']}`")
            st.write(f"• Architecture: `{models['baseline'].get('architecture') or 'N/A'}`")
            b_params = models['baseline'].get('total_parameters')
            st.write(f"• Total Parameters: `{b_params:,}`" if isinstance(b_params, (int, float)) else f"• Total Parameters: `{b_params or 'N/A'}`")
            st.write(f"• Framework: `{models['baseline'].get('framework') or 'N/A'}`")
            st.write(f"• Modality: `{models['baseline']['modality']}`")
            st.write(f"• Dataset: `{models['baseline']['dataset_name']}`")
            st.write(f"• Evaluated: {models['baseline']['created_at']}")
        with c_comp_info:
            st.markdown(f"**Comparison (`{selected_comp_id}`)**")
            st.write(f"• Model: `{models['comparison']['model_name']}`")
            st.write(f"• Architecture: `{models['comparison'].get('architecture') or 'N/A'}`")
            c_params = models['comparison'].get('total_parameters')
            st.write(f"• Total Parameters: `{c_params:,}`" if isinstance(c_params, (int, float)) else f"• Total Parameters: `{c_params or 'N/A'}`")
            st.write(f"• Framework: `{models['comparison'].get('framework') or 'N/A'}`")
            st.write(f"• Modality: `{models['comparison']['modality']}`")
            st.write(f"• Dataset: `{models['comparison']['dataset_name']}`")
            st.write(f"• Evaluated: {models['comparison']['created_at']}")

    # 4. Attack-by-Attack Side-by-Side Comparison Table
    st.markdown("### ⚔️ Attack-by-Attack Telemetry & Hyperparameters Comparison")
    if not attack_comps:
        st.info("No attacks found in either assessment.")
    else:
        for atk in attack_comps:
            atk_name = atk.get("attack_name", "Attack")
            matched = atk.get("matched", False)

            if matched:
                delta_asr = atk.get("delta_asr", 0.0)
                delta_adv_acc = atk.get("delta_adv_acc", 0.0)
                eval_status = atk.get("eval_status", "neutral")

                status_badge = {
                    "improved": ("🛡️ Improved", "var(--stat-val-green)"),
                    "degraded": ("⚠️ Degraded", "var(--stat-val-red)"),
                    "neutral": ("⚖️ Neutral", "var(--text-secondary)"),
                }.get(eval_status, ("⚖️ Neutral", "var(--text-secondary)"))

                with st.container():
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.5rem;">
                            <span style="font-weight: 700; font-size: 1.05rem;">{atk_name}</span>
                            <span style="font-weight: 700; color: {status_badge[1]};">{status_badge[0]}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    c_a1, c_a2, c_a3, c_a4 = st.columns(4)
                    c_a1.metric(
                        "ASR (Baseline → Comp)",
                        f"{atk.get('baseline_asr', 0.0):.1f}% → {atk.get('comparison_asr', 0.0):.1f}%",
                        delta=f"{delta_asr:+.1f}%",
                        delta_color="inverse",
                    )
                    c_a2.metric(
                        "Adversarial Acc",
                        f"{atk.get('baseline_adv_acc', 0.0):.1f}% → {atk.get('comparison_adv_acc', 0.0):.1f}%",
                        delta=f"{delta_adv_acc:+.1f}%",
                        delta_color="normal",
                    )
                    c_a3.metric(
                        "Clean Acc",
                        f"{atk.get('baseline_clean_acc', 0.0):.1f}% → {atk.get('comparison_clean_acc', 0.0):.1f}%",
                        delta=f"{atk.get('delta_clean_acc', 0.0):+.1f}%",
                        delta_color="normal",
                    )
                    delta_time = atk.get("delta_time_seconds", 0.0)
                    c_a4.metric(
                        "Execution Time",
                        f"{atk.get('baseline_time_seconds', 0.0):.2f}s → {atk.get('comparison_time_seconds', 0.0):.2f}s",
                        delta=f"{delta_time:+.2f}s",
                        delta_color="inverse",
                    )

                    # Attack Parameters & Distortion Norms Side-by-Side
                    param_diffs = atk.get("parameter_differences", {})
                    b_p = atk.get("baseline_parameters", {})
                    b_samples = atk.get("baseline_samples", 0)
                    c_samples = atk.get("comparison_samples", 0)

                    st.markdown(
                        f"""
                        <div style="font-size: 0.85rem; color: var(--text-secondary); margin: 0.35rem 0;">
                            <strong>Samples Evaluated:</strong> Baseline: <code>{b_samples}</code> vs Comparison: <code>{c_samples}</code>
                            &nbsp;|&nbsp;
                            <strong>Distortion (L_inf):</strong> Baseline: <code>{atk.get('baseline_linf') or 'N/A'}</code> vs Comparison: <code>{atk.get('comparison_linf') or 'N/A'}</code>
                            &nbsp;|&nbsp;
                            <strong>L_2:</strong> Baseline: <code>{atk.get('baseline_l2') or 'N/A'}</code> vs Comparison: <code>{atk.get('comparison_l2') or 'N/A'}</code>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if param_diffs:
                        diff_items = " • ".join([f"<strong>{k}:</strong> <code>{v['baseline']} → {v['comparison']}</code>" for k, v in param_diffs.items()])
                        st.markdown(
                            f"""
                            <div style="background: rgba(245, 158, 11, 0.1); border-left: 3px solid #f59e0b; padding: 0.35rem 0.6rem; border-radius: 4px; font-size: 0.82rem; margin-top: 0.3rem;">
                                ⚙️ <strong>Configured Parameter Variations:</strong> {diff_items}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    elif b_p:
                        p_str = ", ".join([f"{k}={v}" for k, v in b_p.items()])
                        st.caption(f"⚙️ Attack Parameters: {p_str}")

                    st.markdown("---")
            else:
                pres = atk.get("present_in", "")
                st.caption(f"ℹ️ **{atk_name}** was only evaluated in the {pres.replace('_', ' ')} scan.")

    # 5. Defense Hardening Contrast (if present)
    if defense_comp.get("baseline_defenses") or defense_comp.get("comparison_defenses"):
        st.markdown("### 🛡️ Defense Hardening Telemetry")
        c_def_b, c_def_c = st.columns(2)
        with c_def_b:
            st.markdown(f"**Baseline Defenses ({len(defense_comp.get('baseline_defenses', []))})**")
            for d in defense_comp.get("baseline_defenses", []):
                st.write(f"• **{d.get('defense_name')}**: ASR {d.get('asr_before')}% → {d.get('asr_after')}% (Rec: {d.get('accuracy_recovery')}%)")
        with c_def_c:
            st.markdown(f"**Comparison Defenses ({len(defense_comp.get('comparison_defenses', []))})**")
            for d in defense_comp.get("comparison_defenses", []):
                st.write(f"• **{d.get('defense_name')}**: ASR {d.get('asr_before')}% → {d.get('asr_after')}% (Rec: {d.get('accuracy_recovery')}%)")

