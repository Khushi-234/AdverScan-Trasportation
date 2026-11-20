"""
Modality-Independent Dynamic XAI Visualization Component for AdverScan.
Renders feature attribution, spatial heatmaps, token highlights, temporal curves,
and pre- vs. post-defense explanation comparisons dynamically based on returned data structures.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def _to_numpy_array(data: Any) -> Optional[np.ndarray]:
    """Helper to convert tensor, list, or nested structures to a numpy array safely."""
    if data is None:
        return None
    try:
        if hasattr(data, "detach"):
            data = data.detach().cpu().numpy()
        elif hasattr(data, "cpu"):
            data = data.cpu().numpy()
        arr = np.asarray(data, dtype=np.float32)
        return arr
    except Exception:
        return None


def render_xai_metrics_summary(xai_result: Dict[str, Any]) -> None:
    """Render high-level metric cards for XAI diagnostics."""
    tech = str(xai_result.get("technique", "attribution")).upper()
    c_pred = xai_result.get("clean_prediction")
    a_pred = xai_result.get("adversarial_prediction")
    c_conf = xai_result.get("clean_confidence")
    a_conf = xai_result.get("adversarial_confidence")
    pred_changed = xai_result.get("prediction_changed")
    drift = xai_result.get("local_explanation", {}).get("attribution_drift")
    if drift is None:
        drift = xai_result.get("attribution_drift")
    distortion = xai_result.get("local_explanation", {}).get("feature_salience_distortion")
    if distortion is None:
        distortion = xai_result.get("feature_salience_distortion")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="modern-card card-accent-cyan" style="padding: 10px 14px;">
                <div class="stat-label">Explainer Technique</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: var(--stat-val-cyan);">{tech}</div>
                <div class="stat-subtext">M6 Root-Cause Engine</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        c_conf_str = f"({c_conf*100:.1f}%)" if isinstance(c_conf, (int, float)) and c_conf <= 1.0 else (f"({c_conf:.1f}%)" if c_conf else "")
        c_pred_disp = f"Class {c_pred}" if c_pred is not None else "N/A"
        st.markdown(
            f"""
            <div class="modern-card card-accent-emerald" style="padding: 10px 14px;">
                <div class="stat-label">Clean Prediction</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: var(--stat-val-emerald);">{c_pred_disp} {c_conf_str}</div>
                <div class="stat-subtext">Baseline Unperturbed Input</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        a_conf_str = f"({a_conf*100:.1f}%)" if isinstance(a_conf, (int, float)) and a_conf <= 1.0 else (f"({a_conf:.1f}%)" if a_conf else "")
        a_pred_disp = f"Class {a_pred}" if a_pred is not None else "N/A"
        if c_pred is None or a_pred is None or pred_changed is None:
            color_class = "var(--text-muted, #94a3b8)"
            card_class = "card-accent-cyan"
            pred_subtext = "N/A (Indeterminate)"
        elif pred_changed:
            color_class = "var(--stat-val-rose)"
            card_class = "card-accent-rose"
            pred_subtext = "⚠️ Invariant Violated"
        else:
            color_class = "var(--stat-val-emerald)"
            card_class = "card-accent-emerald"
            pred_subtext = "✅ Invariant Preserved"

        st.markdown(
            f"""
            <div class="modern-card {card_class}" style="padding: 10px 14px;">
                <div class="stat-label">Adversarial Prediction</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: {color_class};">{a_pred_disp} {a_conf_str}</div>
                <div class="stat-subtext">{pred_subtext}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        drift_disp = f"{drift:.4f}" if isinstance(drift, (int, float)) else "N/A"
        if isinstance(distortion, (int, float)):
            distortion_subtext = f"Salience Distortion: {distortion:.4f}"
        elif isinstance(drift, (int, float)):
            distortion_subtext = "Normalized Attribution Drift (L1)"
        else:
            distortion_subtext = "Feature Salience Distortion: N/A"

        st.markdown(
            f"""
            <div class="modern-card card-accent-amber" style="padding: 10px 14px;">
                <div class="stat-label">Attribution Shift / Drift</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #f59e0b;">{drift_disp}</div>
                <div class="stat-subtext">{distortion_subtext}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_feature_importance_bars(xai_result: Dict[str, Any]) -> bool:
    """
    Render horizontal comparative bar charts for clean vs adversarial feature importance.
    Applicable across Tabular, Text, Audio, Time-Series, and Image patches.
    Returns True if rendered, False if no feature importance data is present.
    """
    local_exp = xai_result.get("local_explanation", {})
    feat_imp = xai_result.get("feature_importance", {})

    clean_top = local_exp.get("clean_top_features") or feat_imp.get("clean_top_features", [])
    adv_top = local_exp.get("adversarial_top_features") or feat_imp.get("adversarial_top_features", [])

    if not clean_top and not adv_top:
        return False

    st.markdown("##### 📊 Comparative Feature Importance & Vulnerability Shift")

    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.markdown("**Clean Prediction Key Attributions**")
        if clean_top:
            names = [str(f.get("feature", f.get("name", f"Feature {i}"))) for i, f in enumerate(clean_top)]
            weights = [float(f.get("weight", f.get("importance", 0.0))) for f in clean_top]

            fig_c = go.Figure(
                go.Bar(
                    x=weights,
                    y=names,
                    orientation="h",
                    marker=dict(color="#10b981"),
                    text=[f"{w:.4f}" for w in weights],
                    textposition="auto",
                )
            )
            fig_c.update_layout(
                margin=dict(l=10, r=10, t=20, b=20),
                height=max(180, len(names) * 35),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(autorange="reversed"),
                font=dict(color="#94a3b8"),
            )
            st.plotly_chart(fig_c, use_container_width=True)
        else:
            st.info("No clean feature rankings recorded.")

    with col_b2:
        st.markdown("**Adversarial Prediction Top Attributions (Exploited)**")
        if adv_top:
            names = [str(f.get("feature", f.get("name", f"Feature {i}"))) for i, f in enumerate(adv_top)]
            weights = [float(f.get("weight", f.get("importance", 0.0))) for f in adv_top]

            fig_a = go.Figure(
                go.Bar(
                    x=weights,
                    y=names,
                    orientation="h",
                    marker=dict(color="#ef4444"),
                    text=[f"{w:.4f}" for w in weights],
                    textposition="auto",
                )
            )
            fig_a.update_layout(
                margin=dict(l=10, r=10, t=20, b=20),
                height=max(180, len(names) * 35),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(autorange="reversed"),
                font=dict(color="#94a3b8"),
            )
            st.plotly_chart(fig_a, use_container_width=True)
        else:
            st.info("No adversarial feature rankings recorded.")

    return True


def render_clean_vs_adversarial_attribution_table(xai_result: Dict[str, Any]) -> bool:
    """
    Render a side-by-side clean vs. adversarial feature attribution table.
    Eliminates spurious N/A values by aligning full attribution arrays,
    contributing features from failure analysis, or spatial/feature dimensions.
    Shows N/A only when genuine data is absent.
    Returns True if table was rendered.
    """
    rows = []

    # Priority 1: contributing_features from failure analysis (full aligned array analysis)
    contrib = (
        xai_result.get("contributing_features")
        or xai_result.get("failure_analysis", {}).get("contributing_features")
    )
    if contrib and isinstance(contrib, list):
        for f in contrib:
            if not isinstance(f, dict):
                continue
            name = str(f.get("feature", f.get("name", "")))
            c_val = f.get("clean_importance")
            if c_val is None:
                c_val = f.get("clean_attribution")
            a_val = f.get("adversarial_importance")
            if a_val is None:
                a_val = f.get("adversarial_attribution")
            shift = f.get("importance_shift")
            if shift is None:
                shift = f.get("attribution_shift")
            if shift is None and c_val is not None and a_val is not None:
                try:
                    shift = float(a_val) - float(c_val)
                except Exception:
                    shift = None

            c_str = f"{float(c_val):.4f}" if c_val is not None and not (isinstance(c_val, float) and np.isnan(c_val)) else "N/A"
            a_str = f"{float(a_val):.4f}" if a_val is not None and not (isinstance(a_val, float) and np.isnan(a_val)) else "N/A"
            shift_str = f"{float(shift):+.4f}" if shift is not None and not (isinstance(shift, float) and np.isnan(shift)) else "N/A"
            rows.append({"Feature": name, "Clean": c_str, "Adversarial": a_str, "Shift (Adv − Clean)": shift_str})

    # Priority 2: Full attribution vectors when available
    if not rows:
        clean_arr = _to_numpy_array(xai_result.get("clean_attribution"))
        adv_arr = _to_numpy_array(xai_result.get("adversarial_attribution"))
        if clean_arr is not None and adv_arr is not None:
            c_flat = clean_arr.ravel()
            a_flat = adv_arr.ravel()
            if len(c_flat) == len(a_flat) and len(c_flat) > 0:
                feat_names = xai_result.get("feature_names") or []
                diff = np.abs(a_flat - c_flat)
                top_indices = np.argsort(diff)[::-1][:10]
                for idx in top_indices:
                    name = feat_names[idx] if idx < len(feat_names) else f"Feature {idx}"
                    c_val = float(c_flat[idx])
                    a_val = float(a_flat[idx])
                    shift = a_val - c_val
                    rows.append({
                        "Feature": name,
                        "Clean": f"{c_val:.4f}",
                        "Adversarial": f"{a_val:.4f}",
                        "Shift (Adv − Clean)": f"{shift:+.4f}",
                    })

    # Priority 3: Fallback to top feature lists, looking up indices from full vectors if possible
    if not rows:
        local_exp = xai_result.get("local_explanation", {})
        feat_imp = xai_result.get("feature_importance", {})

        clean_top = local_exp.get("clean_top_features") or feat_imp.get("clean_top_features", [])
        adv_top = local_exp.get("adversarial_top_features") or feat_imp.get("adversarial_top_features", [])

        if not clean_top and not adv_top:
            return False

        clean_arr = _to_numpy_array(xai_result.get("clean_attribution"))
        adv_arr = _to_numpy_array(xai_result.get("adversarial_attribution"))
        c_flat = clean_arr.ravel() if clean_arr is not None else None
        a_flat = adv_arr.ravel() if adv_arr is not None else None
        feat_names = xai_result.get("feature_names") or []

        def _build_feat_map(feats: List[Dict]) -> Dict[str, float]:
            m: Dict[str, float] = {}
            for f in feats:
                name = str(f.get("feature", f.get("name", "")))
                val = f.get("weight", f.get("importance", None))
                if name:
                    m[name] = float(val) if val is not None else float("nan")
            return m

        clean_map = _build_feat_map(clean_top)
        adv_map = _build_feat_map(adv_top)
        all_features = list(dict.fromkeys(list(clean_map.keys()) + list(adv_map.keys())))

        for feat in all_features:
            c_val = clean_map.get(feat)
            a_val = adv_map.get(feat)

            # Try to resolve missing value from full vector
            idx = None
            if feat in feat_names:
                idx = feat_names.index(feat)
            elif feat.startswith("Feature ") and feat.replace("Feature ", "").isdigit():
                idx = int(feat.replace("Feature ", ""))
            elif feat.startswith("f") and feat[1:].isdigit():
                idx = int(feat[1:])

            if (c_val is None or np.isnan(c_val)) and c_flat is not None and idx is not None and 0 <= idx < len(c_flat):
                c_val = float(c_flat[idx])
            if (a_val is None or np.isnan(a_val)) and a_flat is not None and idx is not None and 0 <= idx < len(a_flat):
                a_val = float(a_flat[idx])

            c_str = f"{c_val:.4f}" if c_val is not None and not np.isnan(c_val) else "N/A"
            a_str = f"{a_val:.4f}" if a_val is not None and not np.isnan(a_val) else "N/A"
            if c_val is not None and a_val is not None and c_str != "N/A" and a_str != "N/A":
                shift = a_val - c_val
                shift_str = f"{shift:+.4f}"
            else:
                shift_str = "N/A"
            rows.append({"Feature": feat, "Clean": c_str, "Adversarial": a_str, "Shift (Adv − Clean)": shift_str})

    if rows:
        st.markdown("##### 📋 Clean vs. Adversarial Feature Attribution")
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)
        return True
    return False


def render_spatial_heatmaps(xai_result: Dict[str, Any]) -> bool:
    """
    Render 2D attribution heatmaps for Image and Audio Spectrogram modalities.
    Returns True if 2D heatmaps were discovered and rendered, False otherwise.
    """
    raw_res = xai_result.get("_result_obj")
    attr_dict = xai_result.get("attribution") or {}
    if hasattr(raw_res, "attribution") and isinstance(raw_res.attribution, dict):
        attr_dict = raw_res.attribution

    clean_hm = _to_numpy_array(xai_result.get("clean_attribution"))
    adv_hm = _to_numpy_array(xai_result.get("adversarial_attribution"))

    if isinstance(attr_dict, dict):
        if clean_hm is None and "clean_attribution" in attr_dict:
            clean_hm = _to_numpy_array(attr_dict["clean_attribution"])
        if clean_hm is None and "clean" in attr_dict and isinstance(attr_dict["clean"], dict):
            clean_hm = _to_numpy_array(attr_dict["clean"].get("attribution"))
        if adv_hm is None and "adversarial_attribution" in attr_dict:
            adv_hm = _to_numpy_array(attr_dict["adversarial_attribution"])
        if adv_hm is None and "adversarial" in attr_dict and isinstance(attr_dict["adversarial"], dict):
            adv_hm = _to_numpy_array(attr_dict["adversarial"].get("attribution"))
        if clean_hm is None and "heatmap" in attr_dict:
            adv_hm = _to_numpy_array(attr_dict["heatmap"])

    if clean_hm is None and adv_hm is None:
        raw_arr = _to_numpy_array(attr_dict)
        if raw_arr is not None and raw_arr.ndim == 2:
            adv_hm = raw_arr

    # Normalize heatmaps to 2D
    def _ensure_2d(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            return arr
        while arr.ndim > 2:
            if arr.shape[0] in (1, 3, 4):
                arr = np.mean(arr, axis=0)
            elif arr.shape[-1] in (1, 3, 4):
                arr = np.mean(arr, axis=-1)
            else:
                arr = np.mean(arr, axis=0)
        if arr.ndim == 2:
            return arr
        return None

    clean_2d = _ensure_2d(clean_hm)
    adv_2d = _ensure_2d(adv_hm)

    if clean_2d is None and adv_2d is None:
        return False

    st.markdown("##### 🗺️ Spatial / Spectral Attribution Heatmap")
    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.markdown("**Clean Baseline Saliency Map**")
        if clean_2d is not None:
            fig_c = px.imshow(clean_2d, color_continuous_scale="Viridis", labels=dict(color="Salience"))
            fig_c.update_layout(margin=dict(l=0, r=0, t=10, b=10), height=280)
            st.plotly_chart(fig_c, use_container_width=True)
        else:
            st.info("Clean saliency heatmap not recorded.")

    with col_h2:
        st.markdown("**Adversarial Exploited Saliency Map**")
        if adv_2d is not None:
            fig_a = px.imshow(adv_2d, color_continuous_scale="Inferno", labels=dict(color="Attribution"))
            fig_a.update_layout(margin=dict(l=0, r=0, t=10, b=10), height=280)
            st.plotly_chart(fig_a, use_container_width=True)
        else:
            st.info("Adversarial saliency heatmap not recorded.")

    return True


def render_token_attribution_view(xai_result: Dict[str, Any]) -> bool:
    """
    Render highlighted text token badges with color intensity based on attribution magnitude.
    Returns True if token attribution is present and rendered, False otherwise.
    """
    raw_res = xai_result.get("_result_obj")
    attr_dict = xai_result.get("attribution") or {}
    if hasattr(raw_res, "attribution") and isinstance(raw_res.attribution, dict):
        attr_dict = raw_res.attribution

    tokens = None
    weights = None

    if isinstance(attr_dict, dict):
        tokens = attr_dict.get("tokens")
        weights = attr_dict.get("attribution") or attr_dict.get("token_attributions") or attr_dict.get("attributions")

    local_exp = xai_result.get("local_explanation", {})
    if not tokens and isinstance(local_exp, dict):
        tokens = local_exp.get("tokens")
    if weights is None and isinstance(local_exp, dict):
        weights = local_exp.get("attributions") or local_exp.get("token_attributions") or local_exp.get("attribution")

    if not tokens:
        tokens = xai_result.get("tokens")
    if weights is None:
        weights = xai_result.get("token_attributions") or xai_result.get("attributions")

    if not tokens or weights is None:
        return False

    weights_arr = _to_numpy_array(weights)
    if weights_arr is not None:
        weights_arr = np.squeeze(weights_arr)
        if weights_arr.ndim > 1:
            weights_arr = np.mean(weights_arr, axis=-1)
    if weights_arr is None or len(tokens) != len(weights_arr):
        return False

    st.markdown("##### 🔤 Token-Level Attribution & Sensitivity Map")

    max_w = float(np.max(np.abs(weights_arr))) if len(weights_arr) > 0 and np.max(np.abs(weights_arr)) > 0 else 1.0

    token_spans = []
    for token, weight in zip(tokens, weights_arr):
        norm_w = float(weight) / max_w
        if norm_w > 0.05:
            # Positive weight: driving adversarial misclassification
            bg = f"rgba(239, 68, 68, {min(0.8, abs(norm_w)):.2f})"
            color = "#ffffff"
        elif norm_w < -0.05:
            # Negative weight: supporting clean classification
            bg = f"rgba(16, 185, 129, {min(0.8, abs(norm_w)):.2f})"
            color = "#ffffff"
        else:
            bg = "rgba(100, 116, 139, 0.15)"
            color = "var(--text-primary)"

        token_spans.append(
            f'<span style="background-color: {bg}; color: {color}; padding: 3px 6px; margin: 2px; border-radius: 4px; font-family: monospace; display: inline-block;" title="Attribution: {weight:.4f}">{token}</span>'
        )

    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.6); padding: 14px; border-radius: 8px; border: 1px solid rgba(148, 163, 184, 0.2); line-height: 2.2;">
            {' '.join(token_spans)}
        </div>
        """,
        unsafe_allow_html=True,
    )
    return True


def render_temporal_attribution_curve(xai_result: Dict[str, Any]) -> bool:
    """
    Render 1D temporal attribution curves for time-series and sequential signals.
    Returns True if temporal attributions exist and are rendered, False otherwise.
    """
    raw_res = xai_result.get("_result_obj")
    attr_dict = xai_result.get("attribution") or {}
    if hasattr(raw_res, "attribution") and isinstance(raw_res.attribution, dict):
        attr_dict = raw_res.attribution

    def _ensure_1d_step(data: Any) -> Optional[np.ndarray]:
        arr = _to_numpy_array(data)
        if arr is None:
            return None
        arr = np.squeeze(arr)
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2:
            return np.mean(np.abs(arr), axis=-1)
        return None

    step_imp = _ensure_1d_step(xai_result.get("step_importance"))
    if step_imp is None and isinstance(attr_dict, dict) and "step_importance" in attr_dict:
        step_imp = _ensure_1d_step(attr_dict["step_importance"])
    elif step_imp is None and isinstance(attr_dict, dict) and "clean" in attr_dict and isinstance(attr_dict["clean"], dict):
        step_imp = _ensure_1d_step(attr_dict["clean"].get("attribution"))
    elif step_imp is None and isinstance(attr_dict, dict) and "adversarial" in attr_dict and isinstance(attr_dict["adversarial"], dict):
        step_imp = _ensure_1d_step(attr_dict["adversarial"].get("attribution"))
    elif step_imp is None and isinstance(attr_dict, dict) and "attribution" in attr_dict:
        step_imp = _ensure_1d_step(attr_dict["attribution"])
    elif step_imp is None and isinstance(attr_dict, (np.ndarray, list)):
        step_imp = _ensure_1d_step(attr_dict)
    elif step_imp is None and "clean_attribution" in xai_result:
        step_imp = _ensure_1d_step(xai_result["clean_attribution"])
    elif step_imp is None and "adversarial_attribution" in xai_result:
        step_imp = _ensure_1d_step(xai_result["adversarial_attribution"])

    if step_imp is None or len(step_imp) == 0:
        return False

    st.markdown("##### ⏱️ Temporal Step Attribution Profile")
    steps = list(range(len(step_imp)))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=step_imp,
            mode="lines+markers",
            name="Temporal Saliency",
            line=dict(color="#38bdf8", width=2),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.15)",
        )
    )
    fig.update_layout(
        xaxis_title="Time Step (t)",
        yaxis_title="Attribution Magnitude",
        margin=dict(l=10, r=10, t=20, b=20),
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
    )
    st.plotly_chart(fig, use_container_width=True)
    return True


def render_pre_post_defense_comparison(pre_xai: Dict[str, Any], post_xai: Dict[str, Any]) -> None:
    """
    Render side-by-side comparative feature attribution before vs. after defense hardening.
    Demonstrates how applied mitigations suppressed adversarial feature vulnerability.
    """
    st.markdown("### 🛡️ Pre-Defense vs. Post-Defense XAI Attribution Comparison")
    st.caption("Verification of feature stabilization and adversarial attribution neutralization post-hardening")

    pre_local = pre_xai.get("local_explanation", {})
    post_local = post_xai.get("local_explanation", {})

    pre_top = pre_local.get("adversarial_top_features", [])
    post_top = post_local.get("adversarial_top_features", [])

    pre_pred = pre_xai.get("adversarial_prediction")
    post_pred = post_xai.get("adversarial_prediction")
    clean_pred = pre_xai.get("clean_prediction")

    pre_pred_disp = f"Class {pre_pred}" if pre_pred is not None else "N/A"
    post_pred_disp = f"Class {post_pred}" if post_pred is not None else "N/A"

    pre_tech = str(pre_xai.get("technique", "")).upper()
    post_tech = str(post_xai.get("technique", "")).upper()
    if pre_tech and post_tech and pre_tech != post_tech:
        st.caption(f"Method Comparison Note: Pre-defense explainer ({pre_tech}) differs from Post-defense explainer ({post_tech}).")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="modern-card card-accent-rose" style="padding: 10px 14px;">
                <div class="stat-label">Pre-Hardening Attack State</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: var(--stat-val-rose);">{pre_pred_disp} (Perturbed)</div>
                <div class="stat-subtext">Original vulnerable model</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        is_restored = bool(post_pred is not None and clean_pred is not None and post_pred == clean_pred)
        col_res = "var(--stat-val-emerald)" if is_restored else "var(--stat-val-amber)"
        st.markdown(
            f"""
            <div class="modern-card card-accent-emerald" style="padding: 10px 14px;">
                <div class="stat-label">Post-Hardening Prediction</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: {col_res};">{post_pred_disp} {'(Restored ✅)' if is_restored else ''}</div>
                <div class="stat-subtext">Hardened model under attack</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        pre_drift = pre_local.get("attribution_drift") if pre_local.get("attribution_drift") is not None else pre_xai.get("attribution_drift")
        post_drift = post_local.get("attribution_drift") if post_local.get("attribution_drift") is not None else post_xai.get("attribution_drift")
        if isinstance(pre_drift, (int, float)) and isinstance(post_drift, (int, float)) and pre_drift > 0:
            drift_drop = max(0.0, (pre_drift - post_drift) / pre_drift * 100.0)
            drift_txt = f"{drift_drop:.1f}% Attrib. Stabilization"
        elif isinstance(post_drift, (int, float)) and pre_drift is None:
            drift_txt = f"Post Drift: {post_drift:.4f}"
        else:
            drift_txt = "N/A"

        st.markdown(
            f"""
            <div class="modern-card card-accent-cyan" style="padding: 10px 14px;">
                <div class="stat-label">Attribution Vulnerability Drop</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: var(--stat-val-cyan);">{drift_txt}</div>
                <div class="stat-subtext">Suppression of adversarial salience</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Side-by-side Top Features
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown("**Top Vulnerable Features (Pre-Hardening)**")
        if pre_top:
            p_df = pd.DataFrame([{"Feature": f.get("feature"), "Weight": f.get("weight", 0.0)} for f in pre_top[:5]])
            st.dataframe(p_df, hide_index=True, use_container_width=True)
        else:
            st.info("No pre-defense features logged.")

    with col_f2:
        st.markdown("**Top Features Post-Hardening (Neutralized)**")
        if post_top:
            post_df = pd.DataFrame([{"Feature": f.get("feature"), "Weight": f.get("weight", 0.0)} for f in post_top[:5]])
            st.dataframe(post_df, hide_index=True, use_container_width=True)
        else:
            st.info("Feature attributions stabilized across dimensions.")


def render_root_cause_analysis_card(xai_result: Dict[str, Any]) -> None:
    """
    Render Section D: Root Cause Analysis Card.
    Synthesizes prediction shift narrative, invariant status, attack causation,
    and diagnostic feature shifts table.
    """
    st.markdown("##### 🩺 Root Cause & Attribution Shift Analysis")

    narrative = (
        xai_result.get("attack_reason")
        or xai_result.get("failure_analysis", {}).get("attack_reason")
        or xai_result.get("metadata", {}).get("attack_reason")
    )

    c_pred = xai_result.get("clean_prediction")
    a_pred = xai_result.get("adversarial_prediction")
    c_conf = xai_result.get("clean_confidence")
    a_conf = xai_result.get("adversarial_confidence")
    pred_changed = xai_result.get("prediction_changed")
    caused_failure = xai_result.get("attack_caused_failure")
    if caused_failure is None:
        caused_failure = xai_result.get("failure_analysis", {}).get("attack_caused_failure")
    failure_mode = xai_result.get("failure_analysis", {}).get("failure_mode")

    c_conf_txt = f" ({c_conf*100:.1f}% conf)" if isinstance(c_conf, (int, float)) and c_conf <= 1.0 else (f" ({c_conf:.1f}% conf)" if c_conf else "")
    a_conf_txt = f" ({a_conf*100:.1f}% conf)" if isinstance(a_conf, (int, float)) and a_conf <= 1.0 else (f" ({a_conf:.1f}% conf)" if a_conf else "")
    c_pred_txt = f"Class {c_pred}{c_conf_txt}" if c_pred is not None else "N/A"
    a_pred_txt = f"Class {a_pred}{a_conf_txt}" if a_pred is not None else "N/A"

    if pred_changed is True:
        status_badge = '<span style="background-color: #ef4444; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;">CRITICAL: Decision Flipped</span>'
    elif pred_changed is False:
        status_badge = '<span style="background-color: #10b981; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;">STABLE: Decision Preserved</span>'
    else:
        status_badge = '<span style="background-color: #64748b; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;">INCONCLUSIVE</span>'

    causation_txt = "Yes (Adversarial perturbation induced classification error)" if caused_failure else ("No (Model resilient or baseline error)" if caused_failure is False else "N/A")

    st.markdown(
        f"""
        <div class="modern-card card-accent-cyan" style="padding: 14px 18px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 600; font-size: 1.05rem;">Prediction Shift & Decision Boundary Analysis</span>
                {status_badge}
            </div>
            <div style="font-size: 0.95rem; margin-bottom: 6px;">
                <strong>Baseline Transition:</strong> <code>{c_pred_txt}</code> ➔ <code>{a_pred_txt}</code>
            </div>
            <div style="font-size: 0.88rem; color: var(--text-muted, #94a3b8); margin-bottom: 8px;">
                <strong>Attack-Caused Failure:</strong> {causation_txt} &nbsp;|&nbsp; <strong>Failure Mode:</strong> <code>{failure_mode or 'N/A'}</code>
            </div>
            <div style="font-size: 0.92rem; line-height: 1.5; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 6px; border-left: 3px solid #38bdf8;">
                <strong>Root Cause Synthesis:</strong> {narrative or 'Attribution shift analysis observed perturbation across primary feature coordinates.'}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_attack_impact_and_vulnerability_card(xai_result: Dict[str, Any]) -> None:
    """
    Render Section E: Attack Impact & Vulnerability Analysis Card.
    Displays attack technique, objective, perturbation magnitude metrics (L_inf, L_2),
    and attribution-perturbation gradient alignment.
    """
    st.markdown("##### 💥 Attack Impact & Vulnerability Characteristics")

    attack_name = str(xai_result.get("attack_name", "adversarial_attack")).upper()
    fail_analysis = xai_result.get("failure_analysis", {})
    objective = fail_analysis.get("attack_objective", "untargeted")

    pert_metrics = xai_result.get("perturbation_metrics") or fail_analysis.get("perturbation_metrics") or {}
    l_inf = pert_metrics.get("linf") or pert_metrics.get("l_inf") or pert_metrics.get("max_perturbation")
    l_2 = pert_metrics.get("l2") or pert_metrics.get("l_2") or pert_metrics.get("mean_perturbation")

    alignment = xai_result.get("perturbation_alignment")
    if alignment is None:
        alignment = xai_result.get("comparison", {}).get("perturbation_alignment")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="modern-card card-accent-rose" style="padding: 10px 14px;">
                <div class="stat-label">Attack Technique</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: var(--stat-val-rose);">{attack_name}</div>
                <div class="stat-subtext">Objective: {objective}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        linf_str = f"{float(l_inf):.5f}" if isinstance(l_inf, (int, float)) else "N/A"
        st.markdown(
            f"""
            <div class="modern-card card-accent-amber" style="padding: 10px 14px;">
                <div class="stat-label">Perturbation L_∞</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #f59e0b;">{linf_str}</div>
                <div class="stat-subtext">Peak coordinate noise</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        l2_str = f"{float(l_2):.5f}" if isinstance(l_2, (int, float)) else "N/A"
        st.markdown(
            f"""
            <div class="modern-card card-accent-cyan" style="padding: 10px 14px;">
                <div class="stat-label">Perturbation L_2</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: var(--stat-val-cyan);">{l2_str}</div>
                <div class="stat-subtext">Euclidean energy</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        align_val = None
        align_desc = "Unavailable: metric not computed"
        if isinstance(alignment, dict):
            align_val = alignment.get("cosine_alignment")
            align_status = alignment.get("alignment_status")
            align_msg = alignment.get("message")
            if align_val is not None:
                align_val = float(align_val)
            elif align_status == "shape_mismatch":
                align_desc = f"Unavailable: {align_msg or 'incompatible attribution dimensions'}"
            elif align_status == "empty_input":
                align_desc = "Unavailable: empty input vectors"
            elif align_msg:
                align_desc = f"Unavailable: {align_msg}"
        elif isinstance(alignment, (int, float)):
            align_val = float(alignment)

        if align_val is not None and not (isinstance(align_val, float) and np.isnan(align_val)):
            align_str = f"{align_val:.4f}"
            if align_val > 0.5:
                align_desc = "Strong gradient alignment"
            elif align_val > 0.1:
                align_desc = "Moderate correlation"
            else:
                align_desc = "Low / orthogonal alignment"
        else:
            align_str = "N/A"
            if align_desc == "Unavailable: metric not computed":
                align_desc = "Unavailable: incompatible attribution dimensions"

        st.markdown(
            f"""
            <div class="modern-card card-accent-emerald" style="padding: 10px 14px;">
                <div class="stat-label">Perturbation Alignment</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: var(--stat-val-emerald);">{align_str}</div>
                <div class="stat-subtext">{align_desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_defense_recommendations_section(xai_result: Dict[str, Any]) -> None:
    """
    Render Section F: Evidence-Based Defense Recommendations.
    Displays candidate mitigations suggested by XAI findings, explicit candidate status badge,
    supporting evidence, rationale, expected benefit, limitations, and multi-stage defense architecture.
    Uses native Streamlit components to prevent HTML escaping or raw markup display bugs.
    """
    st.markdown("##### 💡 Evidence-Based Defense Recommendations")

    # Candidate status badge using native st.info
    st.info(
        "ℹ️ **CANDIDATE RECOMMENDATIONS**: The mitigations below are candidate defenses recommended by M6 explainability diagnostics "
        "based on observed attribution shifts and perturbation characteristics. They represent proposed candidate mitigations for "
        "evaluation and validation in Module M4 (Adversarial Defense Hardening), not active or applied defenses."
    )

    hints = xai_result.get("defense_hints") or xai_result.get("failure_analysis", {}).get("defense_hints", [])
    pipeline = xai_result.get("defense_pipeline") or xai_result.get("failure_analysis", {}).get("defense_pipeline", {})
    rec_list = xai_result.get("recommended_defenses") or xai_result.get("failure_analysis", {}).get("recommended_defenses", [])

    if hints:
        for idx, hint in enumerate(hints):
            if not isinstance(hint, dict):
                continue
            finding = hint.get("finding", "Observed attribution vulnerability")
            technique = hint.get("recommended_technique") or hint.get("defense") or hint.get("suggested_family") or "Adversarial Defense Mitigation"
            family = hint.get("suggested_family") or hint.get("suggested_defense_family") or hint.get("category") or "Model Hardening"
            stage = str(hint.get("defense_stage") or hint.get("stage") or "Model Stage").replace("_", " ").title()
            if not stage.endswith("Stage"):
                stage = f"{stage} Stage"

            raw_conf = hint.get("confidence") or hint.get("recommendation_confidence")
            if raw_conf is not None and not isinstance(raw_conf, str):
                conf_label = f"{raw_conf * 100:.0f}%" if raw_conf <= 1.0 else f"{raw_conf:.0f}%"
            elif raw_conf:
                conf_label = str(raw_conf)
            else:
                conf_label = "N/A (Heuristic)"

            why_rec = hint.get("why_recommended") or hint.get("rationale") or "Attribution shifts highlight sensitivity along primary feature coordinates."
            benefit = hint.get("expected_benefit") or "Reduces gradient exploitability and bounds decision boundary sensitivity under adversarial noise."
            limits = hint.get("limitations") or "Candidate mitigation requiring empirical robustness evaluation in Module M4."

            evidence = hint.get("evidence") or hint.get("quantitative_evidence", {})

            with st.container(border=True):
                head_col1, head_col2 = st.columns([3, 1])
                with head_col1:
                    st.markdown(f"**{idx + 1}. {technique}**")
                with head_col2:
                    st.caption(f"**Confidence:** `{conf_label}`")

                st.caption(f"**Family:** `{family}` &nbsp;|&nbsp; **Stage:** `{stage}` &nbsp;|&nbsp; **Finding:** {finding}")

                if evidence and isinstance(evidence, dict):
                    ev_items = []
                    for k, v in evidence.items():
                        k_fmt = k.replace("_", " ").title()
                        v_fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
                        ev_items.append(f"**{k_fmt}:** `{v_fmt}`")
                    st.markdown(f"📊 **Supporting Evidence:** {' &nbsp;|&nbsp; '.join(ev_items)}")

                st.markdown(f"• **Why Recommended:** {why_rec}")
                st.markdown(f"• **Expected Benefit:** {benefit}")
                st.markdown(f"• **Limitations & Validation:** *{limits}*")

    elif rec_list:
        st.markdown("**Suggested Candidate Mitigations:**")
        for idx, rec in enumerate(rec_list):
            if isinstance(rec, dict):
                r_name = rec.get("defense") or rec.get("name") or "Defense"
                r_desc = rec.get("description") or rec.get("rationale") or "Proposed candidate mitigation."
                with st.container(border=True):
                    st.markdown(f"**{idx + 1}. {r_name}**")
                    st.markdown(f"{r_desc}")
            else:
                st.markdown(f"• **{rec}** *(Candidate Defense for M4 evaluation)*")
    else:
        st.info("No specific defense hints generated for this evaluation.")

    # Multi-stage defense architecture pipeline
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("##### 🛡️ Multi-Stage Defense Architecture Pipeline (Candidate Mitigations)")
    st.caption("Proposed multi-tier defense architecture derived from M6 diagnostics for evaluation in Module M4.")

    col_p1, col_p2, col_p3 = st.columns(3)

    # Extract pipeline techniques or modality fallbacks
    p_input = []
    p_model = []
    p_post = []

    if isinstance(pipeline, dict):
        if "1_preprocessing" in pipeline or "2_transformation" in pipeline:
            for s_key in ("1_preprocessing", "2_transformation"):
                if s_key in pipeline and isinstance(pipeline[s_key], dict):
                    p_input.extend(pipeline[s_key].get("techniques", []))
        if "4_smoothing" in pipeline or "5_feature_level" in pipeline or "6_ensemble" in pipeline:
            for s_key in ("4_smoothing", "5_feature_level", "6_ensemble"):
                if s_key in pipeline and isinstance(pipeline[s_key], dict):
                    p_model.extend(pipeline[s_key].get("techniques", []))
        if "3_detection" in pipeline and isinstance(pipeline["3_detection"], dict):
            p_post.extend(pipeline["3_detection"].get("techniques", []))

        # Check for 3-tier keys if present
        if "input_stage" in pipeline and isinstance(pipeline["input_stage"], list):
            p_input.extend(pipeline["input_stage"])
        if "model_stage" in pipeline and isinstance(pipeline["model_stage"], list):
            p_model.extend(pipeline["model_stage"])
        if "post_processing_stage" in pipeline and isinstance(pipeline["post_processing_stage"], list):
            p_post.extend(pipeline["post_processing_stage"])

    # Fallback to sensible modality defaults if empty
    mod = str(xai_result.get("modality", "tabular")).lower().replace("-", "_")
    if not p_input:
        if mod == "image":
            p_input = ["Bit-depth reduction (5-bit quantization)", "Spatial median blur (3x3 kernel)"]
        elif mod in ("timeseries", "time_series"):
            p_input = ["Wavelet sub-band thresholding", "Rolling window median filtering"]
        elif mod == "text":
            p_input = ["Token normalization", "Low-perplexity candidate filtering"]
        else:
            p_input = ["Feature squeezing & quantization", "Domain boundary range clamping"]

    if not p_model:
        if mod == "image":
            p_model = ["Adversarial Training (PGD-AT / TRADES)", "Randomized Smoothing (Gaussian noise)"]
        elif mod in ("timeseries", "time_series"):
            p_model = ["Temporal dynamic regularization", "Multi-sensor robust training"]
        elif mod == "text":
            p_model = ["Adversarial token retraining", "Attention dispersion regularization"]
        else:
            p_model = ["Constrained Adversarial Training (TRADES)", "Tree ensemble column subsampling"]

    if not p_post:
        p_post = ["Mahalanobis distance OOD gating", "Autoencoder reconstruction anomaly detector"]

    with col_p1:
        with st.container(border=True):
            st.markdown("#### 1️⃣ Input Stage")
            st.caption("**Focus:** Input Sanitization & Denoising")
            st.markdown("---")
            for t in p_input[:3]:
                st.markdown(f"• {t}")
            st.caption("🔒 *Candidate mitigation for M4 evaluation*")

    with col_p2:
        with st.container(border=True):
            st.markdown("#### 2️⃣ Model Stage")
            st.caption("**Focus:** Robust Training & Regularization")
            st.markdown("---")
            for t in p_model[:3]:
                st.markdown(f"• {t}")
            st.caption("🔒 *Candidate mitigation for M4 evaluation*")

    with col_p3:
        with st.container(border=True):
            st.markdown("#### 3️⃣ Post-Processing Stage")
            st.caption("**Focus:** Anomaly Gating & Detector")
            st.markdown("---")
            for t in p_post[:3]:
                st.markdown(f"• {t}")
            st.caption("🔒 *Candidate mitigation for M4 evaluation*")


def render_technical_diagnostics_section(xai_result: Dict[str, Any]) -> None:
    """
    Render Section G: Technical Diagnostics & Explainer Execution Metadata.
    Provides routing decisions, device, execution time, model queries, batch stats,
    and diagnostic error messages.
    """
    with st.expander("🛠️ Technical Diagnostics & Explainer Execution Metadata", expanded=False):
        tech = str(xai_result.get("technique", "Unknown")).upper()
        modality = str(xai_result.get("modality", "Unknown")).title()
        status = str(xai_result.get("status", "Unknown"))
        executed = xai_result.get("executed", False)

        meta = xai_result.get("metadata") or {}
        decision = xai_result.get("decision") or {}
        comp = xai_result.get("comparison") or {}

        exec_time = (
            meta.get("execution_time_seconds")
            or meta.get("explain_seconds")
            or meta.get("execution_time")
            or xai_result.get("execution_time_seconds")
            or comp.get("execution_time")
        )
        if exec_time is None and isinstance(meta.get("duration"), (int, float)):
            exec_time = meta.get("duration")

        total_pipe = meta.get("total_pipeline_seconds")

        if isinstance(exec_time, (int, float)) and exec_time > 0:
            time_str = f"{float(exec_time):.3f} s"
            if total_pipe and total_pipe > exec_time:
                time_help = f"Explainer core: {exec_time:.3f} s | Total pipeline: {total_pipe:.3f} s"
            else:
                time_help = "Backend attribution computation wall-clock duration"
        else:
            time_str = "N/A"
            time_help = "Execution time not recorded"

        device = meta.get("device") or decision.get("device", "CPU / Native")
        model_type = decision.get("model_type") or meta.get("model_type", "PyTorch / Blackbox")
        routing_reason = decision.get("reason") or meta.get("routing_decision", "Selected based on modality and model differentiability.")

        batch_counts = xai_result.get("batch_counts", {})
        total_samples = batch_counts.get("total", 1)
        succeeded_samples = batch_counts.get("succeeded", 1 if executed else 0)
        failed_samples = batch_counts.get("failed", 0 if executed else 1)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Explainer Technique", tech, help="Explainer algorithm selected by M6 engine")
        with col2:
            st.metric("Execution Time", time_str, help=time_help)
        with col3:
            st.metric("Compute Device", str(device), help="Hardware acceleration device utilized")
        with col4:
            st.metric("Batch Evaluated", f"{succeeded_samples}/{total_samples}", help="Number of samples successfully explained")

        st.markdown(f"**Routing Rationale:** {routing_reason}")
        st.markdown(f"**Model Type & Domain:** `{model_type}` | Modality: `{modality}` | Execution Status: `{status}`")

        # Display any captured diagnostic errors or warnings
        err_msg = xai_result.get("message")
        if err_msg and ("fail" in err_msg.lower() or "error" in err_msg.lower() or "partial" in err_msg.lower()):
            st.warning(f"Diagnostic Notice: {err_msg}")


def render_dynamic_xai_dashboard(
    xai_result: Dict[str, Any],
    post_defense_xai: Optional[Dict[str, Any]] = None,
    title: str = "XAI Root Cause & Feature Attribution Analysis",
) -> None:
    """
    Main entry point: dynamically renders all available explanation artifacts
    independent of modality.
    """
    if not xai_result:
        st.info("No XAI explanations available to display.")
        return

    st.markdown(f"#### 🔍 {title}")

    # 1. Summary metrics (Section A)
    render_xai_metrics_summary(xai_result)
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # 2. Dynamic visual artifact inspection (Section B/C)
    rendered_any = False

    # Check for 2D spatial / spectral heatmaps (Image, Audio Spectrogram)
    if render_spatial_heatmaps(xai_result):
        rendered_any = True
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Check for text token spans
    if render_token_attribution_view(xai_result):
        rendered_any = True
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Check for temporal curves (Time-Series, Audio)
    if render_temporal_attribution_curve(xai_result):
        rendered_any = True
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Check for feature importance bars (Tabular, or fallback across any domain)
    if render_feature_importance_bars(xai_result):
        rendered_any = True

    # Clean vs. Adversarial side-by-side attribution table (all modalities)
    if render_clean_vs_adversarial_attribution_table(xai_result):
        rendered_any = True
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # 3. Root Cause Analysis Card (Section D)
    render_root_cause_analysis_card(xai_result)
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # 4. Attack Impact & Vulnerability Characteristics (Section E)
    render_attack_impact_and_vulnerability_card(xai_result)
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # 5. Global summary insights: separate intrinsic importance from adversarial vulnerability
    glob_exp = xai_result.get("global_explanation", {})
    glob_imp = glob_exp.get("global_feature_importance", [])
    vuln_freq = glob_exp.get("vulnerability_frequency", [])
    if not vuln_freq:
        vuln_freq = glob_exp.get("global_top_features", [])

    if glob_imp or vuln_freq:
        tech_label = str(xai_result.get("technique", "Attribution")).upper()
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(f"##### 🌐 Global Feature Importance (Aggregated {tech_label} Attributions)")
            if glob_imp:
                for g in glob_imp[:5]:
                    imp_val = g.get("mean_abs_importance", g.get("importance", 0.0))
                    st.markdown(f"• **{g.get('feature', 'feat')}**: mean |attribution| `{imp_val:.4f}` (sample coverage `{g.get('sample_count', 0)}`)")
            else:
                st.caption("No aggregated global attribution available.")

        with col_g2:
            st.markdown("##### 🎯 Adversarial Vulnerability Frequency")
            if vuln_freq:
                for v in vuln_freq[:5]:
                    succ_cnt = v.get("success_frequency", v.get("frequency", 0))
                    tot_cnt = v.get("frequency", v.get("total_frequency", 0))
                    st.markdown(f"• **{v.get('feature', 'feat')}**: exploited in `{succ_cnt}` successful attacks (total freq `{tot_cnt}`)")
            else:
                st.caption("No recurring feature exploitation patterns detected.")

    # 6. Evidence-Based Defense Recommendations (Section F)
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    render_defense_recommendations_section(xai_result)

    # 7. Post-Defense Comparison if available
    if post_defense_xai:
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        render_pre_post_defense_comparison(xai_result, post_defense_xai)

    # 8. Technical Diagnostics Expander (Section G)
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    render_technical_diagnostics_section(xai_result)
