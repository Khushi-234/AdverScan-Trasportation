"""
AdverScan — Adversarial Robustness and Security Testing Platform
Continuous Assessment Pipeline:
Dashboard → Model Ingestion → Attack Configuration → Attack Execution
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure repository root is on sys.path for clean imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="AdverScan - Robustness Assessment",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from frontend.components.sidebar import (
    render_sidebar,
    render_sidebar_branding,
    render_sidebar_session,
)
from frontend.services.attack_service import (
    execute_configured_attack,
    get_compatible_attacks,
)
from frontend.services.defense_service import (
    execute_configured_defense,
    get_available_defenses_for_domain,
)
from frontend.services.ingestion_service import (
    get_available_presets,
    get_dataset_total_samples,
    load_model_and_dataset,
)
from frontend.services.report_service import generate_security_report
from frontend.services.vulnerability_service import (
    assess_and_score_vulnerability,
    generate_vulnerability_recommendations,
)
from frontend.utils.session_state import (
    get_active_assessment,
    get_attack_result,
    get_defense_result,
    get_report_result,
    get_state,
    get_vulnerability_result,
    get_workflow_stage,
    init_session_state,
    reset_workflow_state,
    set_active_assessment,
    set_attack_result,
    set_defense_result,
    set_report_result,
    set_state,
    set_vulnerability_result,
    set_workflow_stage,
)
from frontend.utils.theme_manager import initialize_theme, load_theme

# Initialize session state first
init_session_state()

# Render sidebar header branding first
render_sidebar_branding()

# Initialize theme from theme manager (renders Interface Theme selector in sidebar & loads CSS)
current_theme = initialize_theme()
set_state("theme", current_theme)

# Render active session information and controls in sidebar
render_sidebar_session()


stage = get_workflow_stage()


# ==============================================================================
# STAGE 1: DASHBOARD (Welcome & Framework Introduction)
# ==============================================================================
if stage == "dashboard":
    st.markdown(
        """
        <div class="hero-banner">
            <div style="display: flex; align-items: center; gap: 16px;">
                <span style="font-size: 3rem;">🛡️</span>
                <div>
                    <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700;">Welcome to AdverScan</h1>
                    <p style="margin: 0.5rem 0 0 0; font-size: 1.05rem;">
                        Automated End-to-End Adversarial Robustness and AI Security Assessment Platform.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_intro1, col_intro2 = st.columns([3, 2])

    with col_intro1:
        st.subheader("The Continuous Assessment Workflow")
        st.markdown("""
            AdverScan coordinates a systematic, step-by-step security assessment lifecycle:

            1. **Model Ingestion & Introspection**: Load pre-trained models across Vision, Tabular, NLP, and Time-Series domains with automated weight introspection and dataset resolution.
            2. **Attack Discovery & Threat Formulation**: Dynamically discover compatible attacks from the verified backend catalog with guided hyperparameter configuration.
            3. **Adversarial Perturbation Execution**: Execute gradient-based and optimization attacks within mathematical bounds with real-time progress tracking.
            4. **Comprehensive Telemetry & Distortion Overview**: Evaluate model degradation, accuracy drop, confidence shifts, and exact perturbation norms ($L_\\infty, L_2, L_0, \\text{MAE}$).
            """)

        st.info(
            "ℹ️ **Current Phase**: **Model Ingestion** and **Adversarial Attack Execution** are the active workflow stages in this phase."
        )

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        if st.button(
            "🚀 Start Model Assessment", type="primary", use_container_width=True
        ):
            set_workflow_stage("ingestion")
            st.rerun()

    with col_intro2:
        st.subheader("Key Platform Capabilities")
        st.markdown("""
            - **Zero Fabricated Telemetry**: Every metric, gradient shift, and distortion score is calculated directly from PyTorch / scikit-learn models.
            - **Hardware-Adaptive Sample Slicing**: Evaluate on flexible sample batches (2 to 100+) for rapid experimentation on CPU without GPU bottlenecks.
            - **Unified Modality Support**: Standardized model adapters and dataset loaders for Computer Vision, Tabular, Text, and Temporal signals.
            - **Mathematical Distortion Bounds**: Precise $L_\\infty$, $L_2$, $L_0$, and Mean Absolute Error verification.
            """)


# ==============================================================================
# STAGE 2: MODEL INGESTION
# ==============================================================================
elif stage == "ingestion":
    st.markdown("## 📥 Model Ingestion & Parameter Introspection")
    st.caption(
        "Step 1 of 3 • Load and validate an AI model, introspect architecture, and prepare evaluation dataset"
    )

    presets = get_available_presets()
    preset_keys = list(presets.keys())

    col_src, col_opts = st.columns([1, 1])

    with col_src:
        source_type = st.radio(
            "Model Source",
            ["Preset Benchmark Model", "Hugging Face Model ID", "Local Model Path"],
            help="Select how to provide the target model",
        )

    with col_opts:
        device_opt = st.selectbox(
            "Execution Device",
            ["Auto-detect", "CPU", "CUDA (GPU)"],
            index=0,
            help="Compute device for forward passes and gradient backpropagation",
        )

    # Input fields based on source
    model_identifier = ""
    default_dataset = ""
    default_domain = "image"
    default_framework = "huggingface"

    if source_type == "Preset Benchmark Model":
        preset_choice = st.selectbox(
            "Select Pre-Configured Model Preset",
            preset_keys,
            format_func=lambda k: f"{k} ({presets[k].get('data_domain', 'image').capitalize()} - {presets[k].get('name', '')})",
        )
        selected_preset_info = presets[preset_choice]
        model_identifier = preset_choice
        default_dataset = selected_preset_info.get("dataset_name", "")
        default_domain = selected_preset_info.get("data_domain", "image")
        default_framework = selected_preset_info.get("framework", "huggingface")

        st.caption(
            f"**Preset Description**: {selected_preset_info.get('description', '')}"
        )

    elif source_type == "Hugging Face Model ID":
        model_identifier = st.text_input(
            "Hugging Face Repository ID",
            value="bazyl/gtsrb-model",
            placeholder="e.g. bazyl/gtsrb-model, google/vit-base-patch16-224",
            help="Hugging Face model repository identifier",
        )
        default_domain = st.selectbox(
            "Data Domain", ["image", "text", "tabular", "time-series"], index=0
        )
        default_framework = "huggingface"

    else:  # Local Model Path
        model_identifier = st.text_input(
            "Local Model File Path",
            value="models/tabular-mlp-adult-census.pt",
            placeholder="e.g. models/classifier.pth or models/tabular-mlp-adult-census.pt",
            help="Path to pre-trained model file on local disk",
        )
        default_domain = st.selectbox(
            "Data Domain", ["tabular", "image", "text", "time-series"], index=0
        )
        default_framework = st.selectbox("Framework", ["pytorch", "sklearn"], index=0)

    col_data, col_sample = st.columns([1, 1])

    with col_data:
        dataset_input = st.text_input(
            "Dataset Identifier or Path",
            value=default_dataset,
            placeholder="e.g. bazyl/GTSRB, data/adult_census_test.csv, or test split",
            help="Dataset name or path to load evaluation samples from",
        )

        effective_dataset = dataset_input.strip() if dataset_input.strip() else default_dataset
        total_avail_samples = get_dataset_total_samples(
            dataset_name=effective_dataset,
            data_domain=default_domain,
            model_identifier=model_identifier,
        )
        half_samples = max(1, total_avail_samples // 2)
        full_samples = total_avail_samples

        st.markdown(
            f"""
            <div style="background: rgba(6, 182, 212, 0.08); border: 1px solid rgba(6, 182, 212, 0.25); border-radius: 8px; padding: 10px 14px; margin-top: 6px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.84rem; font-weight: 600; color: var(--text-secondary); text-transform: uppercase;">
                        📦 Dataset Population in Model/Benchmark
                    </span>
                    <span class="badge-status badge-cyan">{default_domain.upper()}</span>
                </div>
                <div style="font-size: 1.35rem; font-weight: 700; color: var(--stat-val-cyan); margin: 4px 0;">
                    {total_avail_samples:,} <span style="font-size: 0.9rem; font-weight: 500; color: var(--text-secondary);">Total Samples Present</span>
                </div>
                <div style="font-size: 0.8rem; color: var(--text-secondary);">
                    Resolved source: <code>{effective_dataset}</code> • Half split: <strong>{half_samples:,}</strong> • Full split: <strong>{full_samples:,}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_sample:
        sample_choice = st.radio(
            "Evaluation Sample Population",
            [
                "Custom Number of Samples",
                f"Half Dataset (50% • {half_samples:,} samples)",
                f"Full Dataset (100% • {full_samples:,} samples)",
            ],
            index=0,
            help="Select a custom sample batch, half (50%) of the benchmark, or the complete dataset (100%)",
        )

        if sample_choice.startswith("Half Dataset"):
            sample_size = half_samples
            st.markdown(
                f"""
                <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 10px 14px; margin-top: 6px;">
                    <div style="font-weight: 700; color: var(--stat-val-emerald); font-size: 0.95rem;">
                        ⚡ Half Dataset Mode Selected
                    </div>
                    <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 4px; line-height: 1.4;">
                        Evaluating <strong>{sample_size:,}</strong> samples (50.0% coverage of {total_avail_samples:,} total samples).
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif sample_choice.startswith("Full Dataset"):
            sample_size = full_samples
            warning_msg = ""
            if full_samples > 1000 and default_domain in ("image", "vision"):
                warning_msg = f"<div style='font-size: 0.78rem; color: #f59e0b; margin-top: 4px;'>⚠️ Note: Ingesting {sample_size:,} vision tensors will run gradient backpropagation across all images.</div>"
            st.markdown(
                f"""
                <div style="background: rgba(139, 92, 246, 0.08); border: 1px solid rgba(139, 92, 246, 0.25); border-radius: 8px; padding: 10px 14px; margin-top: 6px;">
                    <div style="font-weight: 700; color: #8b5cf6; font-size: 0.95rem;">
                        🚀 Full Dataset Mode Selected
                    </div>
                    <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 4px; line-height: 1.4;">
                        Evaluating all <strong>{sample_size:,}</strong> samples (100.0% full benchmark coverage).
                        {warning_msg}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            sample_size = st.number_input(
                "Evaluation Samples (Batch Size)",
                min_value=1,
                max_value=max(100, total_avail_samples),
                value=min(5, total_avail_samples),
                step=1,
                help=f"Number of samples to load into memory for evaluation (1 to {total_avail_samples:,})",
            )
            pct_cov = (sample_size / max(1, total_avail_samples)) * 100.0
            st.caption(
                f"Selected: **{sample_size:,}** of **{total_avail_samples:,}** samples (**{pct_cov:.1f}%** dataset coverage)"
            )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns([1, 1])

    with col_btn1:
        if st.button("← Back to Dashboard", use_container_width=True):
            set_workflow_stage("dashboard")
            st.rerun()

    with col_btn2:
        ingest_clicked = st.button(
            "Ingest & Validate Model →", type="primary", use_container_width=True
        )

    # Ingestion Execution with multi-step progress bar
    if ingest_clicked:
        if not model_identifier.strip():
            st.error("Please provide a valid model identifier or path.")
        else:
            progress_bar = st.progress(
                0, text="Initializing model ingestion environment..."
            )
            time.sleep(0.08)

            try:
                progress_bar.progress(
                    20,
                    text="Validating model identifier and resolving framework adapter...",
                )
                time.sleep(0.08)

                progress_bar.progress(
                    45,
                    text="Loading model architecture and introspecting parameter tensors...",
                )
                res_data = load_model_and_dataset(
                    model_identifier=model_identifier,
                    dataset_name=dataset_input if dataset_input.strip() else None,
                    data_domain=default_domain,
                    framework=default_framework,
                    device=device_opt,
                    sample_size=sample_size,
                )

                progress_bar.progress(
                    75,
                    text="Loading dataset split and constructing evaluation tensor batches...",
                )
                time.sleep(0.08)

                progress_bar.progress(
                    90,
                    text="Computing parameter statistics, model size, and tensor shapes...",
                )
                time.sleep(0.08)

                set_active_assessment(res_data)

                progress_bar.progress(
                    100, text="Model ingestion and validation complete!"
                )
                time.sleep(0.12)
                progress_bar.empty()

                st.success(
                    "✅ Model and dataset successfully ingested into active session!"
                )
            except Exception as err:
                progress_bar.empty()
                st.error(f"❌ Ingestion Error: {str(err)}")

    # Display confirmed active model details if loaded
    active_data = get_active_assessment()
    if active_data:
        st.markdown("---")
        st.markdown(
            """
            <div class="section-banner">
                <h3>📊 Model Ingestion & Parameter Summary</h3>
                <p>Complete model architecture, parameter footprint, and evaluation batch telemetry</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        meta = active_data.get("metadata_dict") or active_data.get("telemetry") or {}
        total_p = active_data.get("total_parameters") or meta.get("total_parameters")
        train_p = active_data.get("trainable_parameters") or meta.get(
            "trainable_parameters"
        )
        train_pct = active_data.get("trainable_pct", 100.0)
        size_mb = active_data.get("estimated_size_mb") or meta.get("estimated_size_mb")
        ingest_time = active_data.get("ingestion_time")
        arch = (
            active_data.get("model_architecture")
            or meta.get("Architecture")
            or meta.get("architecture")
            or "Unknown"
        )
        num_cls = (
            active_data.get("num_classes")
            or meta.get("Total Classes")
            or "Not available"
        )
        if str(num_cls) == "N/A":
            num_cls = "Not available"
        in_shape = (
            active_data.get("input_shape") or meta.get("Input Shape") or "Not available"
        )
        if str(in_shape) == "N/A":
            in_shape = "Not available"
        tensor_shape = (
            active_data.get("batch_tensor_shape")
            or meta.get("batch_shape")
            or "Not available"
        )
        if str(tensor_shape) == "N/A":
            tensor_shape = "Not available"
        samples_eval = active_data.get("sample_size", sample_size)
        device_used = active_data.get("device") or meta.get("Device") or device_opt
        domain_name = active_data.get("domain", default_domain)
        grad_access = (
            active_data.get("gradient_access")
            or meta.get("Gradient Access")
            or "Full Backprop (White-Box)"
        )
        dataset_ident = (
            active_data.get("dataset_name")
            or meta.get("Dataset")
            or "In-memory dataset"
        )
        ingest_disp = (
            f"{ingest_time:.3f}s"
            if (ingest_time is not None and ingest_time > 0)
            else "Not measured"
        )

        # 3 Structured Modern Cards
        col_c1, col_c2, col_c3 = st.columns(3)

        with col_c1:
            total_p_disp = (
                f"{total_p:,} (~{total_p / 1e6:.2f}M)"
                if total_p is not None
                else "Not available"
            )
            train_p_disp = (
                f"{train_p:,} ({train_pct:.1f}%)"
                if train_p is not None
                else "Not available"
            )
            size_disp = (
                f"~{size_mb:.2f} MB (In-Memory Weights)"
                if (size_mb is not None and size_mb > 0)
                else "Not available"
            )

            st.markdown(
                f"""
                <div class="modern-card card-accent-cyan">
                    <div class="stat-label">Model Architecture & Footprint</div>
                    <div style="margin: 0.5rem 0;">
                        <span style="font-size: 1.15rem; font-weight: 700; color: var(--text-primary);">{meta.get('Model Name', model_identifier)}</span>
                    </div>
                    <div style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.6;">
                        • <strong>Architecture:</strong> {arch}<br>
                        • <strong>Total Parameters:</strong> {total_p_disp}<br>
                        • <strong>Trainable Parameters:</strong> {train_p_disp}<br>
                        • <strong>Estimated Model Size:</strong> {size_disp}<br>
                        • <strong>Framework:</strong> {meta.get('Framework', default_framework)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_c2:
            st.markdown(
                f"""
                <div class="modern-card card-accent-emerald">
                    <div class="stat-label">Modality & Input Specifications</div>
                    <div style="margin: 0.5rem 0;">
                        <span class="badge-status badge-emerald">{domain_name.upper()} DOMAIN</span>
                        <span class="badge-status badge-cyan">{meta.get('Task Type', 'Classification')}</span>
                    </div>
                    <div style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.6; margin-top: 0.6rem;">
                        • <strong>Total Output Classes:</strong> {num_cls}<br>
                        • <strong>Input Shape:</strong> {in_shape}<br>
                        • <strong>Execution Device:</strong> {device_used}<br>
                        • <strong>Gradient Access:</strong> {grad_access}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_c3:
            tot_ds_samples = active_data.get("total_dataset_samples") or meta.get("Total Dataset Samples") or meta.get("total_dataset_samples")
            tot_samples_str = f"{tot_ds_samples:,} samples" if isinstance(tot_ds_samples, int) else (f"{tot_ds_samples}" if tot_ds_samples else "Available")
            cov_pct_str = f"({(samples_eval / tot_ds_samples * 100):.1f}% coverage)" if isinstance(tot_ds_samples, int) and tot_ds_samples > 0 else ""
            st.markdown(
                f"""
                <div class="modern-card card-accent-violet">
                    <div class="stat-label">Dataset & Runtime Telemetry</div>
                    <div style="margin: 0.5rem 0;">
                        <span class="badge-status badge-violet">READY FOR ATTACK</span>
                    </div>
                    <div style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.6; margin-top: 0.6rem;">
                        • <strong>Dataset Identifier:</strong> {dataset_ident}<br>
                        • <strong>Dataset Population:</strong> {tot_samples_str}<br>
                        • <strong>Evaluated Samples:</strong> {samples_eval} samples {cov_pct_str}<br>
                        • <strong>Batch Tensor Shape:</strong> {tensor_shape}<br>
                        • <strong>Ingestion Wall Time:</strong> {ingest_disp}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if active_data and active_data.get("is_synthetic_dataset"):
            st.warning(
                "⚠️ **Synthetic Fallback Data Active:** The requested dataset could not be loaded; "
                "synthetic inputs were generated to allow functional inspection. "
                "These results must **not** be presented as a real dataset robustness experiment."
            )

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        if st.button(
            "Proceed to Attack Discovery & Configuration →",
            type="primary",
            use_container_width=True,
        ):
            set_workflow_stage("attack_config")
            st.rerun()


# ==============================================================================
# STAGE 3: ATTACK DISCOVERY & CONFIGURATION
# ==============================================================================
elif stage == "attack_config":
    st.markdown("## ⚡ Attack Discovery & Configuration")
    st.caption(
        "Step 2 of 3 • Select and configure verified backend attacks compatible with model architecture"
    )

    active_data = get_active_assessment()

    # Enforce continuous workflow order
    if not active_data or not active_data.get("adapter"):
        st.warning(
            "⚠️ No model has been ingested yet. Please complete Model Ingestion first."
        )
        if st.button("Go to Model Ingestion", type="primary"):
            set_workflow_stage("ingestion")
            st.rerun()
        st.stop()

    # Synthetic fallback alert (A05)
    if active_data.get("is_synthetic_dataset"):
        st.warning(
            "⚠️ **Notice: Operating on Synthetic Fallback Data.** "
            "Real evaluation dataset was unavailable. Generated samples are in use for demonstration."
        )

    model_adapter = active_data["adapter"]
    domain = active_data.get("domain", "image")
    meta_dict = active_data.get("metadata_dict") or active_data.get("telemetry") or {}
    available_samples = (
        active_data.get("test_inputs").shape[0]
        if active_data.get("test_inputs") is not None
        else 1
    )

    # Target Model Banner
    model_name_disp = (
        meta_dict.get("Model Name")
        or active_data.get("model_name")
        or active_data.get("dataset_name")
        or "Ingested Model"
    )
    arch_disp = (
        active_data.get("model_architecture")
        or meta_dict.get("Architecture")
        or "Unknown"
    )
    total_p = active_data.get("total_parameters") or meta_dict.get("total_parameters")
    params_disp = (
        f"~{total_p / 1e6:.2f}M ({total_p:,} params)" if total_p else "Not available"
    )
    device_disp = active_data.get("device") or meta_dict.get("Device") or "CPU"
    threat_badge = (
        "THREAT MODEL: WHITE-BOX GRADIENT ACCESS"
        if "White-Box" in str(active_data.get("gradient_access", "White-Box"))
        else "THREAT MODEL: BLACK-BOX ACCESS"
    )

    st.markdown(
        f"""
        <div class="modern-card card-accent-cyan" style="margin-bottom: 1.25rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="font-size: 0.75rem; font-weight: 700; color: var(--primary-color); letter-spacing: 0.05em;">TARGET MODEL READY</span>
                    <h3 style="margin: 0.2rem 0; color: var(--text-primary); font-size: 1.4rem;">{model_name_disp}</h3>
                    <span style="color: var(--text-muted); font-size: 0.85rem;">
                        Domain: <strong style="color: var(--text-primary);">{domain.capitalize()}</strong> •
                        Architecture: <strong style="color: var(--text-primary);">{arch_disp}</strong> •
                        Parameters: <strong style="color: var(--text-primary);">{params_disp}</strong> •
                        Device: <strong style="color: var(--text-primary);">{device_disp}</strong>
                    </span>
                </div>
                <div>
                    <span class="badge-status badge-cyan">{threat_badge}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Attack Domain & Threat Model Typography
    st.markdown(
        f"""
        <div class="section-banner" style="border-left: 4px solid var(--accent-violet); margin-bottom: 1.5rem;">
            <div style="color: var(--text-primary); font-size: 0.95rem; font-weight: 600;">
                Domain Context: {domain.upper()} ADVERSARIAL ROBUSTNESS
            </div>
            <div style="color: var(--text-secondary); font-size: 0.85rem; line-height: 1.5; margin-top: 0.35rem;">
                AdverScan evaluates your model under a <strong>White-Box</strong> threat model. The attack engine leverages complete
                access to the model's loss landscape, forward activation maps, and backpropagation gradients to construct minimal
                adversarial perturbations. Only verified canonical attacks implemented in the backend engine are presented below.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Retrieve canonical attacks matching backend
    with st.spinner("Querying backend attack engine for compatible attacks..."):
        compatible_attacks = get_compatible_attacks(
            model=model_adapter, data_domain=domain
        )

    if not compatible_attacks:
        st.warning(
            f"No registered attacks in the backend are compatible with domain '{domain}' and model capabilities."
        )
        if st.button("← Back to Model Ingestion"):
            set_workflow_stage("ingestion")
            st.rerun()
        st.stop()

    # Attack Selection
    col_atk_sel, col_sample_sel = st.columns([3, 2])

    with col_atk_sel:
        attack_options = [
            f"{a.get('name', 'Attack')} — {a.get('full_name', a.get('name', ''))} ({a.get('category', 'Gradient-based')})"
            for a in compatible_attacks
        ]
        selected_idx = st.selectbox(
            "Select Attack Algorithm",
            range(len(compatible_attacks)),
            format_func=lambda i: attack_options[i],
            help="Canonical attacks discovered and verified from the backend attack engine",
        )
        selected_attack = compatible_attacks[selected_idx]

    with col_sample_sel:
        # Prominent Sample Count Selection for CPU / GPU resource management
        sample_choice = st.selectbox(
            "Evaluation Sample Count (Batch Slicing)",
            options=[
                f"{min(2, available_samples)} samples (Ultra-Fast / CPU)",
                f"{min(5, available_samples)} samples (Fast / Recommended for CPU)",
                f"{min(10, available_samples)} samples (Standard)",
                f"{min(20, available_samples)} samples (Thorough)",
                f"All Ingested ({available_samples} samples)",
            ],
            index=1 if available_samples >= 5 else 0,
            help="Allows users without high-end GPUs to evaluate attacks rapidly on smaller batches",
        )
        # Parse selected sample count
        if "All Ingested" in sample_choice:
            eval_sample_count = available_samples
        else:
            try:
                eval_sample_count = int(sample_choice.split()[0])
            except Exception:
                eval_sample_count = min(5, available_samples)
        eval_sample_count = min(max(1, eval_sample_count), available_samples)
        st.caption(
            f"⚡ Evaluating **{eval_sample_count}** of **{available_samples}** loaded sample(s)"
        )

    # Display Attack Specifications & Characteristics
    col_spec1, col_spec2, col_spec3 = st.columns(3)
    with col_spec1:
        st.markdown(
            f"**Methodology**: `{selected_attack.get('category', 'Gradient-based')}`"
        )
        st.markdown(
            f"**Threat Model**: `{selected_attack.get('attack_type', selected_attack.get('threat_model', 'White-Box'))}`"
        )
    with col_spec2:
        norms_list = selected_attack.get("supported_norms", ["Linf"])
        st.markdown(f"**Supported Norms**: `{', '.join(norms_list)}`")
        st.markdown(
            f"**Optimization Type**: `{'Multi-step Iterative' if selected_attack.get('iterative', False) else 'Single-step Closed-form'}`"
        )
    with col_spec3:
        st.markdown(
            f"**Targeted Mode**: `{'Supported' if selected_attack.get('targeted_supported', False) else 'Untargeted Only'}`"
        )
        st.markdown(
            f"**Computational Cost**: `{selected_attack.get('computational_cost', 'Medium')}`"
        )

    st.markdown(f"*{selected_attack.get('description', '')}*")

    st.markdown("---")
    st.subheader("Configure Attack Parameters & Telemetry")

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        epsilon = st.slider(
            "Perturbation Budget (ε)",
            min_value=0.001,
            max_value=0.5,
            value=0.05,
            step=0.005,
            format="%.3f",
            help="Maximum perturbation magnitude (L-infinity / L2 bound)",
        )
        st.markdown(
            """
            <div class="param-explainer">
                <strong>Perturbation Budget (ε):</strong> Defines the maximum allowable distance between clean and adversarial inputs.
                <span class="param-impact">• <em>Impact:</em> Higher ε increases Attack Success Rate (ASR) by allowing larger distortions, but makes noise more human-visible. Lower ε ensures imperceptibility but requires higher model vulnerability to succeed.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_clip_a, col_clip_b = st.columns(2)
        with col_clip_a:
            clip_min = st.number_input(
                "Clip Min", value=0.0, step=0.1, help="Minimum valid tensor value"
            )
        with col_clip_b:
            clip_max = st.number_input(
                "Clip Max", value=1.0, step=0.1, help="Maximum valid tensor value"
            )

        st.markdown(
            """
            <div class="param-explainer">
                <strong>Tensor Clip Bounds [min, max]:</strong> Constrains perturbed features within physically valid boundaries (e.g. [0.0, 1.0] for normalized pixels).
                <span class="param-impact">• <em>Impact:</em> Prevents mathematical overflow/underflow artifacts that would be trivially caught by input sanitization filters.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    params_dict = {}

    with col_p2:
        if selected_attack.get("iterative", False):
            num_steps = st.slider(
                "Optimization Iterations (Steps)",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                help="Number of iterative gradient ascent steps",
            )
            params_dict["num_steps"] = num_steps
            st.markdown(
                """
                <div class="param-explainer">
                    <strong>Optimization Iterations:</strong> Number of gradient ascent steps taken along the loss surface.
                    <span class="param-impact">• <em>Impact:</em> More steps increase attack efficacy and convergence against hardened models, but compute time scales linearly with step count.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            alpha = st.slider(
                "Step Size (α)",
                min_value=0.001,
                max_value=0.1,
                value=0.01,
                step=0.002,
                format="%.3f",
                help="Magnitude of perturbation added at each step before projection",
            )
            params_dict["alpha"] = alpha
            st.markdown(
                """
                <div class="param-explainer">
                    <strong>Step Size (α):</strong> Magnitude of perturbation added per iteration before projection back into the ε-ball.
                    <span class="param-impact">• <em>Impact:</em> Typically set to ε/4 or ε/10. Large α causes optimization oscillation around boundaries; small α slows convergence.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if selected_attack.get("targeted_supported", False):
            is_targeted = st.checkbox(
                "Targeted Attack Mode",
                value=False,
                help="Attempt to force a specific incorrect class rather than any misclassification",
            )
            params_dict["targeted"] = is_targeted
            if is_targeted:
                raw_num_classes = (
                    active_data.get("num_classes")
                    or meta_dict.get("Total Classes")
                    or meta_dict.get("num_classes")
                )
                try:
                    num_classes_val = int(raw_num_classes)
                    max_class_idx = max(1, num_classes_val - 1)
                except (ValueError, TypeError):
                    max_class_idx = 9999

                target_class_choice = st.number_input(
                    "Target Class Label / Index",
                    min_value=0,
                    max_value=max_class_idx,
                    value=1 if max_class_idx >= 1 else 0,
                    step=1,
                    help="Specific class label index that the attack should force the model to predict",
                )
                params_dict["target_label"] = int(target_class_choice)
            else:
                params_dict.pop("target_label", None)
                params_dict.pop("target", None)

            st.markdown(
                """
                <div class="param-explainer">
                    <strong>Targeted vs Untargeted:</strong> Untargeted forces any incorrect prediction; Targeted forces a chosen malicious label.
                    <span class="param-impact">• <em>Impact:</em> Targeted attacks require higher perturbation budgets and more steps to achieve high success rates.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Validation
    valid_params = True
    if epsilon <= 0.0:
        st.error("Epsilon must be greater than 0.")
        valid_params = False
    if clip_min >= clip_max:
        st.error("Clip minimum must be strictly less than clip maximum.")
        valid_params = False

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    col_nav1, col_nav2 = st.columns([1, 1])

    with col_nav1:
        if st.button("← Back to Model Ingestion", use_container_width=True):
            set_workflow_stage("ingestion")
            st.rerun()

    with col_nav2:
        run_btn = st.button(
            "Execute Adversarial Attack ⚡",
            type="primary",
            use_container_width=True,
            disabled=not valid_params,
        )

    if run_btn and valid_params:
        test_inputs = active_data.get("test_inputs")
        test_labels = active_data.get("test_labels")

        # Slice samples based on user selection
        if test_inputs is not None:
            eval_inputs = test_inputs[:eval_sample_count]
        else:
            eval_inputs = None

        if test_labels is not None:
            eval_labels = test_labels[:eval_sample_count]
        else:
            eval_labels = None

        atk_name = selected_attack.get("name", "Adversarial Attack")
        atk_id = selected_attack.get("id", "fgsm")

        # Multi-step progress bar during attack execution
        atk_prog = st.progress(0, text=f"Initializing {atk_name} attack tensors...")
        time.sleep(0.08)

        try:
            atk_prog.progress(
                25, text="Binding model gradient graphs and verifying tensor bounds..."
            )
            time.sleep(0.08)

            atk_prog.progress(
                55,
                text=f"Executing {atk_name} optimization ({eval_sample_count} samples)...",
            )
            attack_result = execute_configured_attack(
                model=model_adapter,
                attack_name=atk_id,
                inputs=eval_inputs,
                labels=eval_labels,
                epsilon=epsilon,
                clip_min=clip_min,
                clip_max=clip_max,
                params=params_dict,
            )

            atk_prog.progress(
                80,
                text="Evaluating clean vs. adversarial predictions and classification confidences...",
            )
            time.sleep(0.08)

            atk_prog.progress(
                90,
                text="Executing M5 Vulnerability Engine assessment & multi-vector scoring...",
            )
            time.sleep(0.08)

            # Compute authentic M5 Vulnerability Assessment & Scoring (A11, A12, A13)
            vuln_result = assess_and_score_vulnerability(
                model=model_adapter,
                attack_result=attack_result,
                inputs=eval_inputs,
                labels=eval_labels,
                domain=domain,
                dataset_name=active_data.get("dataset_name"),
            )
            attack_result["vulnerability_assessment"] = vuln_result
            attack_result["vulnerability_score"] = vuln_result["vulnerability_score"]
            attack_result["risk_level"] = vuln_result["risk_level"]

            atk_prog.progress(
                100,
                text="Computing perturbation distortion metrics (L_inf, L_2, L_0, MAE)...",
            )
            time.sleep(0.1)
            atk_prog.empty()

            set_vulnerability_result(vuln_result)
            set_attack_result(attack_result)
            set_workflow_stage("attack_results")
            st.rerun()
        except Exception as exc:
            atk_prog.empty()
            st.error(f"❌ Attack Execution Failed: {str(exc)}")


# ==============================================================================
# STAGE 4: ATTACK EXECUTION RESULTS
# ==============================================================================
elif stage == "attack_results":
    st.markdown("## 📊 Adversarial Attack Results & Robustness Telemetry")
    st.caption(
        "Step 3 of 3 • Actual outputs, prediction shifts, and perturbation metrics from backend attack engine"
    )

    result = get_attack_result()

    if not result:
        st.warning("No attack results found in session memory.")
        if st.button("Return to Attack Configuration", type="primary"):
            set_workflow_stage("attack_config")
            st.rerun()
        st.stop()

    active_data = get_active_assessment() or {}

    # Synthetic fallback alert (A05)
    if active_data.get("is_synthetic_dataset"):
        st.warning(
            "⚠️ **Notice: Evaluation executed on Synthetic Fallback Data.** "
            "This experiment reflects model behavior on generated synthetic samples, not a real dataset benchmark."
        )

    asr = result.get("asr_percentage", 0.0)
    clean_acc = result.get("clean_accuracy", 0.0)
    adv_acc = result.get("adversarial_accuracy", 0.0)
    acc_drop = result.get("accuracy_drop", 0.0)

    clean_conf = result.get("clean_confidence", 0.0)
    adv_conf = result.get("adversarial_confidence", 0.0)
    conf_drop = result.get("confidence_drop", 0.0)

    # Normalize values to percentages [0..100] regardless of whether backend returned fractions or percentages
    def _to_pct_val(val: float | None) -> float:
        if val is None:
            return 0.0
        if 0.0 < abs(val) <= 1.0:
            return val * 100.0
        return float(val)

    clean_acc_disp = _to_pct_val(clean_acc)
    adv_acc_disp = _to_pct_val(adv_acc)
    acc_drop_disp = _to_pct_val(acc_drop)

    clean_conf_disp = _to_pct_val(clean_conf)
    adv_conf_disp = _to_pct_val(adv_conf)
    conf_drop_disp = _to_pct_val(conf_drop)

    exec_time = result.get("execution_time", 0.0)
    ms_sample = result.get("ms_per_sample", 0.0)
    succ_cnt = result.get("successful_count", 0)
    tot_samples = result.get("total_samples", 0)
    clean_correct_cnt = result.get("clean_correct_count", tot_samples)
    is_targeted_run = bool(result.get("is_targeted", False))
    target_label_val = result.get("target_label", None)
    asr_undef = result.get("asr_undefined", False)

    # Retrieve authentic M5 vulnerability assessment and scoring result (A11, A12, A13)
    vuln_res = get_vulnerability_result() or result.get("vulnerability_assessment") or {}
    vuln_score = float(vuln_res.get("vulnerability_score", result.get("vulnerability_score", 0.0)))
    risk_level = str(vuln_res.get("risk_level", result.get("risk_level", "LOW"))).upper()

    # 1. 4 Primary Top-line Modern Stat Cards
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)

    with col_r1:
        if asr_undef:
            asr_subtext = "0 clean samples correct (undefined)"
            asr_val_str = "N/A"
            asr_bar_pct = 0.0
        elif is_targeted_run:
            asr_subtext = f"<strong>{succ_cnt}</strong> of <strong>{clean_correct_cnt}</strong> initially correct reached target {target_label_val}"
            asr_val_str = f"{asr:.1f}%"
            asr_bar_pct = min(100.0, max(0.0, asr))
        else:
            asr_subtext = f"<strong>{succ_cnt}</strong> of <strong>{clean_correct_cnt}</strong> initially correct samples flipped"
            asr_val_str = f"{asr:.1f}%"
            asr_bar_pct = min(100.0, max(0.0, asr))

        st.markdown(
            f"""
            <div class="modern-card card-accent-rose">
                <div class="stat-label">{'Targeted Success Rate' if is_targeted_run else 'Attack Success Rate (ASR)'}</div>
                <div class="stat-value" style="color: var(--stat-val-rose);">{asr_val_str}</div>
                <div style="background: rgba(128, 128, 128, 0.2); border-radius: 9999px; height: 5px; width: 100%; margin: 6px 0; overflow: hidden;">
                    <div style="background: #f43f5e; width: {asr_bar_pct:.1f}%; height: 100%; border-radius: 9999px;"></div>
                </div>
                <div class="stat-subtext">{asr_subtext}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_r2:
        vuln_color = "#f43f5e" if risk_level in ["CRITICAL", "HIGH"] else ("#f59e0b" if risk_level == "MEDIUM" else "#10b981")
        st.markdown(
            f"""
            <div class="modern-card card-accent-rose">
                <div class="stat-label">Vulnerability Score (M5)</div>
                <div class="stat-value" style="color: {vuln_color};">{vuln_score:.1f}<span style="font-size: 1rem; color: var(--text-secondary);">/100</span></div>
                <div style="background: rgba(128, 128, 128, 0.2); border-radius: 9999px; height: 5px; width: 100%; margin: 6px 0; overflow: hidden;">
                    <div style="background: {vuln_color}; width: {min(100.0, max(0.0, vuln_score)):.1f}%; height: 100%; border-radius: 9999px;"></div>
                </div>
                <div class="stat-subtext">Risk Severity: <strong class="risk-badge risk-{risk_level.lower()}" style="font-size: 0.72rem; padding: 2px 6px;">{risk_level}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_r3:
        acc_drop_pct = min(100.0, max(0.0, abs(acc_drop_disp)))
        st.markdown(
            f"""
            <div class="modern-card card-accent-amber">
                <div class="stat-label">Accuracy Degradation</div>
                <div class="stat-value" style="color: var(--stat-val-amber);">{clean_acc_disp:.1f}% → {adv_acc_disp:.1f}%</div>
                <div style="background: rgba(128, 128, 128, 0.2); border-radius: 9999px; height: 5px; width: 100%; margin: 6px 0; overflow: hidden;">
                    <div style="background: #f59e0b; width: {acc_drop_pct:.1f}%; height: 100%; border-radius: 9999px;"></div>
                </div>
                <div class="stat-subtext">Accuracy Drop: <strong style="color: var(--stat-val-rose);">-{abs(acc_drop_disp):.1f}%</strong> (Conf: -{abs(conf_drop_disp):.1f}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_r4:
        st.markdown(
            f"""
            <div class="modern-card card-accent-cyan">
                <div class="stat-label">Execution Telemetry</div>
                <div class="stat-value" style="color: var(--stat-val-cyan);">{exec_time:.3f}s</div>
                <div style="background: rgba(128, 128, 128, 0.2); border-radius: 9999px; height: 5px; width: 100%; margin: 6px 0; overflow: hidden;">
                    <div style="background: #06b6d4; width: 100%; height: 100%; border-radius: 9999px;"></div>
                </div>
                <div class="stat-subtext">Throughput: <strong>{ms_sample:.1f} ms/sample</strong> ({tot_samples} samples)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Dedicated M5 Vulnerability Analysis & Metric Breakdown Section (A11, A12, A13, A14)
    st.markdown(
        """
        <div class="section-banner">
            <h3>🛡️ Module 5 — Vulnerability Analysis & Multi-Vector Scoring</h3>
            <p>Empirical multi-vector vulnerability scoring computed directly by the backend VulnerabilityEngine</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_v1, col_v2, col_v3 = st.columns(3)
    sub_scores = vuln_res.get("sub_scores", {})
    asr_sc = float(sub_scores.get("asr_score", vuln_res.get("asr_score", asr)))
    deg_sc = float(sub_scores.get("degradation_score", vuln_res.get("degradation_score", 0.0)))
    stl_sc = float(sub_scores.get("stealth_score", vuln_res.get("stealth_score", 0.0)))
    weights = vuln_res.get("weights", {"weight_asr": 0.50, "weight_degradation": 0.35, "weight_stealth": 0.15})
    w_asr = float(weights.get("weight_asr", 0.50))
    w_deg = float(weights.get("weight_degradation", 0.35))
    w_stl = float(weights.get("weight_stealth", 0.15))

    with col_v1:
        st.markdown(
            f"""
            <div class="modern-card card-accent-rose">
                <div class="stat-label">ASR Sub-Score (Weight: {w_asr:.0%})</div>
                <div style="font-size: 1.45rem; font-weight: 700; color: var(--text-primary);">{asr_sc:.1f} <span style="font-size: 0.95rem; color: var(--text-secondary);">/ 100</span></div>
                <div style="background: rgba(128, 128, 128, 0.2); border-radius: 9999px; height: 6px; width: 100%; margin: 8px 0; overflow: hidden;">
                    <div style="background: #f43f5e; width: {min(100.0, max(0.0, asr_sc)):.1f}%; height: 100%; border-radius: 9999px;"></div>
                </div>
                <div class="stat-subtext">Composite Contribution: <strong style="color: var(--stat-val-rose);">+{(asr_sc * w_asr):.1f} pts</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_v2:
        st.markdown(
            f"""
            <div class="modern-card card-accent-amber">
                <div class="stat-label">Degradation Sub-Score (Weight: {w_deg:.0%})</div>
                <div style="font-size: 1.45rem; font-weight: 700; color: var(--text-primary);">{deg_sc:.1f} <span style="font-size: 0.95rem; color: var(--text-secondary);">/ 100</span></div>
                <div style="background: rgba(128, 128, 128, 0.2); border-radius: 9999px; height: 6px; width: 100%; margin: 8px 0; overflow: hidden;">
                    <div style="background: #f59e0b; width: {min(100.0, max(0.0, deg_sc)):.1f}%; height: 100%; border-radius: 9999px;"></div>
                </div>
                <div class="stat-subtext">Composite Contribution: <strong style="color: var(--stat-val-amber);">+{(deg_sc * w_deg):.1f} pts</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_v3:
        st.markdown(
            f"""
            <div class="modern-card card-accent-violet">
                <div class="stat-label">Stealthiness Sub-Score (Weight: {w_stl:.0%})</div>
                <div style="font-size: 1.45rem; font-weight: 700; color: var(--text-primary);">{stl_sc:.1f} <span style="font-size: 0.95rem; color: var(--text-secondary);">/ 100</span></div>
                <div style="background: rgba(128, 128, 128, 0.2); border-radius: 9999px; height: 6px; width: 100%; margin: 8px 0; overflow: hidden;">
                    <div style="background: #8b5cf6; width: {min(100.0, max(0.0, stl_sc)):.1f}%; height: 100%; border-radius: 9999px;"></div>
                </div>
                <div class="stat-subtext">Composite Contribution: <strong style="color: #8b5cf6;">+{(stl_sc * w_stl):.1f} pts</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Vulnerability Scoring Vectors & Empirical Telemetry Table
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    st.markdown("##### 📋 Vulnerability Score Vectors & Empirical Telemetry")

    metrics = result.get("perturbation_metrics", {})
    linf_val_prev = metrics.get("linf_mean") if metrics.get("linf_mean") is not None else metrics.get("linf", 0.0)
    l2_val_prev = metrics.get("l2_mean") if metrics.get("l2_mean") is not None else metrics.get("l2", 0.0)
    linf_str = f"{float(linf_val_prev):.4f}" if metrics else "N/A"
    l2_str = f"{float(l2_val_prev):.3f}" if metrics else "N/A"

    vuln_table_data = [
        {
            "Score Vector Component": "Attack Success Rate (ASR)",
            "Empirical Metric Value": f"{asr:.1f}% flipped ({succ_cnt}/{clean_correct_cnt} clean-correct samples)",
            "Sub-Score (/100)": f"{asr_sc:.1f}",
            "Engine Weight": f"{w_asr:.0%}",
            "Composite Contribution": f"+{(asr_sc * w_asr):.1f} pts",
            "Measurement Basis": "Susceptibility of initially correct baseline inputs to adversarial manipulation",
        },
        {
            "Score Vector Component": "Model Performance Degradation",
            "Empirical Metric Value": f"Acc Drop: -{abs(acc_drop_disp):.1f}% | Conf Drop: -{abs(conf_drop_disp):.1f}%",
            "Sub-Score (/100)": f"{deg_sc:.1f}",
            "Engine Weight": f"{w_deg:.0%}",
            "Composite Contribution": f"+{(deg_sc * w_deg):.1f} pts",
            "Measurement Basis": "Empirical F1-score erosion and model confidence collapse under perturbation",
        },
        {
            "Score Vector Component": "Adversarial Stealthiness",
            "Empirical Metric Value": f"L_inf: {linf_str} | L2: {l2_str}",
            "Sub-Score (/100)": f"{stl_sc:.1f}",
            "Engine Weight": f"{w_stl:.0%}",
            "Composite Contribution": f"+{(stl_sc * w_stl):.1f} pts",
            "Measurement Basis": "Imperceptibility scaling (higher stealthiness with successful evasion amplifies threat severity)",
        },
        {
            "Score Vector Component": "Composite Vulnerability Score",
            "Empirical Metric Value": f"Risk Level: {risk_level}",
            "Sub-Score (/100)": f"{vuln_score:.1f}",
            "Engine Weight": "100%",
            "Composite Contribution": f"{vuln_score:.1f} / 100",
            "Measurement Basis": "Standardized multi-vector risk severity index evaluated by backend VulnerabilityEngine",
        },
    ]
    st.table(pd.DataFrame(vuln_table_data))

    # Metric Calculation Rationale & M5 Findings Box
    explanations = vuln_res.get("metric_explanations") or {}
    st.markdown("##### 📖 Metric Calculation & Severity Findings")
    st.markdown(
        f"""
        <div class="param-explainer" style="line-height: 1.65; font-size: 0.9rem; padding: 16px; border-radius: 8px;">
            <div style="margin-bottom: 8px;"><strong>Executive Summary:</strong> {explanations.get('summary', 'Standardized M5 Vulnerability Engine evaluation.')}</div>
            <div style="margin-bottom: 4px;">• <strong>ASR Metric Component:</strong> {explanations.get('asr_component', f'ASR Score: {asr_sc:.1f}/100')}</div>
            <div style="margin-bottom: 4px;">• <strong>Degradation Metric Component:</strong> {explanations.get('degradation_component', f'Degradation Score: {deg_sc:.1f}/100')}</div>
            <div style="margin-bottom: 4px;">• <strong>Stealthiness Metric Component:</strong> {explanations.get('stealth_component', f'Stealthiness Score: {stl_sc:.1f}/100')}</div>
            <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border-color);">
                <strong>Risk Severity Determination:</strong> {explanations.get('risk_rationale', f'Severity assigned as {risk_level} based on multi-vector empirical evaluation.')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # 2. Perturbation Distance Metrics (matching run_pipeline_for_multimodel.py)
    st.markdown(
        """
        <div class="section-banner">
            <h3>📏 Perturbation Distortion Norms & Geometric Bounds</h3>
            <p>Mathematical distance between clean inputs and adversarial examples across Lp norms</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if metrics:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        with col_m1:
            eps_budget = float(result.get("epsilon_used", 0.05))
            linf_val = float(metrics.get("linf_mean") if metrics.get("linf_mean") is not None else metrics.get("linf", 0.0))
            linf_max_val = float(metrics.get("linf_max", linf_val))
            linf_bar = min(100.0, max(0.0, (linf_val / max(1e-6, eps_budget)) * 100.0)) if eps_budget > 0 else 0.0
            st.markdown(
                f"""
                <div class="modern-card card-accent-cyan">
                    <div class="stat-label">L-infinity Distance (L_inf)</div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: var(--text-primary);">{linf_val:.5f}</div>
                    <div style="background: rgba(128, 128, 128, 0.2); border-radius: 9999px; height: 5px; width: 100%; margin: 6px 0; overflow: hidden;">
                        <div style="background: #06b6d4; width: {linf_bar:.1f}%; height: 100%; border-radius: 9999px;"></div>
                    </div>
                    <div class="stat-subtext">Budget (ε): <strong>{eps_budget:.3f}</strong> • Max: <strong>{linf_max_val:.4f}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_m2:
            l2_val = float(metrics.get("l2_mean") if metrics.get("l2_mean") is not None else metrics.get("l2", 0.0))
            l2_max_val = float(metrics.get("l2_max", l2_val))
            l2_bar = min(100.0, max(0.0, l2_val * 25.0))
            st.markdown(
                f"""
                <div class="modern-card card-accent-emerald">
                    <div class="stat-label">L2 Euclidean Distance</div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: var(--text-primary);">{l2_val:.4f}</div>
                    <div style="background: rgba(128, 128, 128, 0.2); border-radius: 9999px; height: 5px; width: 100%; margin: 6px 0; overflow: hidden;">
                        <div style="background: #10b981; width: {l2_bar:.1f}%; height: 100%; border-radius: 9999px;"></div>
                    </div>
                    <div class="stat-subtext">Mean Euclidean • Max: <strong>{l2_max_val:.3f}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_m3:
            l0_val = int(metrics.get("l0_mean_count") if metrics.get("l0_mean_count") is not None else metrics.get("l0", 0))
            l0_pct = metrics.get("l0_mean_pct") if metrics.get("l0_mean_pct") is not None else metrics.get("l0_pct")
            if l0_pct is None or l0_pct == 0.0:
                test_inputs = active_data.get("test_inputs")
                if (
                    test_inputs is not None
                    and hasattr(test_inputs, "numel")
                    and test_inputs.size(0) > 0
                ):
                    elem_per_sample = test_inputs[0].numel()
                    l0_pct = (
                        min(100.0, (l0_val / elem_per_sample * 100.0))
                        if elem_per_sample > 0
                        else 0.0
                    )
                else:
                    l0_pct = 0.0

            st.markdown(
                f"""
                <div class="modern-card card-accent-violet">
                    <div class="stat-label">L0 Perturbed Elements</div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: var(--text-primary);">{l0_val:,}</div>
                    <div style="background: rgba(128, 128, 128, 0.2); border-radius: 9999px; height: 5px; width: 100%; margin: 6px 0; overflow: hidden;">
                        <div style="background: #8b5cf6; width: {min(100.0, max(0.0, l0_pct)):.1f}%; height: 100%; border-radius: 9999px;"></div>
                    </div>
                    <div class="stat-subtext"><strong>{l0_pct:.1f}%</strong> of total tensor elements changed</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_m4:
            mae_val = float(metrics.get("mae_mean") if metrics.get("mae_mean") is not None else metrics.get("mean_abs_error", 0.0))
            mae_bar = min(100.0, max(0.0, mae_val * 50.0))
            st.markdown(
                f"""
                <div class="modern-card card-accent-amber">
                    <div class="stat-label">Mean Absolute Error (MAE)</div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: var(--text-primary);">{mae_val:.5f}</div>
                    <div style="background: rgba(128, 128, 128, 0.2); border-radius: 9999px; height: 5px; width: 100%; margin: 6px 0; overflow: hidden;">
                        <div style="background: #f59e0b; width: {mae_bar:.1f}%; height: 100%; border-radius: 9999px;"></div>
                    </div>
                    <div class="stat-subtext">Average absolute element change</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Perturbation Distortion Norms Table
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("##### 📐 Perturbation Norms & Geometric Bounds")
        norms_table_data = [
            {
                "Norm Dimension": "L-infinity (Chebyshev Norm)",
                "Empirical Value": f"{linf_val:.5f}",
                "Reference Budget / Bound": f"ε = {eps_budget:.4f}",
                "Modification Scope": "Maximum single-element displacement across all dimensions",
                "Stealth & Physical Meaning": "Perceptibility ceiling; guarantees no individual pixel or feature shifts beyond this bound",
            },
            {
                "Norm Dimension": "L2 (Euclidean Norm)",
                "Empirical Value": f"{l2_val:.4f}",
                "Reference Budget / Bound": "Global energy displacement",
                "Modification Scope": "Root-mean-square displacement across the input tensor",
                "Stealth & Physical Meaning": "Measures total physical perturbation energy; lower indicates more stealthy distribution",
            },
            {
                "Norm Dimension": "L0 (Sparsity / Hamming Norm)",
                "Empirical Value": f"{l0_val:,} elements",
                "Reference Budget / Bound": f"{l0_pct:.1f}% of total tensor elements",
                "Modification Scope": f"{l0_val} modified elements",
                "Stealth & Physical Meaning": "Sparsity of the attack; shows whether modifications are localized or uniformly dispersed",
            },
            {
                "Norm Dimension": "MAE (Mean Absolute Error)",
                "Empirical Value": f"{mae_val:.5f}",
                "Reference Budget / Bound": "Average absolute element shift",
                "Modification Scope": "Element-wise average deviation",
                "Stealth & Physical Meaning": "Direct measure of expected alteration per feature across the entire evaluated batch",
            },
        ]
        st.table(pd.DataFrame(norms_table_data))
    else:
        st.info("Perturbation metrics not applicable for this data modality.")

    # 3. Actionable Security & Hardening Recommendations Section
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-banner">
            <h3>💡 Actionable Security & Hardening Recommendations</h3>
            <p>Domain-specific, prioritized mitigation strategies based on empirical attack susceptibility and vulnerability scoring</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    domain = str(active_data.get("domain", "image")).lower()
    recommendations = vuln_res.get("recommendations")
    if not recommendations:
        recommendations = generate_vulnerability_recommendations(
            domain=domain,
            attack_name=str(result.get("attack_name", "attack")),
            vuln_score=vuln_score,
            risk_level=risk_level,
            asr=asr,
            is_targeted=is_targeted_run,
        )

    rec_cols = st.columns(len(recommendations)) if 1 <= len(recommendations) <= 4 else st.columns(3)
    for i, rec in enumerate(recommendations):
        col_idx = i % len(rec_cols)
        prio = str(rec.get("priority", "Recommended")).capitalize()
        if prio.lower() == "urgent":
            prio_badge = '<span class="badge-status badge-rose">🚨 URGENT</span>'
            card_accent = "card-accent-rose"
        elif prio.lower() == "high":
            prio_badge = '<span class="badge-status badge-amber">⚠️ HIGH PRIORITY</span>'
            card_accent = "card-accent-amber"
        else:
            prio_badge = '<span class="badge-status badge-emerald">🛡️ RECOMMENDED</span>'
            card_accent = "card-accent-emerald"

        with rec_cols[col_idx]:
            st.markdown(
                f"""
                <div class="modern-card {card_accent}" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="font-size: 0.82rem; font-weight: 600; color: var(--text-secondary); text-transform: uppercase;">
                                {rec.get('icon', '🛡️')} {rec.get('category', 'Hardening')}
                            </span>
                            {prio_badge}
                        </div>
                        <h4 style="margin: 0 0 8px 0; font-size: 1.05rem; font-weight: 700; color: var(--text-primary);">
                            {rec.get('title', 'Hardening Recommendation')}
                        </h4>
                        <p style="margin: 0; font-size: 0.88rem; line-height: 1.5; color: var(--text-secondary);">
                            {rec.get('description', '')}
                        </p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 4. Hyperparameter Configuration & Execution Telemetry Table
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-banner">
            <h3>⚙️ Attack Telemetry & Hyperparameters</h3>
            <p>Detailed runtime environment parameters matching framework evaluation scripts</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    params_used = result.get("parameters_used") or result.get("parameters") or {}
    is_targeted_run = bool(params_used.get("targeted", False) or result.get("is_targeted", False))
    tgt_lbl = params_used.get("target_label", result.get("target_label"))
    tgt_display = f"True (Target Class: {tgt_lbl})" if is_targeted_run and tgt_lbl is not None else ("True" if is_targeted_run else "False (Untargeted)")
    telemetry_rows = [
        {
            "Parameter": "Attack Algorithm",
            "Value": str(result.get("attack_name", "N/A")),
        },
        {
            "Parameter": "Vulnerability Score (M5)",
            "Value": f"{vuln_score:.2f} / 100 ({risk_level})",
        },
        {
            "Parameter": "Perturbation Budget (ε)",
            "Value": f"{result.get('epsilon_used', 0.0):.4f}",
        },
        {"Parameter": "Clip Minimum", "Value": str(params_used.get("clip_min", "0.0"))},
        {"Parameter": "Clip Maximum", "Value": str(params_used.get("clip_max", "1.0"))},
        {
            "Parameter": "Optimization Iterations",
            "Value": str(params_used.get("num_steps", "1 (Single-step)")),
        },
        {"Parameter": "Step Size (α)", "Value": str(params_used.get("alpha", "N/A"))},
        {
            "Parameter": "Targeted Attack",
            "Value": tgt_display,
        },
        {"Parameter": "Evaluated Batch Size", "Value": f"{tot_samples} samples"},
        {"Parameter": "Clean Correct Baseline", "Value": f"{clean_correct_cnt} of {tot_samples} samples"},
        {
            "Parameter": "Execution Device",
            "Value": str(active_data.get("device", "CPU")),
        },
    ]
    st.table(pd.DataFrame(telemetry_rows))

    # 5. Sample-by-Sample Prediction & Confidence Table
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-banner">
            <h3>🔬 Sample-by-Sample Assessment</h3>
            <p>Comparison of clean predictions vs. adversarial predictions with confidence shifts</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    comparisons = result.get("sample_comparisons", [])
    if comparisons:
        # 1. Primary: Use verified ground-truth counts directly from backend result
        succ_o = int(result.get("successful_count", 0))
        clean_correct = int(result.get("clean_correct_count", tot_samples))
        inv_o = max(0, clean_correct - succ_o)
        mis_o = max(0, tot_samples - clean_correct)

        # 2. Secondary fallback verification from sample_comparisons rows if result counts missing
        if succ_o == 0 and inv_o == 0 and mis_o == 0 and len(comparisons) > 0:
            atk_outcomes = [str(c.get("Outcome", c.get("outcome", ""))) for c in comparisons]
            succ_o = sum(1 for o in atk_outcomes if any(k in o for k in ("Flip", "Targeted Hit", "Successful")))
            inv_o = sum(1 for o in atk_outcomes if any(k in o for k in ("Preserved", "Invariant", "Resistant", "Clean Invariant")))
            mis_o = sum(1 for o in atk_outcomes if any(k in o for k in ("Initial Clean Error", "Misclassified", "Error")))

        col_o1, col_o2, col_o3 = st.columns(3)

        with col_o1:
            st.markdown(
                f"""
                <div class="modern-card card-accent-rose" style="padding: 10px 14px;">
                    <div class="stat-label">Successful Adversarial Flips</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: var(--stat-val-rose);">{succ_o} / {tot_samples} samples</div>
                    <div class="stat-subtext">Susceptible to perturbation</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_o2:
            st.markdown(
                f"""
                <div class="modern-card card-accent-emerald" style="padding: 10px 14px;">
                    <div class="stat-label">Clean Invariant (Resistant)</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: var(--stat-val-emerald);">{inv_o} / {tot_samples} samples</div>
                    <div class="stat-subtext">Maintained correct prediction</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_o3:
            st.markdown(
                f"""
                <div class="modern-card card-accent-cyan" style="padding: 10px 14px;">
                    <div class="stat-label">Initially Misclassified</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: var(--stat-val-cyan);">{mis_o} / {tot_samples} samples</div>
                    <div class="stat-subtext">Excluded from ASR denominator</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        df_comp = pd.DataFrame(comparisons)
        # Rename columns for clean human presentation
        display_cols = {
            "sample_idx": "Sample #",
            "true_label": "Ground Truth",
            "clean_pred": "Clean Pred",
            "clean_conf": "Clean Conf",
            "adv_pred": "Adv Pred",
            "adv_conf": "Adv Conf",
            "outcome": "Assessment Outcome",
        }
        df_comp = df_comp.rename(columns=display_cols)
        st.table(df_comp)
    else:
        st.info("No per-sample prediction comparisons available.")

    st.markdown("---")

    # Workflow Actions
    col_back, col_fwd = st.columns([1, 1])

    with col_back:
        if st.button("⬅️ Test Another Attack on this Model", use_container_width=True):
            set_workflow_stage("attack_config")
            st.rerun()

    with col_fwd:
        if st.button("🛡️ Apply Defense Technique ➡️", type="primary", use_container_width=True):
            set_workflow_stage("defense_config")
            st.rerun()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    if st.button(
        "🔄 Start New Assessment (Reset)", use_container_width=True, key="reset_from_attack_results"
    ):
        reset_workflow_state()
        set_workflow_stage("dashboard")
        st.rerun()


# ==============================================================================
# STAGE 5: DEFENSE CONFIGURATION & HARDENING SELECTION
# ==============================================================================
elif stage == "defense_config":
    st.markdown("## 🛡️ Model Hardening & Defense Configuration")
    st.caption(
        "Step 4 of 6 • Configure deployment-time defenses and iterative hardening for adversarial mitigation"
    )

    active_data = get_active_assessment()
    attack_res = get_attack_result()

    if not active_data or not attack_res:
        st.warning("No active attack results found to harden against.")
        if st.button("Return to Attack Configuration", type="primary"):
            set_workflow_stage("attack_config")
            st.rerun()
        st.stop()

    model_name = active_data.get("metadata_dict", {}).get("Model Name") or "Active Model"
    domain = active_data.get("domain", "image")
    adapter = active_data.get("adapter")
    test_inputs = active_data.get("test_inputs")
    test_labels = active_data.get("test_labels")
    adv_inputs = attack_res.get("_adversarial_examples")

    atk_name = attack_res.get("attack_name", "Adversarial Attack")
    asr_pct = attack_res.get("asr_percentage", 0.0)
    acc_drop = attack_res.get("accuracy_drop", 0.0)
    vuln_res = get_vulnerability_result() or attack_res.get("vulnerability_assessment") or {}
    vuln_score = float(vuln_res.get("vulnerability_score", attack_res.get("vulnerability_score", asr_pct)))
    risk_level = str(vuln_res.get("risk_level", attack_res.get("risk_level", "HIGH" if asr_pct >= 50.0 else "LOW"))).upper()

    # 1. Threat Context Banner
    st.markdown(
        f"""
        <div class="modern-card card-accent-rose" style="margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
                <div>
                    <div class="stat-label">Identified Vulnerability Context (M5)</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: var(--text-primary); margin-top: 0.2rem;">
                        {model_name} • Attacked with {atk_name}
                    </div>
                </div>
                <div style="display: flex; gap: 0.75rem; flex-wrap: wrap;">
                    <span class="risk-badge risk-{risk_level.lower()}">Risk Level: {risk_level}</span>
                    <span class="info-badge">Vuln Score: {vuln_score:.1f}/100</span>
                    <span class="info-badge">ASR: {asr_pct:.1f}%</span>
                    <span class="info-badge">Accuracy Drop: -{abs(acc_drop):.1f}%</span>
                </div>
            </div>
            <div style="margin-top: 0.6rem; font-size: 0.86rem; color: var(--text-secondary);">
                Hardening applies pre-inference transformations, stochastic smoothing, or detection filters to restore robust predictions without requiring retraining.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Defense Catalog & Mode Selection
    available_defenses = get_available_defenses_for_domain(domain)
    defense_options = {d["id"]: d for d in available_defenses}

    st.markdown(
        """
        <div class="section-banner">
            <h3>🛡️ Defense Technique Selection</h3>
            <p>Choose an automated iterative auto-tuning search or a dedicated mitigation algorithm</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    def_mode = st.radio(
        "Hardening Strategy",
        options=["auto", "manual"],
        format_func=lambda x: (
            "✨ Auto-Iterative Defense Selection (Evaluates candidate defenses and selects best)"
            if x == "auto"
            else "🎯 Specific Defense Technique (Manual configuration)"
        ),
        index=0,
    )

    selected_defense_id = "auto"
    custom_params: Dict[str, Any] = {}

    if def_mode == "manual":
        # Filter out auto from manual list
        manual_candidates = [d for d in available_defenses if not d.get("is_auto")]
        man_ids = [d["id"] for d in manual_candidates]

        selected_defense_id = st.selectbox(
            "Select Defense Algorithm",
            options=man_ids,
            format_func=lambda d_id: (
                f"{defense_options[d_id]['name']} — [{defense_options[d_id]['family']}]"
            ),
        )

        chosen_def = defense_options[selected_defense_id]
        st.info(f"**{chosen_def['name']}**: {chosen_def['description']}")

        # Render custom parameter sliders if defined
        if chosen_def.get("param_specs"):
            st.markdown("##### ⚙️ Hyperparameter Configuration")
            param_cols = st.columns(len(chosen_def["param_specs"]))
            for idx, p_spec in enumerate(chosen_def["param_specs"]):
                with param_cols[idx]:
                    p_type = p_spec.get("type", "slider")
                    p_name = p_spec["name"]
                    p_label = p_spec.get("label", p_name)
                    p_help = p_spec.get("help", "")

                    if p_type == "slider":
                        custom_params[p_name] = st.slider(
                            p_label,
                            min_value=p_spec["min_value"],
                            max_value=p_spec["max_value"],
                            value=p_spec.get("default"),
                            step=p_spec.get("step", 1),
                            help=p_help,
                        )
                    elif p_type == "select":
                        custom_params[p_name] = st.selectbox(
                            p_label,
                            options=p_spec["options"],
                            index=p_spec["options"].index(p_spec.get("default", p_spec["options"][0])),
                            help=p_help,
                        )
    else:
        st.markdown(
            """
            <div class="modern-card card-accent-cyan" style="margin-bottom: 1rem;">
                <div style="font-size: 0.95rem; font-weight: 600; color: var(--text-primary);">
                    🤖 Automated Iterative Hardening Engine
                </div>
                <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.3rem;">
                    The engine will systematically evaluate multiple defense candidates for this model. Each candidate is independently applied and re-tested against empirical acceptance criteria:
                    <ul style="margin-top: 0.4rem; margin-bottom: 0;">
                        <li><strong>Vulnerability Score Reduction</strong>: Requires positive improvement (&gt; 0 pts).</li>
                        <li><strong>Attack Success Rate Reduction</strong>: Evaluates perturbation suppression ratio.</li>
                        <li><strong>Clean Accuracy Preservation</strong>: Ensures clean utility drop remains below allowable thresholds.</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Fallback to Most Effective Candidate Checkbox
    fallback_to_best = st.checkbox(
        "⭐ Select most effective candidate if strict ASR reduction ratio is not met",
        value=True,
        help=(
            "When enabled, if none of the candidate defenses satisfy strict acceptance ratio thresholds "
            "(e.g. on highly aggressive attacks), the engine selects the candidate achieving the highest "
            "vulnerability reduction while preserving clean accuracy."
        ),
    )

    st.markdown("---")

    # Workflow Navigation (Backward left, Forward right, Reset below)
    col_back, col_fwd = st.columns([1, 1])

    with col_back:
        if st.button("⬅️ Back to Attack Results", use_container_width=True):
            set_workflow_stage("attack_results")
            st.rerun()

    with col_fwd:
        apply_btn = st.button(
            "🚀 Apply Defense & Harden Model ➡️",
            type="primary",
            use_container_width=True,
        )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    if st.button(
        "🔄 Start New Assessment (Reset)", use_container_width=True, key="reset_from_defense_config"
    ):
        reset_workflow_state()
        set_workflow_stage("dashboard")
        st.rerun()

    if apply_btn:
        progress_bar = st.progress(0, text="Initializing Hardening Engine and defense candidates...")
        try:
            time.sleep(0.05)
            progress_bar.progress(25, text="Applying defense algorithm to target model...")
            time.sleep(0.05)
            progress_bar.progress(50, text="Evaluating clean accuracy preservation and latency...")
            time.sleep(0.05)
            progress_bar.progress(75, text="Re-testing under adversarial conditions & computing ASR reduction...")

            # Run defense execution service
            defense_res = execute_configured_defense(
                model=adapter,
                domain=domain,
                defense_choice=selected_defense_id,
                test_inputs=test_inputs,
                test_labels=test_labels,
                adversarial_examples=adv_inputs,
                attack_result=attack_res,
                fallback_to_best=fallback_to_best,
                custom_params=custom_params,
            )

            progress_bar.progress(100, text="Hardening complete! Loading re-test telemetry...")
            time.sleep(0.08)
            progress_bar.empty()

            set_defense_result(defense_res)
            set_workflow_stage("defense_results")
            st.rerun()
        except Exception as exc:
            progress_bar.empty()
            st.error(f"❌ Hardening Execution Failed: {str(exc)}")


# ==============================================================================
# STAGE 6: DEFENSE RESULTS & HARDENING DASHBOARD
# ==============================================================================
elif stage == "defense_results":
    st.markdown("## 🛡️ Hardening & Empirical Re-Test Dashboard")
    st.caption(
        "Step 5 of 6 • Verification of applied defense mitigations, prediction restoration, and residual risk"
    )

    def_result = get_defense_result()
    attack_res = get_attack_result()
    active_data = get_active_assessment() or {}

    if not def_result:
        st.warning("No defense results found in session memory.")
        if st.button("Configure Defense", type="primary"):
            set_workflow_stage("defense_config")
            st.rerun()
        st.stop()

    sel_def = def_result.get("selected_defense", "None")
    sel_def_disp = str(sel_def).replace("_", " ").title()
    imp_suff = def_result.get("improvement_sufficient", False)
    fallback_used = def_result.get("fallback_used", False)
    exec_sec = def_result.get("execution_time_seconds", 0.0)

    # 1. Top Status Banner
    if imp_suff:
        status_badge = '<span class="risk-badge risk-low" style="font-size: 0.9rem;">ACCEPTED — Met Acceptance Thresholds</span>'
        status_note = "All acceptance thresholds for ASR reduction, vulnerability improvement, and clean accuracy preservation were met."
    elif fallback_used:
        status_badge = '<span class="risk-badge risk-medium" style="font-size: 0.9rem;">MOST EFFECTIVE CANDIDATE SELECTED</span>'
        status_note = "None of the candidates satisfied strict ASR reduction ratio thresholds; automatically selected the most effective candidate achieving maximum vulnerability reduction."
    else:
        status_badge = '<span class="risk-badge risk-high" style="font-size: 0.9rem;">EVALUATED</span>'
        status_note = "Defense evaluation completed."

    st.markdown(
        f"""
        <div class="modern-card card-accent-cyan" style="margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
                <div>
                    <div class="stat-label">Active Hardened Defense</div>
                    <div style="font-size: 1.35rem; font-weight: 800; color: var(--text-primary); margin-top: 0.15rem;">
                        🏆 {sel_def_disp}
                    </div>
                </div>
                <div>{status_badge}</div>
            </div>
            <div style="font-size: 0.88rem; color: var(--text-secondary); margin-top: 0.6rem;">
                {status_note}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Key Metrics Cards
    b_asr = def_result.get("before_asr", 0.0)
    a_asr = def_result.get("after_asr", 0.0)
    asr_red = def_result.get("asr_reduction_percentage", 0.0)

    b_adv = def_result.get("before_adv_accuracy", 0.0)
    a_adv = def_result.get("after_adv_accuracy", 0.0)
    adv_gain = a_adv - b_adv

    b_clean = def_result.get("before_clean_accuracy", 100.0)
    a_clean = def_result.get("after_clean_accuracy", 100.0)
    clean_drop = max(0.0, b_clean - a_clean)

    b_vuln = def_result.get("before_vulnerability_score", 0.0)
    a_vuln = def_result.get("after_vulnerability_score", 0.0)
    vuln_imp = def_result.get("vulnerability_improvement", 0.0)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.markdown(
            f"""
            <div class="modern-card card-accent-emerald">
                <div class="stat-label">Attack Success Rate (ASR)</div>
                <div class="stat-value" style="color: var(--stat-val-emerald);">{b_asr:.1f}% → {a_asr:.1f}%</div>
                <div class="stat-subtext">ASR Reduction: <strong style="color: var(--stat-val-emerald);">-{asr_red:.1f}%</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m2:
        st.markdown(
            f"""
            <div class="modern-card card-accent-cyan">
                <div class="stat-label">Defended Accuracy Under Attack</div>
                <div class="stat-value" style="color: var(--stat-val-cyan);">{b_adv:.1f}% → {a_adv:.1f}%</div>
                <div class="stat-subtext">Robustness Recovery: <strong style="color: var(--stat-val-cyan);">+{adv_gain:.1f}%</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m3:
        st.markdown(
            f"""
            <div class="modern-card card-accent-amber">
                <div class="stat-label">Clean Accuracy Preservation</div>
                <div class="stat-value" style="color: var(--stat-val-amber);">{b_clean:.1f}% → {a_clean:.1f}%</div>
                <div class="stat-subtext">Utility Drop: <strong style="color: var(--text-secondary);">{clean_drop:.1f}%</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m4:
        st.markdown(
            f"""
            <div class="modern-card card-accent-violet">
                <div class="stat-label">Vulnerability Score</div>
                <div class="stat-value" style="color: var(--stat-val-violet);">{b_vuln:.1f} → {a_vuln:.1f}</div>
                <div class="stat-subtext">Improvement: <strong style="color: var(--stat-val-emerald);">+{vuln_imp:.1f} pts</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Comparative Telemetry & Visual Comparison
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-banner">
            <h3>📊 Hardening Robustness Metrics & Visual Comparison</h3>
            <p>Empirical before vs. after comparison of accuracy recovery, ASR suppression, and restoration distributions</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("##### 🎯 Classification Accuracy Under Adversarial Stress")
        chart_data_acc = pd.DataFrame(
            {
                "Stage": ["1. Clean Baseline", "2. Under Attack", "3. Defended (Hardened)"],
                "Accuracy (%)": [b_clean, b_adv, a_adv],
            }
        ).set_index("Stage")
        st.bar_chart(chart_data_acc)

    with col_chart2:
        st.markdown("##### 📉 Attack Success Rate (ASR) Suppression")
        chart_data_asr = pd.DataFrame(
            {
                "Assessment Stage": ["Original Attack", "Post-Defense Retest"],
                "ASR (%)": [b_asr, a_asr],
            }
        ).set_index("Assessment Stage")
        st.bar_chart(chart_data_asr)

    # Outcome distribution chart
    sample_restorations = def_result.get("sample_restorations", [])
    if sample_restorations:
        outcomes = [s.get("Outcome", "Unknown") for s in sample_restorations]
        from collections import Counter
        outcome_counts = Counter(outcomes)
        df_outcome_chart = pd.DataFrame(
            {
                "Restoration Outcome": list(outcome_counts.keys()),
                "Number of Samples": list(outcome_counts.values()),
            }
        ).set_index("Restoration Outcome")
        st.markdown("##### 📈 Sample Label Recovery Outcome Distribution")
        st.bar_chart(df_outcome_chart)

    # 4. Candidate Defense Attempts Breakdown & Chart
    candidate_attempts = def_result.get("candidate_attempts", [])
    if candidate_attempts:
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-banner">
                <h3>🔍 Iterative Candidate Defense Evaluation Log</h3>
                <p>Telemetry, charts, and status across all evaluated candidate algorithms</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Candidate comparison chart if candidates exist
        chart_candidates = []
        for att in candidate_attempts:
            try:
                asr_val = float(str(att.get("ASR Reduction", "0")).replace("%", "").strip())
            except Exception:
                asr_val = 0.0
            try:
                drop_val = float(str(att.get("Clean Acc Drop", "0")).replace("%", "").strip())
            except Exception:
                drop_val = 0.0
            chart_candidates.append(
                {
                    "Defense Candidate": str(att.get("Defense", "Unknown")),
                    "ASR Reduction (%)": asr_val,
                    "Clean Acc Drop (%)": drop_val,
                }
            )

        if chart_candidates:
            df_cand_chart = pd.DataFrame(chart_candidates).set_index("Defense Candidate")
            st.markdown("##### 📊 Candidate Defense Performance Comparison")
            st.bar_chart(df_cand_chart)

        df_attempts = pd.DataFrame(candidate_attempts)
        st.table(df_attempts)

    # 5. Sample-by-Sample Restored Prediction Table
    if sample_restorations:
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-banner">
                <h3>🔬 Sample-by-Sample Prediction Restoration</h3>
                <p>Tracking label recovery from adversarial perturbation to hardened output</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        df_restored = pd.DataFrame(sample_restorations)
        st.table(df_restored)

    # 6. Hardening Recommendations
    recs = def_result.get("recommendations", [])
    if recs:
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown("##### 💡 Strategic Hardening Recommendations")
        for r in recs:
            st.markdown(f"• {r}")

    st.markdown("---")

    # Workflow Actions (Backward left, Forward right, Reset below)
    col_back, col_fwd = st.columns([1, 1])

    with col_back:
        if st.button("⬅️ Reconfigure Defense", use_container_width=True):
            set_workflow_stage("defense_config")
            st.rerun()

    with col_fwd:
        if st.button("📄 Generate Security Assessment Report ➡️", type="primary", use_container_width=True):
            set_workflow_stage("report")
            st.rerun()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    if st.button(
        "🔄 Start New Assessment (Reset)", use_container_width=True, key="reset_from_defense_results"
    ):
        reset_workflow_state()
        set_workflow_stage("dashboard")
        st.rerun()


# ==============================================================================
# STAGE 7: DETAILED SECURITY ASSESSMENT REPORT & EXPORT
# ==============================================================================
elif stage == "report":
    st.markdown("## 📑 Comprehensive AI Security Assessment Report")
    st.caption(
        "Step 6 of 6 • Certified End-to-End Adversarial Robustness, Attack Vector & Hardening Evaluation"
    )

    active_data = get_active_assessment() or {}
    attack_res = get_attack_result() or {}
    defense_res = get_defense_result()
    report_res = get_report_result()

    if not attack_res:
        st.warning("No assessment results available to generate report.")
        if st.button("Return to Dashboard", type="primary"):
            set_workflow_stage("dashboard")
            st.rerun()
        st.stop()

    # Generate report DTO if not already cached
    if not report_res:
        with st.spinner("Compiling certified 15-section security assessment report..."):
            model_meta = active_data.get("metadata_dict", {})
            dataset_info = {
                "dataset_name": active_data.get("dataset_name", "Evaluation Batch"),
                "total_samples": attack_res.get("total_samples", 0),
            }
            report_res = generate_security_report(
                model_metadata=model_meta,
                dataset_info=dataset_info,
                attack_result=attack_res,
                defense_result=defense_res,
                domain=active_data.get("domain", "image"),
                vulnerability_result=get_vulnerability_result() or attack_res.get("vulnerability_assessment"),
            )
            set_report_result(report_res)

    report_dict = report_res.to_dict() if hasattr(report_res, "to_dict") else report_res
    formatted_md = (
        getattr(report_res, "formatted_report", "")
        or report_dict.get("formatted_report", "")
    )
    report_json = (
        report_res.to_json()
        if hasattr(report_res, "to_json")
        else json.dumps(report_dict, indent=2)
    )

    # 1. Executive Certification Banner
    rep_id = report_dict.get("report_id", "REP-UNKNOWN")
    rep_time = report_dict.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    final_risk = report_dict.get("risk_level", "LOW").upper()
    final_score = report_dict.get("vulnerability_score", 0.0)

    st.markdown(
        f"""
        <div class="modern-card card-accent-cyan" style="margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.75rem;">
                <div>
                    <span class="stat-label">Certified Report Identity</span>
                    <div style="font-size: 1.25rem; font-weight: 800; color: var(--text-primary); margin-top: 0.15rem;">
                        {rep_id}
                    </div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.2rem;">
                        Timestamp: <strong>{rep_time}</strong> • Status: <strong>CERTIFIED COMPLETE</strong>
                    </div>
                </div>
                <div style="display: flex; gap: 0.75rem; align-items: center;">
                    <span class="risk-badge risk-{final_risk.lower()}">Overall Risk: {final_risk}</span>
                    <div class="modern-card" style="padding: 0.4rem 0.8rem; background: var(--card-bg-subtle);">
                        <span class="stat-label" style="font-size: 0.7rem;">Vulnerability Score</span>
                        <div style="font-size: 1.1rem; font-weight: 700; color: var(--stat-val-cyan);">{final_score:.1f} / 100</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Detailed On-Screen 15-Section Report Tabs & Sections
    with st.expander("📋 Section 1: Executive Summary & Overview", expanded=True):
        exec_sum = report_dict.get("executive_summary", {})
        st.markdown(
            f"""
            **Overall Security Posture**: Model evaluated under white-box adversarial stress testing.
            - **Evaluated Target**: `{report_dict.get('model_info', {}).get('model_name', 'Model')}`
            - **Risk Level**: `{final_risk}`
            - **Vulnerability Score**: `{final_score:.2f} / 100`
            - **Primary Mitigation Applied**: `{report_dict.get('hardening_results', {}).get('selected_defense', 'None')}`
            """
        )
        if exec_sum.get("key_findings"):
            st.markdown("##### Key Assessment Findings:")
            for kf in exec_sum["key_findings"]:
                st.markdown(f"• {kf}")

    with st.expander("🤖 Section 2: Model Specifications & Parameters (M1)", expanded=False):
        m_info = report_dict.get("model_info", {})
        m_rows = [
            {"Property": "Model Name", "Value": str(m_info.get("model_name"))},
            {"Property": "Architecture", "Value": str(m_info.get("model_architecture"))},
            {"Property": "Framework", "Value": str(m_info.get("framework"))},
            {"Property": "Total Parameters", "Value": f"{m_info.get('total_parameters', 0):,}"},
            {"Property": "Trainable Parameters", "Value": f"{m_info.get('trainable_parameters', 0):,}"},
            {"Property": "Input Shape", "Value": str(m_info.get("input_shape"))},
            {"Property": "Classes", "Value": str(m_info.get("num_classes"))},
        ]
        st.table(pd.DataFrame(m_rows))

    with st.expander("📦 Section 3: Evaluation Dataset Configuration (M2)", expanded=False):
        d_info = report_dict.get("dataset_config", {})
        d_rows = [
            {"Parameter": "Dataset Name", "Value": str(d_info.get("dataset_name"))},
            {"Parameter": "Domain / Modality", "Value": str(d_info.get("domain", "image")).capitalize()},
            {"Parameter": "Total Samples Tested", "Value": str(d_info.get("total_samples"))},
            {"Parameter": "Evaluation Split", "Value": str(d_info.get("data_split", "test"))},
        ]
        st.table(pd.DataFrame(d_rows))

    with st.expander("⚡ Section 4: Baseline Clean Model Performance (M2)", expanded=False):
        b_info = report_dict.get("baseline_performance", {})
        b_acc = b_info.get("accuracy", 1.0) * 100.0 if b_info.get("accuracy", 1.0) <= 1.0 else b_info.get("accuracy", 100.0)
        b_conf = b_info.get("confidence", 0.9) * 100.0 if b_info.get("confidence", 0.9) <= 1.0 else b_info.get("confidence", 90.0)
        st.markdown(
            f"""
            - **Clean Classification Accuracy**: `{b_acc:.2f}%`
            - **Mean Baseline Confidence**: `{b_conf:.2f}%`
            - **Inference Latency**: `{b_info.get('throughput_samples_per_sec', 0.0):.1f} samples/sec`
            """
        )

    with st.expander("⚔️ Section 5: Adversarial Attack Vectors & Results (M3)", expanded=True):
        atk_results = report_dict.get("attack_results", {})
        for a_k, a_v in atk_results.items():
            st.markdown(f"#### Attack Vector: `{a_v.get('attack_name', a_k).upper()}`")
            st.markdown(
                f"""
                - **Category**: `{a_v.get('category', 'Gradient-based')}`
                - **Threat Model**: `{a_v.get('threat_model', 'White-Box')}`
                - **Epsilon Perturbation**: `{a_v.get('epsilon', 0.05)}`
                - **Execution Time**: `{a_v.get('execution_time_seconds', 0.0):.3f}s`
                """
            )
            if a_v.get("perturbation_metrics"):
                st.json(a_v["perturbation_metrics"])

    with st.expander("📊 Section 6 & 7: Vulnerability Assessment & Risk Severity (M5)", expanded=True):
        v_metrics = report_dict.get("vulnerability_metrics", {})
        st.markdown(
            f"""
            - **Overall Vulnerability Score**: `{final_score:.2f} / 100`
            - **Risk Severity Classification**: `{final_risk}`
            """
        )
        for vk, vv in v_metrics.items():
            assess = vv.get("assessment", {})
            asr_v = assess.get("attack_success_rate", 0.0) * 100.0
            acc_d = assess.get("accuracy_drop", 0.0) * 100.0
            st.markdown(
                f"• **Attack [{vk.upper()}]**: ASR = **{asr_v:.2f}%**, Accuracy Drop = **-{acc_d:.2f}%**"
            )

    with st.expander("🛡️ Section 8: MITRE ATLAS Threat Framework Mapping", expanded=False):
        mitre_info = report_dict.get("mitre_atlas_mapping", {})
        if mitre_info:
            for atk_k, m_data in mitre_info.items():
                st.markdown(f"**Threat Vector**: `{atk_k.upper()}`")
                st.markdown(f"- **Tactic**: `{m_data.get('tactic')}`")
                st.markdown(f"- **Technique**: `{m_data.get('technique')}`")
                st.markdown(f"- **Mitigation**: `{m_data.get('mitigation')}`")
        else:
            st.markdown(
                """
                - **Tactic**: `AML.TA0000 — ML Attack Staging`
                - **Technique**: `AML.T0043 — Craft Adversarial Data`
                - **Mitigation**: `AML.M0002 — Model Hardening & AML.M0003 — Adversarial Input Detection`
                """
            )

    with st.expander("🔍 Section 9: Explainable AI (XAI) Root Cause Analysis (M6)", expanded=False):
        xai_info = report_dict.get("xai_findings", {})
        if xai_info:
            st.json(xai_info)
        else:
            st.markdown(
                "Attribution analysis shows that adversarial perturbations alter high-frequency pixel saliency patterns and gradient alignment, driving confident misclassifications across decision boundaries."
            )

    with st.expander("🛠️ Section 10: Model Hardening Engine & Applied Mitigations (M7)", expanded=True):
        hard_data = report_dict.get("hardening_results", {})
        st.markdown(
            f"""
            - **Selected Hardening Defense**: `{str(hard_data.get('selected_defense', 'None')).replace('_', ' ').title()}`
            - **Optimization Status**: `{hard_data.get('status', 'SUCCESS')}`
            - **Acceptance Criteria**: `{'Met (Accepted)' if hard_data.get('improvement_sufficient') else 'Most Effective Candidate Fallback'}`
            """
        )
        attempts = hard_data.get("defense_attempts", [])
        if attempts:
            st.markdown("##### Candidate Evaluation Attempts:")
            st.table(pd.DataFrame(attempts))

    with st.expander("📈 Section 11 & 12: Empirical Re-Test & Robustness Comparison (M8)", expanded=True):
        bva = report_dict.get("before_vs_after", {})
        if bva:
            st.markdown("##### Before vs. After Hardening Performance:")
            comp_rows = []
            for atk_k, c_data in bva.items():
                comp_rows.append(
                    {
                        "Vector": atk_k.upper(),
                        "ASR Before": f"{c_data.get('before_asr', 0.0) * 100:.1f}%",
                        "ASR After": f"{c_data.get('after_asr', 0.0) * 100:.1f}%",
                        "ASR Delta": f"-{c_data.get('delta_asr', 0.0) * 100:.1f}%",
                        "Acc Before": f"{c_data.get('before_accuracy', 0.0) * 100:.1f}%",
                        "Acc Defended": f"{c_data.get('after_accuracy', 0.0) * 100:.1f}%",
                        "Acc Recovery": f"+{c_data.get('delta_accuracy', 0.0) * 100:.1f}%",
                        "Robustness Status": "Improved ✅" if c_data.get("is_improved") else "Stable ⚠️",
                    }
                )
            st.table(pd.DataFrame(comp_rows))
        else:
            st.info("No comparative re-test data available.")

    with st.expander("⏱️ Section 13: Execution Performance & Hardware Telemetry (M9)", expanded=False):
        exec_perf = report_dict.get("execution_summary", {})
        st.markdown(
            f"""
            - **Total Pipeline Execution Time**: `{exec_perf.get('total_time_seconds', 0.0):.2f}s`
            - **Pipeline Status**: `{exec_perf.get('pipeline_status', 'SUCCESS')}`
            - **Execution Device**: `{active_data.get('device', 'CPU')}`
            """
        )

    with st.expander("💡 Section 14: Strategic Security Recommendations & Action Plan", expanded=True):
        recs = report_dict.get("recommendations", [])
        if recs:
            for idx, r in enumerate(recs, 1):
                st.markdown(f"**{idx}.** {r}")
        else:
            st.markdown("• Maintain deployed input smoothing and sanitization pre-inference filters.")
            st.markdown("• Integrate multi-step PGD adversarial perturbations into continuous CI/CD evaluation.")

    with st.expander("📜 Section 15: Final Security Attestation & Certification", expanded=False):
        final_sec = report_dict.get("final_security_summary", {})
        st.markdown(
            f"""
            <div class="modern-card card-accent-emerald" style="padding: 1rem;">
                <div style="font-weight: 700; color: var(--text-primary); margin-bottom: 0.3rem;">
                    AdverScan AI Security Assurance Attestation
                </div>
                <div style="font-size: 0.85rem; color: var(--text-secondary);">
                    This report verifies that the evaluated model has completed empirical adversarial robustness testing under the AdverScan framework. Hardening filters have been evaluated and benchmarked for deployment readiness.
                </div>
                <div style="margin-top: 0.6rem; font-size: 0.8rem; color: var(--text-secondary);">
                    Scan ID: <strong>{rep_id}</strong> • Certified: <strong>{rep_time}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 3. Report Downloads & Workflow Actions
    st.markdown("### 📥 Export & Download Report")
    st.caption("Download the complete assessment report in your preferred format")

    col_dl1, col_dl2, col_dl3 = st.columns(3)

    with col_dl1:
        st.download_button(
            label="⬇️ Download Markdown Report (.md)",
            data=formatted_md,
            file_name=f"adverscan_report_{rep_id}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with col_dl2:
        st.download_button(
            label="⬇️ Download Full JSON Report (.json)",
            data=report_json,
            file_name=f"adverscan_report_{rep_id}.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_dl3:
        st.download_button(
            label="⬇️ Download Plaintext Report (.txt)",
            data=formatted_md,
            file_name=f"adverscan_report_{rep_id}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    col_nav1, col_nav2 = st.columns([1, 1])

    with col_nav1:
        if st.button("⬅️ Back to Defense Dashboard", use_container_width=True):
            set_workflow_stage("defense_results")
            st.rerun()

    with col_nav2:
        if st.button("🔄 Start New Assessment (Reset)", type="primary", use_container_width=True):
            reset_workflow_state()
            set_workflow_stage("dashboard")
            st.rerun()

