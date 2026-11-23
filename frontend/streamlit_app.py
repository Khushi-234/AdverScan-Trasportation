"""
AdverScan — Adversarial Robustness and Security Testing Platform
Continuous Assessment Pipeline:
Dashboard → Model Ingestion → Attack Configuration → Attack Execution
"""

from __future__ import annotations

import os

# Prevent CUDA VRAM memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
import torch

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
from frontend.components.history_view import render_history_view
from frontend.components.comparison_view import render_comparison_view
from frontend.components.xai_visualizer import (
    render_dynamic_xai_dashboard,
    render_pre_post_defense_comparison,
)
from frontend.services.assessment_history_service import get_storage_repository
from frontend.services.attack_service import (
    execute_configured_attack,
    get_class_labels_mapping,
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
from frontend.services.xai_service import (
    explain_assessment,
    get_available_xai_techniques,
    get_suggested_xai_technique,
    TECHNIQUE_DISPLAY_NAMES,
)
from frontend.utils.session_state import (
    get_active_assessment,
    get_active_scan_id,
    get_attack_result,
    get_defense_result,
    get_defense_xai_result,
    get_report_result,
    get_state,
    get_vulnerability_result,
    get_workflow_stage,
    get_xai_result,
    init_session_state,
    reset_workflow_state,
    set_active_assessment,
    set_active_scan_id,
    set_attack_result,
    set_defense_result,
    set_defense_xai_result,
    set_report_result,
    set_state,
    set_vulnerability_result,
    set_workflow_stage,
    set_xai_result,
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


def ensure_active_scan_id() -> str:
    """Ensure an active scan ID exists in SQLite and session state, creating one if missing."""
    sid = get_active_scan_id()
    if sid:
        return sid

    active_data = get_active_assessment() or {}
    meta = active_data.get("metadata")
    telemetry = active_data.get("telemetry", {})
    m_name = (
        active_data.get("model_name")
        or getattr(meta, "model_name", None)
        or getattr(meta, "name", None)
        or get_state("active_model_name")
        or "evaluation-model"
    )
    d_name = (
        active_data.get("dataset_name")
        or getattr(meta, "dataset_name", None)
        or "evaluation-dataset"
    )
    dom = (
        active_data.get("domain")
        or getattr(meta, "domain", None)
        or get_state("active_model_domain")
        or "image"
    )

    try:
        repo = get_storage_repository()
        m_info = {
            "model_id": getattr(meta, "model_id", None) or m_name,
            "model_name": m_name,
            "architecture": getattr(meta, "model_architecture", None) or telemetry.get("architecture"),
            "framework": getattr(meta, "framework", None) or telemetry.get("framework") or "PyTorch",
            "modality": dom,
            "total_parameters": telemetry.get("total_parameters"),
            "trainable_parameters": telemetry.get("trainable_parameters"),
            "num_classes": telemetry.get("num_classes"),
        }
        d_info = {
            "dataset_id": d_name,
            "dataset_name": d_name,
            "modality": dom,
            "number_of_samples": telemetry.get("total_dataset_samples") or (len(active_data.get("test_inputs")) if active_data.get("test_inputs") is not None else None),
            "data_split": "test",
        }
        scan_rec = repo.create_assessment(
            model_info=m_info,
            dataset_info=d_info,
            modality=dom,
        )
        sid = scan_rec["scan_id"]
        set_active_scan_id(sid)
        return sid
    except Exception as err:
        import uuid
        fallback_sid = f"SCAN-{uuid.uuid4().hex[:8].upper()}"
        set_active_scan_id(fallback_sid)
        return fallback_sid


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
                f'<div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 10px 14px; margin-top: 6px;">'
                f'<div style="font-weight: 700; color: var(--stat-val-emerald); font-size: 0.95rem;">⚡ Half Dataset Mode Selected</div>'
                f'<div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 4px; line-height: 1.4;">'
                f'Evaluating <strong>{sample_size:,}</strong> samples (50.0% coverage of {total_avail_samples:,} total samples).'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        elif sample_choice.startswith("Full Dataset"):
            sample_size = full_samples
            warning_msg = ""
            if full_samples > 1000 and default_domain in ("image", "vision"):
                warning_msg = f"<div style='font-size: 0.78rem; color: #f59e0b; margin-top: 4px;'>⚠️ Note: Ingesting {sample_size:,} vision tensors will run gradient backpropagation across all images.</div>"
            st.markdown(
                f'<div style="background: rgba(139, 92, 246, 0.08); border: 1px solid rgba(139, 92, 246, 0.25); border-radius: 8px; padding: 10px 14px; margin-top: 6px;">'
                f'<div style="font-weight: 700; color: #8b5cf6; font-size: 0.95rem;">🚀 Full Dataset Mode Selected</div>'
                f'<div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 4px; line-height: 1.4;">'
                f'Evaluating all <strong>{sample_size:,}</strong> samples (100.0% full benchmark coverage).'
                f'{warning_msg}</div></div>',
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

                # Initialize persistent SQLite record for this assessment run
                try:
                    repo = get_storage_repository()
                    metadata_obj = res_data.get("metadata")
                    telemetry_obj = res_data.get("telemetry", {})
                    meta_dict = res_data.get("metadata_dict", {})

                    res_model_name = (
                        res_data.get("model_name")
                        or getattr(metadata_obj, "model_name", None)
                        or getattr(metadata_obj, "name", None)
                        or meta_dict.get("Model Name")
                        or model_identifier
                    )
                    res_arch = (
                        res_data.get("model_architecture")
                        or telemetry_obj.get("architecture")
                        or getattr(metadata_obj, "model_architecture", None)
                        or getattr(metadata_obj, "architecture", None)
                        or meta_dict.get("Architecture")
                    )
                    res_framework = (
                        telemetry_obj.get("framework")
                        or getattr(metadata_obj, "framework", None)
                        or meta_dict.get("Framework")
                        or default_framework
                    )
                    res_total_params = (
                        res_data.get("total_parameters")
                        or telemetry_obj.get("total_parameters")
                        or getattr(metadata_obj, "total_parameters", None)
                    )
                    res_trainable_params = (
                        res_data.get("trainable_parameters")
                        or telemetry_obj.get("trainable_parameters")
                        or getattr(metadata_obj, "trainable_parameters", None)
                    )
                    res_num_classes = (
                        res_data.get("num_classes")
                        or telemetry_obj.get("num_classes")
                        or getattr(metadata_obj, "num_classes", None)
                    )
                    try:
                        res_num_classes_int = int(res_num_classes) if res_num_classes is not None and str(res_num_classes).isdigit() else None
                    except (ValueError, TypeError):
                        res_num_classes_int = None

                    res_input_shape = (
                        res_data.get("input_shape")
                        or telemetry_obj.get("input_shape")
                        or getattr(metadata_obj, "input_shape", None)
                    )
                    res_dataset_name = (
                        res_data.get("dataset_name")
                        or telemetry_obj.get("dataset_name")
                        or getattr(metadata_obj, "dataset_name", None)
                        or meta_dict.get("Dataset")
                        or dataset_input
                        or "evaluation-dataset"
                    )
                    res_samples = (
                        res_data.get("total_dataset_samples")
                        or telemetry_obj.get("total_dataset_samples")
                        or res_data.get("sample_size")
                    )
                    if res_samples is None and res_data.get("test_inputs") is not None:
                        res_samples = len(res_data.get("test_inputs"))

                    m_info = {
                        "model_id": getattr(metadata_obj, "model_id", None) or model_identifier,
                        "model_name": res_model_name,
                        "architecture": res_arch,
                        "framework": str(res_framework).upper(),
                        "modality": getattr(metadata_obj, "domain", None) or default_domain,
                        "total_parameters": res_total_params,
                        "trainable_parameters": res_trainable_params,
                        "num_classes": res_num_classes_int,
                        "input_shape": str(res_input_shape) if res_input_shape else None,
                        "metadata": meta_dict or telemetry_obj,
                    }
                    d_info = {
                        "dataset_id": dataset_input or res_dataset_name,
                        "dataset_name": res_dataset_name,
                        "modality": getattr(metadata_obj, "domain", None) or default_domain,
                        "number_of_samples": res_samples,
                        "number_of_classes": res_num_classes_int,
                        "data_split": "test",
                        "metadata": {"source": res_data.get("dataset_source")},
                    }
                    scan_rec = repo.create_assessment(
                        model_info=m_info,
                        dataset_info=d_info,
                        modality=getattr(metadata_obj, "domain", None) or default_domain,
                    )
                    set_active_scan_id(scan_rec["scan_id"])
                except Exception:
                    pass

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
    raw_test_inputs = active_data.get("test_inputs")
    if raw_test_inputs is not None:
        if hasattr(raw_test_inputs, "shape") and len(raw_test_inputs.shape) > 0:
            available_samples = int(raw_test_inputs.shape[0])
        elif isinstance(raw_test_inputs, dict):
            first_v = next(iter(raw_test_inputs.values()), None)
            available_samples = len(first_v) if first_v is not None else 1
        elif hasattr(raw_test_inputs, "__len__"):
            available_samples = len(raw_test_inputs)
        else:
            available_samples = 1
    else:
        available_samples = 1

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
        # Prominent Sample Count Selection: All Ingested (Default), Half Ingested, or Custom Number of Samples
        half_avail = max(1, available_samples // 2)
        total_ds_samples = (
            active_data.get("total_dataset_samples")
            or meta_dict.get("Total Dataset Samples")
            or meta_dict.get("total_dataset_samples")
        )
        tot_ds_info = (
            f"{total_ds_samples:,} samples in benchmark dataset"
            if isinstance(total_ds_samples, int)
            else "Available in active dataset split"
        )

        sample_choice = st.selectbox(
            "Evaluation Sample Population (Batch Slicing)",
            options=[
                f"All Ingested ({available_samples:,} samples • 100%)",
                f"Half Ingested ({half_avail:,} samples • 50%)",
                "Custom Number of Samples",
            ],
            index=0,
            help=f"Select how many samples of the ingested batch to evaluate. Defaults to all {available_samples:,} loaded samples.",
        )

        if sample_choice.startswith("All Ingested"):
            eval_sample_count = available_samples
        elif sample_choice.startswith("Half Ingested"):
            eval_sample_count = half_avail
        else:
            eval_sample_count = st.number_input(
                "Specify Custom Sample Count",
                min_value=1,
                max_value=available_samples,
                value=min(available_samples, 10 if available_samples >= 10 else 5),
                step=1,
                help=f"Number of samples to evaluate (1 to {available_samples:,})",
            )
        eval_sample_count = min(max(1, int(eval_sample_count)), available_samples)
        batch_pct = (eval_sample_count / max(1, available_samples)) * 100.0
        st.caption(
            f"⚡ Evaluating **{eval_sample_count:,}** of **{available_samples:,}** loaded sample(s) "
            f"(**{batch_pct:.1f}%** batch coverage • Total: {tot_ds_info})"
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

    # Dynamic Algorithm-Specific Attack Parameter Routing
    atk_id = str(selected_attack.get("id", "fgsm")).lower().strip()
    is_cw = atk_id in ("cw", "carlini_wagner", "tabular_cw", "carlini_wagner_tabular")
    is_deepfool = atk_id == "deepfool"
    is_jsma = atk_id in ("jsma", "tabular_jsma", "time_series_jsma")
    is_soft_dtw = atk_id == "soft_dtw"
    is_smooth_pgd = atk_id == "smooth_pgd"
    is_wavelet = atk_id == "wavelet"
    is_lowprofool = atk_id == "lowprofool"
    is_fgsm = atk_id == "fgsm"
    is_fab = atk_id == "fab"
    is_text_tokens = atk_id in ("hotflip", "textbugger_wb", "gradient_substitution")
    is_universal_trigger = atk_id == "universal_trigger"

    col_p1, col_p2 = st.columns(2)
    params_dict: Dict[str, Any] = {}

    if is_cw:
        # --- CARLINI & WAGNER (C&W L2 Optimization) ---
        # No alpha step size! Optimizes in tanh space using Adam.
        with col_p1:
            c_param = st.slider(
                "Optimization Tradeoff Constant (c)",
                min_value=0.01,
                max_value=20.0,
                value=1.0,
                step=0.1,
                help="Balances L2 perturbation magnitude ||x_adv - x||_2^2 against classification loss margin",
            )
            st.markdown(
                """
                <div class="param-explainer">
                    <strong>Tradeoff Constant (c):</strong> Lagrangian multiplier in <em>||x_adv - x||_2^2 + c · f(x_adv)</em>.
                    <span class="param-impact">• <em>Impact:</em> Higher <em>c</em> emphasizes misclassification over noise budget; lower <em>c</em> finds subtler minimal distortions.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_clip_a, col_clip_b = st.columns(2)
            with col_clip_a:
                clip_min = st.number_input("Clip Min", value=0.0, step=0.1, help="Minimum valid tensor bound")
            with col_clip_b:
                clip_max = st.number_input("Clip Max", value=1.0, step=0.1, help="Maximum valid tensor bound")

            st.markdown(
                """
                <div class="param-explainer">
                    <strong>Tanh Space Box Constraints:</strong> C&W maps coordinates to <em>½(clip_min + clip_max) + ½(clip_max - clip_min)·tanh(w)</em>, guaranteeing physical boundary satisfaction.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_p2:
            confidence = st.slider(
                "Confidence Margin (κ)",
                min_value=0.0,
                max_value=15.0,
                value=0.0,
                step=0.5,
                help="Desired classification confidence margin between top logit and adversarial logit",
            )
            max_iter = st.slider(
                "Optimization Steps (Iterations)",
                min_value=10,
                max_value=200,
                value=50,
                step=5,
                help="Adam gradient steps executed in unconstrained tanh space",
            )
            lr = st.slider(
                "Adam Learning Rate (lr)",
                min_value=0.001,
                max_value=0.2,
                value=0.05 if "tabular" in atk_id else 0.01,
                step=0.005,
                format="%.3f",
                help="Optimizer step size in tanh variable space (Note: C&W does not use an alpha step size)",
            )
            if "tabular" not in atk_id:
                binary_search_steps = st.slider(
                    "Binary Search Steps",
                    min_value=1,
                    max_value=9,
                    value=3,
                    step=1,
                    help="Binary search iterations used to discover the minimal trade-off multiplier c",
                )
                params_dict["binary_search_steps"] = binary_search_steps

            if "tabular" in atk_id or domain == "tabular":
                params_dict["c_penalty"] = float(c_param)
            else:
                params_dict["initial_const"] = float(c_param)
            params_dict["confidence"] = float(confidence)
            params_dict["max_iterations"] = int(max_iter)
            params_dict["learning_rate"] = float(lr)
            epsilon = 1.0  # Nominal reference bound for unconstrained L2 C&W

    elif is_deepfool:
        # --- DEEPFOOL (Decision Boundary Hyperplane Projection) ---
        # No alpha step size! Iteratively projects to nearest boundary hyperplane.
        with col_p1:
            max_iter = st.slider(
                "Maximum Iterations (Steps)",
                min_value=5,
                max_value=100,
                value=50,
                step=5,
                help="Maximum number of linearized boundary projection iterations",
            )
            overshoot = st.slider(
                "Boundary Overshoot Factor (η)",
                min_value=0.001,
                max_value=0.1,
                value=0.02,
                step=0.002,
                format="%.3f",
                help="Fractional displacement beyond the decision hyperplane ensuring cross-boundary misclassification",
            )
            st.markdown(
                """
                <div class="param-explainer">
                    <strong>Decision Boundary Linearization:</strong> DeepFool iteratively computes the distance to the closest affine hyperplane bounding the decision region and projects the input just past it.
                    <span class="param-impact">• <em>Impact:</em> Generates minimal, imperceptible perturbations without requiring fixed step sizes (α).</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_p2:
            num_cls_raw = active_data.get("num_classes") or meta_dict.get("Total Classes") or 10
            try:
                max_k = min(20, max(2, int(num_cls_raw)))
            except (ValueError, TypeError):
                max_k = 10
            top_k = st.slider(
                "Candidate Classes (Top-k)",
                min_value=2,
                max_value=max_k,
                value=min(5, max_k),
                step=1,
                help="Number of closest candidate classes evaluated for boundary proximity at each step",
            )
            col_clip_a, col_clip_b = st.columns(2)
            with col_clip_a:
                clip_min = st.number_input("Clip Min", value=0.0, step=0.1, help="Minimum valid tensor value")
            with col_clip_b:
                clip_max = st.number_input("Clip Max", value=1.0, step=0.1, help="Maximum valid tensor value")

            params_dict["max_iter"] = max_iter
            params_dict["overshoot"] = overshoot
            params_dict["top_k"] = top_k
            epsilon = 0.05

    elif is_jsma:
        # --- JSMA (Jacobian Saliency Map Attack) ---
        # L0 sparsity attack; no alpha, no Linf epsilon slider.
        with col_p1:
            max_features = st.slider(
                "Max Perturbed Elements (L0 Budget)",
                min_value=1,
                max_value=25,
                value=5,
                step=1,
                help="Maximum number of features or timestamps permitted to be altered (L0 sparsity limit)",
            )
            step_size = st.slider(
                "Perturbation Step (θ)",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01,
                format="%.2f",
                help="Magnitude of modification applied to each selected salient element per step",
            )
            col_clip_a, col_clip_b = st.columns(2)
            with col_clip_a:
                clip_min = st.number_input("Clip Min", value=0.0, step=0.1, help="Minimum valid tensor value")
            with col_clip_b:
                clip_max = st.number_input("Clip Max", value=1.0, step=0.1, help="Maximum valid tensor value")

            st.markdown(
                """
                <div class="param-explainer">
                    <strong>Jacobian Saliency Mapping:</strong> Identifies high-leverage input dimensions using forward model gradients and greedily modifies them one-by-one.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_p2:
            params_dict["theta"] = step_size
            if atk_id in ("time_series_jsma", "jsma") and domain in ("time-series", "timeseries"):
                params_dict["max_iter"] = int(max_features)
                params_dict["max_fraction"] = 1.0
                params_dict["increase"] = True
            else:
                params_dict["max_features"] = int(max_features)
                params_dict["max_modifications"] = int(max_features)
            epsilon = float(step_size)

    elif is_soft_dtw:
        # --- SOFT-DTW ATTACK ---
        with col_p1:
            max_warp = st.slider(
                "Max Temporal Warping (ε_warp)",
                min_value=0.02,
                max_value=0.5,
                value=0.2,
                step=0.02,
                format="%.2f",
                help="Maximum allowable temporal dilation/compression fraction",
            )
            gamma = st.slider(
                "Soft-DTW Temperature (γ)",
                min_value=0.05,
                max_value=2.0,
                value=0.5,
                step=0.05,
                format="%.2f",
                help="Smoothing parameter for the differentiable soft-minimum operator",
            )
            lambda_dtw = st.slider(
                "DTW Regularization Penalty (λ_dtw)",
                min_value=0.005,
                max_value=0.5,
                value=0.05,
                step=0.005,
                format="%.3f",
                help="Weight of temporal alignment preservation penalty in objective",
            )
            col_clip_a, col_clip_b = st.columns(2)
            with col_clip_a:
                clip_min = st.number_input("Clip Min", value=0.0, step=0.1, help="Minimum valid tensor value")
            with col_clip_b:
                clip_max = st.number_input("Clip Max", value=1.0, step=0.1, help="Maximum valid tensor value")

        with col_p2:
            num_steps = st.slider(
                "Optimization Iterations", min_value=2, max_value=50, value=10, step=1,
                help="Iterations of differentiable temporal warping optimization"
            )
            lr = st.slider(
                "Optimizer Learning Rate", min_value=0.005, max_value=0.2, value=0.05, step=0.005, format="%.3f",
                help="Learning rate for warping parameter optimization"
            )
            params_dict["max_warp"] = max_warp
            params_dict["gamma"] = gamma
            params_dict["lambda_dtw"] = lambda_dtw
            params_dict["num_steps"] = num_steps
            params_dict["alpha"] = lr
            params_dict["lr"] = lr
            epsilon = float(max_warp)

    elif is_smooth_pgd:
        # --- SMOOTH PGD ---
        with col_p1:
            epsilon = st.slider(
                "Perturbation Budget (ε)", min_value=0.005, max_value=0.5, value=0.05, step=0.005, format="%.3f",
                help="Maximum allowable perturbation magnitude"
            )
            tv_weight = st.slider(
                "Total Variation Regularization (λ_tv)", min_value=0.01, max_value=1.0, value=0.1, step=0.01, format="%.2f",
                help="Penalizes sharp inter-timestep gradient transitions to preserve smooth physical waveform continuity"
            )
            col_clip_a, col_clip_b = st.columns(2)
            with col_clip_a:
                clip_min = st.number_input("Clip Min", value=0.0, step=0.1, help="Minimum valid tensor value")
            with col_clip_b:
                clip_max = st.number_input("Clip Max", value=1.0, step=0.1, help="Maximum valid tensor value")

        with col_p2:
            num_steps = st.slider("Optimization Iterations (Steps)", min_value=1, max_value=50, value=10, step=1)
            alpha = st.slider("Step Size (α)", min_value=0.001, max_value=0.1, value=0.01, step=0.002, format="%.3f")
            params_dict["num_steps"] = num_steps
            params_dict["alpha"] = alpha
            params_dict["tv_weight"] = tv_weight
            params_dict["lambda_tv"] = tv_weight

    elif is_wavelet:
        # --- WAVELET / SPECTRAL ATTACK ---
        with col_p1:
            epsilon = st.slider("Perturbation Budget (ε)", min_value=0.005, max_value=0.5, value=0.05, step=0.005, format="%.3f")
            transform_type = st.selectbox(
                "Spectral Transform Type",
                options=["dwt", "fft"],
                format_func=lambda x: "Discrete Wavelet Transform (Haar DWT)" if x == "dwt" else "Fast Fourier Transform (Real FFT)",
            )
            band = st.selectbox("Target Spectral Sub-band", options=["all", "low", "high", "detail", "approx"])
            col_clip_a, col_clip_b = st.columns(2)
            with col_clip_a:
                clip_min = st.number_input("Clip Min", value=0.0, step=0.1, help="Minimum valid tensor value")
            with col_clip_b:
                clip_max = st.number_input("Clip Max", value=1.0, step=0.1, help="Maximum valid tensor value")

        with col_p2:
            num_steps = st.slider("Optimization Iterations", min_value=1, max_value=50, value=10, step=1)
            alpha = st.slider("Step Size (α)", min_value=0.001, max_value=0.1, value=0.01, step=0.002, format="%.3f")
            params_dict["num_steps"] = num_steps
            params_dict["alpha"] = alpha
            params_dict["transform_type"] = transform_type
            params_dict["band"] = band

    elif is_lowprofool:
        # --- LOWPROFOOL ---
        with col_p1:
            epsilon = st.slider("Perturbation Budget (ε)", min_value=0.01, max_value=0.5, value=0.1, step=0.01, format="%.2f")
            lambda_weight = st.slider(
                "Feature Importance Penalty (λ_weight)", min_value=0.01, max_value=1.0, value=0.1, step=0.01, format="%.2f",
                help="Penalizes modifications to critical or low-mutability tabular columns"
            )
            col_clip_a, col_clip_b = st.columns(2)
            with col_clip_a:
                clip_min = st.number_input("Clip Min", value=0.0, step=0.1, help="Minimum valid tensor value")
            with col_clip_b:
                clip_max = st.number_input("Clip Max", value=1.0, step=0.1, help="Maximum valid tensor value")

        with col_p2:
            num_steps = st.slider("Optimization Iterations", min_value=5, max_value=50, value=20, step=1)
            alpha = st.slider("Step Size (α)", min_value=0.005, max_value=0.1, value=0.05, step=0.005, format="%.3f")
            params_dict["num_steps"] = num_steps
            params_dict["alpha"] = alpha
            params_dict["lambda_weight"] = lambda_weight

    elif is_fgsm:
        # --- FGSM (Single-Step Closed Form) ---
        # No alpha, no num_steps!
        with col_p1:
            epsilon = st.slider(
                "Perturbation Budget (ε)",
                min_value=0.001,
                max_value=0.5,
                value=0.05,
                step=0.005,
                format="%.3f",
                help="Maximum perturbation magnitude (L-infinity bound)",
            )
            st.markdown(
                """
                <div class="param-explainer">
                    <strong>Perturbation Budget (ε):</strong> Defines the maximum allowable distance between clean and adversarial inputs.
                    <span class="param-impact">• <em>Impact:</em> Higher ε increases Attack Success Rate (ASR) by allowing larger distortions, but makes noise more human-visible.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_clip_a, col_clip_b = st.columns(2)
            with col_clip_a:
                clip_min = st.number_input("Clip Min", value=0.0, step=0.1, help="Minimum valid tensor value")
            with col_clip_b:
                clip_max = st.number_input("Clip Max", value=1.0, step=0.1, help="Maximum valid tensor value")

        with col_p2:
            st.markdown(
                """
                <div class="modern-card card-accent-emerald" style="padding: 12px 14px; margin-bottom: 12px;">
                    <div style="font-weight: 700; font-size: 0.95rem; color: var(--stat-val-emerald); margin-bottom: 4px;">
                        ⚡ Single-Step Closed-Form Optimization
                    </div>
                    <div style="font-size: 0.84rem; line-height: 1.45; color: var(--text-secondary);">
                        FGSM perturbs clean inputs along the sign of the model's loss gradient in a single backward pass:
                        <br><code>x_adv = x + ε · sign(∇_x Loss(model(x), y))</code><br>
                        It computes perturbations directly without inner iteration loops or a step size (α).
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    elif is_fab:
        # --- FAB (Fast Adaptive Boundary) ---
        with col_p1:
            epsilon = st.slider("Perturbation Budget (ε)", min_value=0.005, max_value=0.5, value=0.05, step=0.005, format="%.3f")
            top_k = st.slider("Candidate Decision Boundaries (Top-k)", min_value=2, max_value=15, value=5, step=1)
            col_clip_a, col_clip_b = st.columns(2)
            with col_clip_a:
                clip_min = st.number_input("Clip Min", value=0.0, step=0.1, help="Minimum valid tensor value")
            with col_clip_b:
                clip_max = st.number_input("Clip Max", value=1.0, step=0.1, help="Maximum valid tensor value")

        with col_p2:
            num_steps = st.slider("Maximum Iterations", min_value=5, max_value=50, value=15, step=1)
            alpha_max = st.slider("Max Step Bound (α_max)", min_value=0.01, max_value=0.3, value=0.1, step=0.01, format="%.2f")
            params_dict["num_steps"] = num_steps
            params_dict["max_iter"] = num_steps
            params_dict["top_k"] = top_k
            params_dict["alpha_max"] = alpha_max

    elif is_text_tokens:
        # --- TEXT TOKEN ATTACKS (HotFlip, TextBugger, Gradient Substitution) ---
        with col_p1:
            max_mod = st.slider("Max Token Replacements", min_value=1, max_value=10, value=3, step=1, help="Maximum words or subwords to substitute")
            clip_min = 0.0
            clip_max = 1.0

        with col_p2:
            params_dict["max_modifications"] = int(max_mod)
            if atk_id == "hotflip":
                cand_size = st.slider(
                    "Candidate Vocab Size (Top-k)",
                    min_value=5,
                    max_value=100,
                    value=50,
                    step=5,
                    help="Top-k nearest gradient candidate vocabulary tokens to evaluate",
                )
                params_dict["candidate_vocab_size"] = int(cand_size)
            elif atk_id == "textbugger_wb":
                bug_types = st.multiselect(
                    "Bug Transformation Types",
                    options=["insert", "delete", "swap", "sub_c", "sub_w"],
                    default=["insert", "delete", "swap", "sub_c", "sub_w"],
                    help="Allowed character/word bug perturbation operators",
                )
                params_dict["bug_types"] = bug_types if bug_types else ["sub_w"]
            elif atk_id == "gradient_substitution":
                num_cand = st.slider(
                    "Candidate Tokens (Top-k)",
                    min_value=2,
                    max_value=50,
                    value=10,
                    step=2,
                    help="Top candidates filtered by gradient first-order Taylor approximation",
                )
                sim_thresh = st.slider(
                    "Cosine Similarity Threshold",
                    min_value=0.1,
                    max_value=0.9,
                    value=0.4,
                    step=0.05,
                    help="Minimum cosine similarity required with original embedding",
                )
                params_dict["num_candidates"] = int(num_cand)
                params_dict["similarity_threshold"] = float(sim_thresh)
            epsilon = 0.05

    elif is_universal_trigger:
        # --- UNIVERSAL ADVERSARIAL TRIGGER ---
        with col_p1:
            trigger_len = st.slider("Trigger Token Length", min_value=1, max_value=10, value=3, step=1, help="Length of static prefix sequence prepended to text")
            num_steps = st.slider("Trigger Optimization Steps", min_value=5, max_value=50, value=20, step=1)
            clip_min = 0.0
            clip_max = 1.0

        with col_p2:
            params_dict["trigger_length"] = int(trigger_len)
            params_dict["num_iterations"] = int(num_steps)
            epsilon = 0.05

    else:
        # --- STANDARD ITERATIVE PGD & GENERAL GRADIENT-BASED ATTACKS ---
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
                    <span class="param-impact">• <em>Impact:</em> Higher ε increases Attack Success Rate (ASR) by allowing larger distortions, but makes noise more human-visible.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_clip_a, col_clip_b = st.columns(2)
            with col_clip_a:
                clip_min = st.number_input("Clip Min", value=0.0, step=0.1, help="Minimum valid tensor value")
            with col_clip_b:
                clip_max = st.number_input("Clip Max", value=1.0, step=0.1, help="Maximum valid tensor value")

            st.markdown(
                """
                <div class="param-explainer">
                    <strong>Tensor Clip Bounds [min, max]:</strong> Constrains perturbed features within physically valid boundaries (e.g. [0.0, 1.0] for normalized pixels).
                </div>
                """,
                unsafe_allow_html=True,
            )

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

    # Targeted Attack Mode Controls (Rendered when supported by attack architecture)
    with col_p2:
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
                except (ValueError, TypeError):
                    num_classes_val = 2

                class_map = get_class_labels_mapping(
                    model=model_adapter,
                    dataset_name=active_data.get("dataset_name"),
                    num_classes=num_classes_val,
                )

                col_tc1, col_tc2 = st.columns([2, 1])
                with col_tc1:
                    target_options = list(range(num_classes_val))
                    default_idx = 1 if num_classes_val > 1 else 0
                    target_class_choice = st.selectbox(
                        "🎯 Target Malicious Class",
                        options=target_options,
                        index=default_idx,
                        format_func=lambda idx: f"Class {idx} — {class_map.get(idx, f'Class {idx}')}",
                        help="Specific class label that the attack optimization should force the model to predict",
                    )
                with col_tc2:
                    st.text_input(
                        "Resolved Class Label",
                        value=class_map.get(int(target_class_choice), f"Class {target_class_choice}"),
                        disabled=True,
                        help="Semantic class meaning for target index",
                    )

                params_dict["target_label"] = int(target_class_choice)
                tgt_name = class_map.get(int(target_class_choice), f"Class {target_class_choice}")

                st.markdown(
                    f"""
                    <div class="modern-card card-accent-violet" style="padding: 10px 14px; margin-top: 6px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <span style="font-weight: 700; font-size: 0.92rem; color: #8b5cf6;">
                                🎯 Target Objective: Class {target_class_choice} ({tgt_name})
                            </span>
                            <span class="badge-status badge-violet">TARGETED CONSTRAINED</span>
                        </div>
                        <div style="font-size: 0.83rem; line-height: 1.45; color: var(--text-secondary);">
                            • <strong>Adversarial Objective:</strong> The attack optimizes gradients to minimize cross-entropy loss specifically toward <strong>Class {target_class_choice} ({tgt_name})</strong> rather than maximizing distance from the clean label.<br>
                            • <strong>Strict Evaluation Metric:</strong> Targeted ASR counts only samples that land precisely on <strong>Class {target_class_choice}</strong>. Flips to other wrong classes are tracked separately as collateral misclassifications.<br>
                            • <strong>Threat Context:</strong> High-severity attack designed to evade security filters or spoof specific authorized identities.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
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
    requires_eps = not (is_cw or is_jsma or is_deepfool or is_text_tokens or is_universal_trigger)
    if requires_eps and epsilon <= 0.0:
        st.error("Perturbation budget (ε) must be strictly greater than 0.")
        valid_params = False
    if clip_min is not None and clip_max is not None and clip_min >= clip_max:
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
            if hasattr(test_inputs, "items"):
                eval_inputs = {k: (v[:eval_sample_count] if hasattr(v, "__getitem__") else v) for k, v in test_inputs.items()}
            elif hasattr(test_inputs, "__getitem__"):
                eval_inputs = test_inputs[:eval_sample_count]
            else:
                eval_inputs = test_inputs
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
                dataset_name=active_data.get("dataset_name"),
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

            # Persist attack execution & findings to SQLite database
            try:
                active_sid = ensure_active_scan_id()
                repo = get_storage_repository()
                full_attack_params = {
                    "epsilon": epsilon,
                    "clip_min": clip_min,
                    "clip_max": clip_max,
                    **(params_dict or {}),
                }
                if not attack_result.get("total_samples") and eval_inputs is not None:
                    attack_result["total_samples"] = len(eval_inputs)

                atk_rec = repo.record_attack_execution(
                    scan_id=active_sid,
                    attack_result=attack_result,
                    attack_params=full_attack_params,
                    adversarial_examples=attack_result.get("adversarial_examples"),
                )
                repo.record_vulnerabilities(
                    scan_id=active_sid,
                    vuln_result=vuln_result,
                    attack_run_id=atk_rec.get("attack_run_id") if atk_rec else None,
                )
                try:
                    recs = generate_vulnerability_recommendations(
                        domain=domain,
                        attack_name=str(attack_result.get("attack_name", "attack")),
                        vuln_score=attack_result.get("vulnerability_score", 50.0),
                        risk_level=attack_result.get("risk_level", "Medium"),
                        asr=attack_result.get("asr_percentage", 0.0),
                    )
                    if recs:
                        repo.record_recommendations(
                            scan_id=active_sid,
                            recommendations=recs,
                        )
                except Exception as rec_err:
                    print(f"[AdverScan Database] Notice generating recommendations: {rec_err}")

                # Ensure scan is marked completed so leaving the pipeline doesn't leave it in 'running'
                repo.update_assessment_status(
                    scan_id=active_sid,
                    status="completed",
                    overall_summary={
                        "risk_level": (vuln_result or {}).get("risk_level", "Medium"),
                        "vulnerability_score": (vuln_result or {}).get("vulnerability_score", 50.0),
                        "asr": attack_result.get("asr_percentage", 0.0),
                    },
                )
            except Exception as persist_err:
                print(f"[AdverScan Database] Error recording attack: {persist_err}")
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

    asr_raw_in = result.get("asr_percentage")
    if asr_raw_in is None:
        asr_raw_in = result.get("attack_success_rate", result.get("asr", 0.0))
    try:
        asr = float(asr_raw_in) if asr_raw_in is not None else 0.0
    except (ValueError, TypeError):
        asr = 0.0

    clean_acc_raw = result.get("clean_accuracy") if result.get("clean_accuracy") is not None else result.get("clean_acc", 0.0)
    try:
        clean_acc = float(clean_acc_raw) if clean_acc_raw is not None else 0.0
    except (ValueError, TypeError):
        clean_acc = 0.0

    adv_acc_raw = result.get("adversarial_accuracy") if result.get("adversarial_accuracy") is not None else result.get("adv_accuracy", 0.0)
    try:
        adv_acc = float(adv_acc_raw) if adv_acc_raw is not None else 0.0
    except (ValueError, TypeError):
        adv_acc = 0.0

    acc_drop_raw = result.get("accuracy_drop")
    try:
        acc_drop = float(acc_drop_raw) if acc_drop_raw is not None else (clean_acc - adv_acc)
    except (ValueError, TypeError):
        acc_drop = 0.0

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
    _raw_score = vuln_res.get("vulnerability_score") if vuln_res.get("vulnerability_score") is not None else result.get("vulnerability_score", 0.0)
    vuln_score = float(_raw_score) if _raw_score is not None else 0.0
    risk_level = str(vuln_res.get("risk_level", result.get("risk_level", "LOW"))).upper()

    # 1. 4 Primary Top-line Modern Stat Cards
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)

    with col_r1:
        if asr_undef:
            asr_subtext = "0 clean samples correct (undefined)"
            asr_val_str = "N/A"
            asr_bar_pct = 0.0
        elif is_targeted_run:
            tgt_name_disp = result.get("target_label_name") or f"Class {target_label_val}"
            asr_subtext = f"<strong>{succ_cnt}</strong> of <strong>{clean_correct_cnt}</strong> reached Target: {tgt_name_disp}"
            asr_val_str = f"{asr:.1f}%"
            asr_bar_pct = min(100.0, max(0.0, asr))
        else:
            asr_subtext = f"<strong>{succ_cnt}</strong> of <strong>{clean_correct_cnt}</strong> initially correct samples flipped"
            asr_val_str = f"{asr:.1f}%"
            asr_bar_pct = min(100.0, max(0.0, asr))

        st.markdown(
            f"""
            <div class="modern-card card-accent-rose">
                <div class="stat-label">{'Targeted Success Rate (ASR)' if is_targeted_run else 'Attack Success Rate (ASR)'}</div>
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

    # Dedicated Targeted Attack Execution Profile Banner
    if is_targeted_run:
        tgt_name_disp = result.get("target_label_name") or f"Class {target_label_val}"
        tgt_hits = int(result.get("targeted_hit_count", succ_cnt))
        collat_flips = int(result.get("collateral_flip_count", max(0, tot_samples - clean_correct_cnt - succ_cnt)))
        tgt_pct = (tgt_hits / clean_correct_cnt * 100.0) if clean_correct_cnt > 0 else 0.0
        collat_pct = (collat_flips / clean_correct_cnt * 100.0) if clean_correct_cnt > 0 else 0.0
        resistant_cnt = max(0, clean_correct_cnt - tgt_hits - collat_flips)

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="modern-card card-accent-violet" style="padding: 12px 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 6px;">
                    <div>
                        <span style="font-size: 0.8rem; font-weight: 700; color: #8b5cf6; text-transform: uppercase;">
                            🎯 Targeted Attack Vector Execution Profile
                        </span>
                        <div style="font-size: 1.15rem; font-weight: 700; color: var(--text-primary); margin-top: 2px;">
                            Target Class: <span style="color: #8b5cf6;">Class {target_label_val} — {tgt_name_disp}</span>
                        </div>
                    </div>
                    <div style="display: flex; gap: 8px;">
                        <span class="badge-status badge-violet">TARGETED CONSTRAINED</span>
                        <span class="badge-status badge-rose">{tgt_hits} / {clean_correct_cnt} TARGET HITS</span>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin-top: 8px;">
                    <div style="background: rgba(139, 92, 246, 0.08); border-radius: 6px; padding: 8px 12px; border: 1px solid rgba(139, 92, 246, 0.2);">
                        <div style="font-size: 0.76rem; color: var(--text-secondary); text-transform: uppercase; font-weight: 600;">Primary Objective (Target Hits)</div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: #8b5cf6;">{tgt_hits} samples ({tgt_pct:.1f}%)</div>
                        <div style="font-size: 0.74rem; color: var(--text-secondary);">Predicted strictly as {tgt_name_disp}</div>
                    </div>
                    <div style="background: rgba(245, 158, 11, 0.08); border-radius: 6px; padding: 8px 12px; border: 1px solid rgba(245, 158, 11, 0.2);">
                        <div style="font-size: 0.76rem; color: var(--text-secondary); text-transform: uppercase; font-weight: 600;">Collateral Flips (Off-Target)</div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: #f59e0b;">{collat_flips} samples ({collat_pct:.1f}%)</div>
                        <div style="font-size: 0.74rem; color: var(--text-secondary);">Misclassified, but missed target class</div>
                    </div>
                    <div style="background: rgba(16, 185, 129, 0.08); border-radius: 6px; padding: 8px 12px; border: 1px solid rgba(16, 185, 129, 0.2);">
                        <div style="font-size: 0.76rem; color: var(--text-secondary); text-transform: uppercase; font-weight: 600;">Resistant (Maintained Label)</div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: #10b981;">{resistant_cnt} samples</div>
                        <div style="font-size: 0.74rem; color: var(--text-secondary);">Remained robust against target coercion</div>
                    </div>
                </div>
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
    sub_scores = (vuln_res or {}).get("sub_scores") or {}

    def _safe_float(val: Any, default: float = 0.0) -> float:
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    asr_raw_val = sub_scores.get("asr_score") if sub_scores.get("asr_score") is not None else (vuln_res or {}).get("asr_score", asr)
    asr_sc = _safe_float(asr_raw_val, asr if asr is not None else 0.0)

    deg_raw_val = sub_scores.get("degradation_score") if sub_scores.get("degradation_score") is not None else (vuln_res or {}).get("degradation_score", 0.0)
    deg_sc = _safe_float(deg_raw_val, 0.0)

    stl_raw_val = sub_scores.get("stealth_score") if sub_scores.get("stealth_score") is not None else (vuln_res or {}).get("stealth_score", 0.0)
    stl_sc = _safe_float(stl_raw_val, 0.0)

    weights = (vuln_res or {}).get("weights") or {"weight_asr": 0.50, "weight_degradation": 0.35, "weight_stealth": 0.15}
    w_asr = _safe_float(weights.get("weight_asr"), 0.50)
    w_deg = _safe_float(weights.get("weight_degradation"), 0.35)
    w_stl = _safe_float(weights.get("weight_stealth"), 0.15)

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

    metrics = result.get("perturbation_metrics") or {}
    linf_val_prev = metrics.get("linf_mean") if metrics.get("linf_mean") is not None else metrics.get("linf", 0.0)
    l2_val_prev = metrics.get("l2_mean") if metrics.get("l2_mean") is not None else metrics.get("l2", 0.0)
    linf_str = f"{_safe_float(linf_val_prev):.4f}" if metrics else "0.0000"
    l2_str = f"{_safe_float(l2_val_prev):.4f}" if metrics else "0.0000"

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

        # -------------------------------------------------------------
        # Outcome Filter & Pagination Controls
        # -------------------------------------------------------------
        tot_eval_rows = len(df_comp)
        outcome_col = "Assessment Outcome" if "Assessment Outcome" in df_comp.columns else ("Outcome" if "Outcome" in df_comp.columns else None)

        col_ctrl1, col_ctrl2 = st.columns([2, 1])

        with col_ctrl1:
            filter_options = ["All Samples"]
            if is_targeted_run:
                filter_options.extend([
                    "Targeted Hits 🎯",
                    "Collateral Flips ⚠️",
                    "Clean Invariant (Resistant) ✅",
                    "Initial Clean Errors ❌",
                ])
            else:
                filter_options.extend([
                    "Adversarial Flips ⚠️",
                    "Clean Invariant (Resistant) ✅",
                    "Initial Clean Errors ❌",
                ])

            selected_filter = st.selectbox(
                "🔍 Filter Samples by Outcome",
                filter_options,
                index=0,
                key="atk_sample_filter",
            )

        with col_ctrl2:
            page_size_options = [5, 10, 20, 50, "All"]
            default_page_size_idx = 1 if tot_eval_rows > 10 else 0
            selected_page_size = st.selectbox(
                "📄 Rows per Page",
                page_size_options,
                index=default_page_size_idx,
                key="atk_page_size",
            )

        # Apply Outcome Filter
        df_filtered = df_comp.copy()
        if outcome_col:
            if selected_filter.startswith("Targeted Hits"):
                df_filtered = df_filtered[df_filtered[outcome_col].astype(str).str.contains("Targeted Hit", na=False)]
            elif selected_filter.startswith("Collateral Flips"):
                df_filtered = df_filtered[df_filtered[outcome_col].astype(str).str.contains("Collateral|Untargeted", na=False)]
            elif selected_filter.startswith("Adversarial Flips"):
                df_filtered = df_filtered[df_filtered[outcome_col].astype(str).str.contains("Flip|Targeted Hit", na=False)]
            elif selected_filter.startswith("Clean Invariant"):
                df_filtered = df_filtered[df_filtered[outcome_col].astype(str).str.contains("Preserved|Invariant", na=False)]
            elif selected_filter.startswith("Initial Clean Errors"):
                df_filtered = df_filtered[df_filtered[outcome_col].astype(str).str.contains("Initial|Error", na=False)]

        filtered_count = len(df_filtered)

        # Pagination calculations
        if selected_page_size == "All" or filtered_count <= 5:
            page_size = max(1, filtered_count)
            total_pages = 1
            current_page = 1
        else:
            page_size = int(selected_page_size)
            total_pages = max(1, math.ceil(filtered_count / page_size))
            if "atk_page_idx" not in st.session_state:
                st.session_state["atk_page_idx"] = 1
            if st.session_state["atk_page_idx"] > total_pages:
                st.session_state["atk_page_idx"] = total_pages
            current_page = st.session_state["atk_page_idx"]

        # Render Table Page Slice
        if filtered_count > 0:
            start_row = (current_page - 1) * page_size
            end_row = min(start_row + page_size, filtered_count)
            df_page = df_filtered.iloc[start_row:end_row]

            st.table(df_page)

            # Pagination Bar
            if total_pages > 1:
                col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
                with col_p1:
                    if st.button("⬅️ Previous", disabled=(current_page <= 1), key="atk_prev_page", use_container_width=True):
                        st.session_state["atk_page_idx"] = max(1, current_page - 1)
                        st.rerun()
                with col_p2:
                    st.markdown(
                        f"""
                        <div style="text-align: center; padding: 6px 0; font-size: 0.88rem; color: var(--text-secondary);">
                            Page <strong>{current_page}</strong> of <strong>{total_pages}</strong> • Showing samples <strong>{start_row + 1}–{end_row}</strong> of <strong>{filtered_count}</strong>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col_p3:
                    if st.button("Next ➡️", disabled=(current_page >= total_pages), key="atk_next_page", use_container_width=True):
                        st.session_state["atk_page_idx"] = min(total_pages, current_page + 1)
                        st.rerun()
            else:
                st.caption(f"Showing all {filtered_count} matching samples (of {tot_eval_rows} total evaluated).")
        else:
            st.info(f"No samples match the filter '{selected_filter}'.")
    else:
        st.info("No per-sample prediction comparisons available.")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # M6: Explainable AI (XAI) Root-Cause Analysis Expander
    # --------------------------------------------------------------------------
    with st.expander("🔍 Explainable AI (XAI) Root-Cause Analysis (M6)", expanded=False):
        st.markdown(
            """
            <div style="font-size: 0.92rem; color: var(--text-secondary); margin-bottom: 12px;">
                Investigate which input features, tokens, frequency bands, or pixels were most leveraged by the adversarial perturbation to cause model failure.
            </div>
            """,
            unsafe_allow_html=True,
        )

        domain = active_data.get("domain", "image")
        model = (
            active_data.get("adapter")
            or active_data.get("raw_model")
            or active_data.get("model")
        )
        vuln_res = get_vulnerability_result() or result.get("vulnerability_assessment")
        available_techs = get_available_xai_techniques(domain, model)
        suggested_tech = get_suggested_xai_technique(domain, model)

        if not available_techs:
            st.warning("No compatible XAI explainers found for this model architecture.")
        else:
            def _format_tech_expander(t: str) -> str:
                base = TECHNIQUE_DISPLAY_NAMES.get(t, t.upper())
                return f"{base} ★ (Suggested)" if t == suggested_tech else base

            default_exp_idx = available_techs.index(suggested_tech) if suggested_tech in available_techs else 0

            col_x1, col_x2 = st.columns([3, 1])
            with col_x1:
                expander_tech = st.selectbox(
                    "Attribution Method",
                    available_techs,
                    index=default_exp_idx,
                    format_func=_format_tech_expander,
                    key="xai_expander_tech_select",
                )
            with col_x2:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                btn_run_xai_expander = st.button(
                    "🚀 Explain Attack",
                    type="primary",
                    use_container_width=True,
                    key="btn_run_xai_expander",
                )

            if btn_run_xai_expander:
                with st.spinner(f"Computing {expander_tech.upper()} root-cause attribution..."):
                    try:
                        res_xai = explain_assessment(
                            model=model,
                            attack_result=result,
                            vulnerability_result=vuln_res,
                            domain=domain,
                            technique=expander_tech,
                            mode="standard",
                        )
                        set_xai_result(res_xai)
                        set_report_result(None)
                        active_sid = ensure_active_scan_id()
                        if res_xai.get("executed", False):
                            try:
                                repo = get_storage_repository()
                                repo.record_xai_result(
                                    scan_id=active_sid,
                                    xai_result=res_xai,
                                    raw_attributions=res_xai.get("attribution"),
                                )
                            except Exception as xai_err:
                                print(f"[AdverScan Database] Error saving XAI: {xai_err}")
                        if res_xai.get("executed", False):
                            st.success(f"Generated {expander_tech.upper()} explanation successfully!")
                        else:
                            st.error(f"XAI {expander_tech.upper()} Generation Failed: {res_xai.get('message', 'Attribution error')}")
                    except Exception as e:
                        st.error(f"XAI Execution Failed: {e}")

        # If XAI result exists, render preview and link to dedicated page
        current_xai = get_xai_result()
        if current_xai and current_xai.get("executed", False):
            render_dynamic_xai_dashboard(current_xai, title="Root-Cause Attribution Preview")
            if st.button("Open Full Dedicated XAI Analysis Page ↗️", use_container_width=True, key="btn_goto_xai_from_expander"):
                set_workflow_stage("xai_analysis")
                st.rerun()

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Workflow Actions
    col_back, col_xai, col_comp, col_fwd = st.columns([1, 1, 1, 1])

    with col_back:
        if st.button("⬅️ Test Another Attack on this Model", use_container_width=True):
            set_workflow_stage("attack_config")
            st.rerun()

    with col_xai:
        if st.button("🔍 Root-Cause XAI Analysis ➡️", use_container_width=True):
            set_workflow_stage("xai_analysis")
            st.rerun()

    with col_comp:
        if st.button("⚖️ Compare Assessment ➡️", use_container_width=True):
            active_sid = get_active_scan_id()
            if active_sid:
                set_state("comparison_baseline_scan_id", active_sid)
            set_workflow_stage("comparison")
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
# STAGE 4: EXPLAINABLE AI (XAI) ROOT-CAUSE & ATTRIBUTION ANALYSIS
# ==============================================================================
elif stage == "xai_analysis":
    st.markdown("## 🔍 Explainable AI (XAI) Root-Cause Analysis")
    st.caption(
        "Step 4 of 7 • Isolate sensitive input features, inspect adversarial attribution drift, and understand decision shifts."
    )

    active_data = get_active_assessment()
    attack_res = get_attack_result()

    if not active_data or not attack_res:
        st.warning("No active attack results found to explain. Run an adversarial attack first.")
        col_warn1, col_warn2 = st.columns(2)
        with col_warn1:
            if st.button("⬅️ Go to Model Ingestion", use_container_width=True):
                set_workflow_stage("ingestion")
                st.rerun()
        with col_warn2:
            if st.button("⬅️ Go to Attack Engine", type="primary", use_container_width=True):
                set_workflow_stage("attack_config")
                st.rerun()
        st.stop()

    model = (
        active_data.get("adapter")
        or active_data.get("raw_model")
        or active_data.get("model")
    )
    domain = active_data.get("domain", "image")
    vuln_res = get_vulnerability_result() or attack_res.get("vulnerability_assessment")
    cached_xai = get_xai_result()

    available_techs = get_available_xai_techniques(domain, model)
    suggested_tech = get_suggested_xai_technique(domain, model)

    if not available_techs:
        st.warning("No compatible XAI explainers found for this model architecture.")
        st.stop()

    def format_tech_label(t: str) -> str:
        base = TECHNIQUE_DISPLAY_NAMES.get(t, t.upper())
        if t == suggested_tech:
            return f"{base} ★ (Suggested)"
        return base

    default_idx = (
        available_techs.index(cached_xai["technique"])
        if (cached_xai and cached_xai.get("technique") in available_techs)
        else (available_techs.index(suggested_tech) if suggested_tech in available_techs else 0)
    )

    # Clean, non-overwhelming Controls Card
    st.markdown(
        """
        <div class="modern-card" style="margin-bottom: 1.25rem;">
            <div style="font-size: 1.05rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.5rem;">
                ⚙️ Explainability Diagnostic Controls
            </div>
        """,
        unsafe_allow_html=True,
    )

    ctrl_col1, ctrl_col2 = st.columns([3, 1])
    with ctrl_col1:
        selected_tech = st.selectbox(
            "Attribution Technique",
            available_techs,
            index=default_idx,
            format_func=format_tech_label,
            help="Supported explainability algorithms verified for this model architecture and data domain.",
            key="xai_tech_select",
        )
    with ctrl_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        btn_run_xai = st.button(
            "🚀 Generate Explanation",
            type="primary",
            use_container_width=True,
            key="btn_run_xai_page",
        )

    # Optional Collapsible Advanced Settings (Keeps screen clean and uncluttered)
    with st.expander("⚙️ Advanced Parameters (Optional)", expanded=False):
        selected_mode = st.selectbox(
            "Evaluation Mode",
            ["standard", "fast", "thorough"],
            index=0,
            help="Fast uses fewer evaluation steps; thorough maximizes sampling / IG integration steps.",
            key="xai_mode_select",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if btn_run_xai:
        with st.spinner(f"Computing {selected_tech.upper()} attributions for {domain} model..."):
            try:
                xai_res = explain_assessment(
                    model=model,
                    attack_result=attack_res,
                    vulnerability_result=vuln_res,
                    domain=domain,
                    technique=selected_tech,
                    mode=selected_mode,
                )
                set_xai_result(xai_res)
                set_report_result(None)
                try:
                    active_sid = ensure_active_scan_id()
                    repo = get_storage_repository()
                    repo.record_xai_result(
                        scan_id=active_sid,
                        xai_result=xai_res,
                        raw_attributions=xai_res.get("attribution"),
                    )
                except Exception as xai_persist_err:
                    print(f"[AdverScan Database] Error recording XAI: {xai_persist_err}")
                cached_xai = xai_res
                if xai_res.get("executed", False):
                    st.success(f"Successfully generated {selected_tech.upper()} explainability diagnostics!")
                else:
                    st.error(f"{selected_tech.upper()} Explainer Error: {xai_res.get('message', 'Attribution generation failed.')}")
            except Exception as e:
                st.error(f"XAI Generation Failed: {e}")

    # Render Active XAI Results
    if cached_xai and cached_xai.get("executed", False):
        render_dynamic_xai_dashboard(cached_xai)
    elif cached_xai and not cached_xai.get("executed", False):
        st.warning(f"XAI Explainer Status: {cached_xai.get('status', 'unknown')}. Message: {cached_xai.get('message', 'No details available.')}")
    else:
        st.info("💡 Click **'🚀 Generate / Refresh XAI Explanation'** above to compute feature attributions, attribution drift, and decision-boundary shift metrics for this attack.")

    # Pre-Defense vs. Post-Defense Comparison Section (if defense has been executed)
    defense_res = get_defense_result()
    if defense_res:
        st.markdown("---")
        st.markdown("### 🛡️ Defense Verification: Pre vs. Post-Hardening Attribution")
        st.caption("Evaluate whether model hardening neutralized adversarial sensitivity hotspots and stabilized feature attributions.")

        defended_xai = get_defense_xai_result()
        if defended_xai:
            render_pre_post_defense_comparison(cached_xai or {}, defended_xai)
        else:
            st.info("Defense hardening has been executed! Compute post-defense explainability to compare attribution stabilization.")
            if st.button("🔬 Compute Post-Defense Explanations", key="btn_compute_def_xai"):
                defended_model = defense_res.get("defended_model", model)
                with st.spinner("Computing post-defense attribution profiles..."):
                    try:
                        post_xai_res = explain_assessment(
                            model=defended_model,
                            attack_result=attack_res,
                            vulnerability_result=vuln_res,
                            domain=domain,
                            technique=selected_tech,
                            mode=selected_mode,
                        )
                        set_defense_xai_result(post_xai_res)
                        set_report_result(None)
                        if post_xai_res.get("executed", False):
                            st.success("Post-defense XAI attribution profile computed successfully!")
                        else:
                            st.error(f"Post-Defense XAI Failed: {post_xai_res.get('message', 'Attribution error')}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Post-Defense XAI Generation Failed: {e}")

    st.markdown("---")

    # Workflow Actions
    col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
    with col_nav1:
        if st.button("⬅️ Back to Attack Results", use_container_width=True):
            set_workflow_stage("attack_results")
            st.rerun()
    with col_nav2:
        if st.button("🛡️ Proceed to Defense Hardening ➡️", type="primary", use_container_width=True):
            set_workflow_stage("defense_config")
            st.rerun()
    with col_nav3:
        if st.button("📄 Jump to Security Report ➡️", use_container_width=True):
            set_workflow_stage("report")
            st.rerun()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Start New Assessment (Reset)", use_container_width=True, key="reset_from_xai"):
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
    adv_inputs = attack_res.get("_adversarial_examples")
    if adv_inputs is None:
        adv_inputs = attack_res.get("adversarial_examples")
    if adv_inputs is None and active_data.get("active_scan_id"):
        try:
            repo = get_storage_repository()
            scan_detail = repo.get_assessment(active_data.get("active_scan_id"))
            if scan_detail and scan_detail.get("attacks"):
                latest_atk = scan_detail["attacks"][-1]
                adv_path = latest_atk.get("adversarial_samples_path")
                if adv_path:
                    adv_inputs = repo.artifacts.load_artifact(adv_path)
        except Exception:
            pass

    test_inputs = attack_res.get("_clean_inputs")
    if test_inputs is None:
        test_inputs = active_data.get("test_inputs")
    test_labels = attack_res.get("_clean_labels")
    if test_labels is None:
        test_labels = active_data.get("test_labels")

    # Guard: Align tensor batch sizes to adversarial examples evaluated in M3
    if adv_inputs is not None and isinstance(adv_inputs, torch.Tensor):
        target_n = adv_inputs.size(0)
        if test_inputs is not None and hasattr(test_inputs, "size") and test_inputs.size(0) > target_n:
            test_inputs = test_inputs[:target_n]
        if test_labels is not None and hasattr(test_labels, "size") and test_labels.size(0) > target_n:
            test_labels = test_labels[:target_n]
    elif adv_inputs is not None and isinstance(adv_inputs, dict):
        first_t = next((v for v in adv_inputs.values() if isinstance(v, torch.Tensor)), None)
        if first_t is not None:
            target_n = first_t.size(0)
            if test_labels is not None and hasattr(test_labels, "size") and test_labels.size(0) > target_n:
                test_labels = test_labels[:target_n]

    atk_name = attack_res.get("attack_name", "Adversarial Attack")
    asr_pct = attack_res.get("asr_percentage", 0.0)
    acc_drop = attack_res.get("accuracy_drop", 0.0)
    vuln_res = get_vulnerability_result() or attack_res.get("vulnerability_assessment") or {}
    _raw_score = vuln_res.get("vulnerability_score") if vuln_res.get("vulnerability_score") is not None else attack_res.get("vulnerability_score", asr_pct)
    vuln_score = float(_raw_score) if _raw_score is not None else float(asr_pct)
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

    # 1b. Early Notice if model is already robust (ASR <= 0.0%)
    if asr_pct <= 0.0:
        st.markdown(
            """
            <div class="modern-card card-accent-emerald" style="margin-bottom: 1.25rem;">
                <div style="font-weight: 700; font-size: 1.05rem; color: #10b981;">
                    🛡️ Model is Already Robust Against This Attack Vector (ASR: 0.0%)
                </div>
                <div style="font-size: 0.88rem; color: var(--text-secondary); margin-top: 0.4rem; line-height: 1.5;">
                    The evaluated attack caused <strong>0 misclassifications</strong> on clean correct samples (Attack Success Rate is 0.0%).
                    Defensive hardening is <strong>not required</strong>, as applying input transformations risks degrading clean accuracy
                    without providing additional security benefit.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_skip1, col_skip2 = st.columns([1, 1])
        with col_skip1:
            if st.button("⬅️ Test Another Attack on this Model", use_container_width=True, key="btn_skip_to_attack"):
                set_workflow_stage("attack_config")
                st.rerun()
        with col_skip2:
            if st.button("📑 Proceed Directly to Security Report ➡️", type="primary", use_container_width=True, key="btn_skip_to_report"):
                set_workflow_stage("report")
                st.rerun()
        st.markdown("---")

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

            # Persist defense execution to SQLite database
            try:
                active_sid = ensure_active_scan_id()
                repo = get_storage_repository()
                if "accuracy_recovery" not in defense_res:
                    base_adv = defense_res.get("before_adv_accuracy", 0.0)
                    after_adv = defense_res.get("after_adv_accuracy", 0.0)
                    defense_res["accuracy_recovery"] = round(after_adv - base_adv, 2)

                chosen_opt = defense_options.get(selected_defense_id) or defense_options.get(defense_res.get("defense_name")) or {}
                if not defense_res.get("defense_family"):
                    defense_res["defense_family"] = chosen_opt.get("family") or defense_res.get("defense_type") or "Preprocessing"
                if not defense_res.get("defense_stage"):
                    defense_res["defense_stage"] = chosen_opt.get("stage") or "preprocessing"
                if "parameters" not in defense_res:
                    defense_res["parameters"] = custom_params or {}

                detail_scan = repo.get_assessment(active_sid)
                atk_runs = (detail_scan.get("attacks") if detail_scan else []) or []
                latest_atk_run_id = atk_runs[-1]["attack_run_id"] if atk_runs else None

                repo.record_defense_execution(
                    scan_id=active_sid,
                    defense_result=defense_res,
                    attack_run_id=latest_atk_run_id,
                )
                if defense_res.get("recommendations"):
                    repo.record_recommendations(
                        scan_id=active_sid,
                        recommendations=defense_res["recommendations"],
                    )
                # Ensure scan is marked completed so leaving the pipeline doesn't leave it in 'running'
                repo.update_assessment_status(
                    scan_id=active_sid,
                    status="completed",
                    overall_summary={
                        "defense_applied": defense_res.get("defense_name"),
                        "asr_after": defense_res.get("after_asr", 0.0),
                        "clean_accuracy_after": defense_res.get("after_clean_accuracy", 0.0),
                    },
                )
            except Exception as def_persist_err:
                print(f"[AdverScan Database] Error recording defense: {def_persist_err}")
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
    def_type = def_result.get("defense_type", "Preprocessing")

    # 1. Top Status Banner & Active Defense
    if not def_result.get("success", True) or def_result.get("status") in ("failed", "error"):
        status_badge = '<span class="badge-status badge-rose" style="font-size: 0.88rem;">❌ RETEST EXECUTION FAILED</span>'
        status_note = f"Retest execution failed: {def_result.get('error', 'Execution error encountered during retest.')}"
        card_accent = "card-accent-rose"
    elif def_result.get("status") in ("not_needed", "already_robust") or def_result.get("defense_effectiveness") == "Not Needed (Model Already Robust)":
        status_badge = '<span class="badge-status badge-cyan" style="font-size: 0.88rem;">🛡️ NOT NEEDED — Model Already Robust</span>'
        status_note = "Baseline Attack Success Rate (ASR) was 0.0%. The model is already fully robust against this attack vector; no hardening transformation was required."
        card_accent = "card-accent-cyan"
    elif def_result.get("status") == "inconclusive" or not def_result.get("post_asr_defined", True):
        status_badge = '<span class="badge-status badge-amber" style="font-size: 0.88rem;">⚠️ INCONCLUSIVE RETEST</span>'
        status_note = def_result.get("reason", "Retest was inconclusive due to undefined baseline metrics or zero clean accuracy.")
        card_accent = "card-accent-amber"
    elif imp_suff:
        status_badge = '<span class="badge-status badge-emerald" style="font-size: 0.88rem;">✅ ACCEPTED — Met Acceptance Thresholds</span>'
        status_note = "All acceptance criteria for ASR reduction, vulnerability improvement, and clean accuracy preservation were met during re-testing."
        card_accent = "card-accent-emerald"
    elif fallback_used:
        status_badge = '<span class="badge-status badge-amber" style="font-size: 0.88rem;">⚠️ BEYOND STRICT THRESHOLD — Best Candidate Fallback</span>'
        status_note = "Candidates did not meet strict ASR reduction criteria; automatically selected the most effective candidate achieving maximum vulnerability reduction."
        card_accent = "card-accent-amber"
    else:
        status_badge = '<span class="badge-status badge-rose" style="font-size: 0.88rem;">❌ INSUFFICIENT IMPROVEMENT / BEYOND THRESHOLD</span>'
        status_note = "Defense evaluation completed. The defense did not satisfy configured acceptance criteria."
        card_accent = "card-accent-rose"

    st.markdown(
        f"""
        <div class="modern-card {card_accent}" style="margin-bottom: 1.25rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
                <div>
                    <div class="stat-label">Active Hardened Defense & Mechanism</div>
                    <div style="font-size: 1.35rem; font-weight: 800; color: var(--text-primary); margin-top: 0.15rem;">
                        🛡️ {sel_def_disp} <span style="font-size: 0.85rem; font-weight: 500; color: var(--text-secondary);">({def_type})</span>
                    </div>
                </div>
                <div>{status_badge}</div>
            </div>
            <div style="font-size: 0.88rem; color: var(--text-secondary); margin-top: 0.6rem; line-height: 1.5;">
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
                <div style="background: rgba(128, 128, 128, 0.2); border-radius: 9999px; height: 5px; width: 100%; margin: 6px 0; overflow: hidden;">
                    <div style="background: #10b981; width: {max(0.0, min(100.0, 100.0 - a_asr)):.1f}%; height: 100%; border-radius: 9999px;"></div>
                </div>
                <div class="stat-subtext">Net Reduction: <strong style="color: var(--stat-val-emerald);">-{asr_red:.1f}%</strong> (Suppression)</div>
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
                <div style="background: rgba(128, 128, 128, 0.2); border-radius: 9999px; height: 5px; width: 100%; margin: 6px 0; overflow: hidden;">
                    <div style="background: #06b6d4; width: {max(0.0, min(100.0, a_adv)):.1f}%; height: 100%; border-radius: 9999px;"></div>
                </div>
                <div class="stat-subtext">Robustness Recovery: <strong style="color: var(--stat-val-cyan);">{adv_gain:+.1f}%</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m3:
        st.markdown(
            f"""
            <div class="modern-card card-accent-amber">
                <div class="stat-label">Clean Accuracy Retention</div>
                <div class="stat-value" style="color: var(--stat-val-amber);">{b_clean:.1f}% → {a_clean:.1f}%</div>
                <div style="background: rgba(128, 128, 128, 0.2); border-radius: 9999px; height: 5px; width: 100%; margin: 6px 0; overflow: hidden;">
                    <div style="background: #f59e0b; width: {max(0.0, min(100.0, a_clean)):.1f}%; height: 100%; border-radius: 9999px;"></div>
                </div>
                <div class="stat-subtext">Clean Utility Drop: <strong>-{clean_drop:.1f}%</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m4:
        st.markdown(
            f"""
            <div class="modern-card card-accent-violet">
                <div class="stat-label">Vulnerability Score (M5)</div>
                <div class="stat-value" style="color: var(--stat-val-violet);">{b_vuln:.1f} → {a_vuln:.1f}</div>
                <div style="background: rgba(128, 128, 128, 0.2); border-radius: 9999px; height: 5px; width: 100%; margin: 6px 0; overflow: hidden;">
                    <div style="background: #8b5cf6; width: {max(0.0, min(100.0, 100.0 - a_vuln)):.1f}%; height: 100%; border-radius: 9999px;"></div>
                </div>
                <div class="stat-subtext">Empirical Improvement: <strong style="color: var(--stat-val-emerald);">+{vuln_imp:.1f} pts</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Acceptance Thresholds & Robustness Compliance Table
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-banner">
            <h3>⚖️ Acceptance Thresholds & Robustness Compliance</h3>
            <p>Verification against configured acceptance criteria to determine if the defense is within bounds or beyond threshold</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    threshold_compliance = def_result.get("threshold_compliance")
    if not threshold_compliance:
        thresh_dict = def_result.get("thresholds", {})
        max_clean_drop = float(thresh_dict.get("max_clean_accuracy_drop", 0.02)) * 100.0
        min_asr_red = float(thresh_dict.get("min_asr_reduction", 0.10)) * 100.0
        min_vuln_imp = float(thresh_dict.get("min_vuln_score_improvement", 5.0))

        clean_drop_passed = clean_drop <= max_clean_drop
        asr_red_passed = asr_red >= min_asr_red
        vuln_imp_passed = vuln_imp >= min_vuln_imp

        threshold_compliance = [
            {
                "Criterion": "Clean Accuracy Preservation",
                "Baseline (Before)": f"{b_clean:.1f}%",
                "Hardened (Re-Test)": f"{a_clean:.1f}%",
                "Observed Change": f"-{clean_drop:.1f}% drop",
                "Acceptance Bound": f"Drop ≤ {max_clean_drop:.1f}%",
                "Compliance Verdict": "PASSED (Within Threshold)" if clean_drop_passed else "BEYOND THRESHOLD (Exceeded Drop)",
            },
            {
                "Criterion": "Attack Success Rate (ASR) Reduction",
                "Baseline (Before)": f"{b_asr:.1f}%",
                "Hardened (Re-Test)": f"{a_asr:.1f}%",
                "Observed Change": f"-{asr_red:.1f}% drop",
                "Acceptance Bound": f"Reduction ≥ {min_asr_red:.1f}%",
                "Compliance Verdict": "PASSED (Within Threshold)" if asr_red_passed else "BEYOND THRESHOLD (Insufficient Reduction)",
            },
            {
                "Criterion": "Vulnerability Score Improvement",
                "Baseline (Before)": f"{b_vuln:.1f} / 100",
                "Hardened (Re-Test)": f"{a_vuln:.1f} / 100",
                "Observed Change": f"+{vuln_imp:.1f} pts gain",
                "Acceptance Bound": f"Gain ≥ {min_vuln_imp:.1f} pts",
                "Compliance Verdict": "PASSED (Within Threshold)" if vuln_imp_passed else "BEYOND THRESHOLD (Insufficient Gain)",
            },
        ]
    st.table(pd.DataFrame(threshold_compliance))

    # 4. Sample-by-Sample Restored Prediction Table & Outcome Breakdown
    sample_restorations = def_result.get("sample_restorations", [])
    if sample_restorations:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-banner">
                <h3>🔬 Empirical Re-Test & Label Restoration Analysis</h3>
                <p>Sample-level verification showing how adversarial misclassifications were restored by the defended model</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        outcomes = [str(s.get("Outcome", "")) for s in sample_restorations]
        tot_eval = len(sample_restorations)
        restored_cnt = sum(1 for o in outcomes if "Restored" in o)
        robust_cnt = sum(1 for o in outcomes if "Robust" in o or "Defended" in o)
        remains_cnt = sum(1 for o in outcomes if "Flipped" in o or "Remains" in o or "Error" in o)

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.markdown(
                f"""
                <div class="modern-card card-accent-emerald" style="padding: 10px 14px;">
                    <div class="stat-label">Restored to Correct Label</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: var(--stat-val-emerald);">{restored_cnt} / {tot_eval} samples</div>
                    <div class="stat-subtext">Successfully recovered by defense</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_s2:
            st.markdown(
                f"""
                <div class="modern-card card-accent-cyan" style="padding: 10px 14px;">
                    <div class="stat-label">Invariant / Resistant Baseline</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: var(--stat-val-cyan);">{robust_cnt} / {tot_eval} samples</div>
                    <div class="stat-subtext">Remained correct throughout</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_s3:
            st.markdown(
                f"""
                <div class="modern-card card-accent-rose" style="padding: 10px 14px;">
                    <div class="stat-label">Remains Adversarially Flipped</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: var(--stat-val-rose);">{remains_cnt} / {tot_eval} samples</div>
                    <div class="stat-subtext">Residual vulnerability</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        df_restored = pd.DataFrame(sample_restorations)
        tot_rest_rows = len(df_restored)
        outcome_def_col = "Outcome" if "Outcome" in df_restored.columns else None

        col_dctrl1, col_dctrl2 = st.columns([2, 1])
        with col_dctrl1:
            def_filter = st.selectbox(
                "🔍 Filter Hardened Sample Outcomes",
                ["All Samples", "Restored to Correct Label ✅", "Invariant / Resistant 🛡️", "Remains Adversarially Flipped ❌"],
                index=0,
                key="def_sample_filter",
            )
        with col_dctrl2:
            def_page_size = st.selectbox(
                "📄 Rows per Page",
                [5, 10, 20, 50, "All"],
                index=1 if tot_rest_rows > 10 else 0,
                key="def_page_size",
            )

        df_def_filtered = df_restored.copy()
        if outcome_def_col:
            if def_filter.startswith("Restored"):
                df_def_filtered = df_def_filtered[df_def_filtered[outcome_def_col].astype(str).str.contains("Restored", na=False)]
            elif def_filter.startswith("Invariant"):
                df_def_filtered = df_def_filtered[df_def_filtered[outcome_def_col].astype(str).str.contains("Robust|Defended|Invariant", na=False)]
            elif def_filter.startswith("Remains"):
                df_def_filtered = df_def_filtered[df_def_filtered[outcome_def_col].astype(str).str.contains("Flipped|Remains|Error", na=False)]

        def_filt_cnt = len(df_def_filtered)
        if def_page_size == "All" or def_filt_cnt <= 5:
            d_page_size = max(1, def_filt_cnt)
            d_total_pages = 1
            d_current_page = 1
        else:
            d_page_size = int(def_page_size)
            d_total_pages = max(1, math.ceil(def_filt_cnt / d_page_size))
            if "def_page_idx" not in st.session_state:
                st.session_state["def_page_idx"] = 1
            if st.session_state["def_page_idx"] > d_total_pages:
                st.session_state["def_page_idx"] = d_total_pages
            d_current_page = st.session_state["def_page_idx"]

        if def_filt_cnt > 0:
            d_start = (d_current_page - 1) * d_page_size
            d_end = min(d_start + d_page_size, def_filt_cnt)
            st.table(df_def_filtered.iloc[d_start:d_end])

            if d_total_pages > 1:
                col_dp1, col_dp2, col_dp3 = st.columns([1, 2, 1])
                with col_dp1:
                    if st.button("⬅️ Previous", disabled=(d_current_page <= 1), key="def_prev_page", use_container_width=True):
                        st.session_state["def_page_idx"] = max(1, d_current_page - 1)
                        st.rerun()
                with col_dp2:
                    st.markdown(
                        f"""
                        <div style="text-align: center; padding: 6px 0; font-size: 0.88rem; color: var(--text-secondary);">
                            Page <strong>{d_current_page}</strong> of <strong>{d_total_pages}</strong> • Showing samples <strong>{d_start + 1}–{d_end}</strong> of <strong>{def_filt_cnt}</strong>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col_dp3:
                    if st.button("Next ➡️", disabled=(d_current_page >= d_total_pages), key="def_next_page", use_container_width=True):
                        st.session_state["def_page_idx"] = min(d_total_pages, d_current_page + 1)
                        st.rerun()
            else:
                st.caption(f"Showing all {def_filt_cnt} matching samples (of {tot_rest_rows} total evaluated).")
        else:
            st.info(f"No samples match the filter '{def_filter}'.")

    # 5. Candidate Defense Attempts Breakdown (if Iterative/Auto mode)
    candidate_attempts = def_result.get("candidate_attempts", [])
    if candidate_attempts and len(candidate_attempts) > 1:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-banner">
                <h3>🔍 Iterative Candidate Defense Evaluation Log</h3>
                <p>Audit trail of candidate defense algorithms evaluated against acceptance thresholds</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        df_attempts = pd.DataFrame(candidate_attempts)
        st.table(df_attempts)

    # 6. Hardening Recommendations
    recs = def_result.get("recommendations", [])
    if recs:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown("##### 💡 Strategic Hardening Recommendations")
        for r in recs:
            st.markdown(f"• {r}")

    st.markdown("---")

    # Workflow Actions (Backward left, Forward right, Reset below)
    col_back, col_xai, col_fwd = st.columns([1, 1, 1])

    with col_back:
        if st.button("⬅️ Reconfigure Defense", use_container_width=True):
            set_workflow_stage("defense_config")
            st.rerun()

    with col_xai:
        if st.button("🔬 Compare Pre vs. Post-Defense XAI ➡️", use_container_width=True):
            set_workflow_stage("xai_analysis")
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
                xai_result=get_xai_result(),
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

    # Compile PDF report bytes
    pdf_bytes = None
    try:
        from app.report_generator.pdf_generator import compile_pdf_report
        pdf_bytes = compile_pdf_report(report_res)
    except Exception:
        pdf_bytes = None

    # Persist report and update assessment status to completed
    active_sid = ensure_active_scan_id()
    if active_sid:
        try:
            repo = get_storage_repository()
            r_id = report_dict.get("report_id", f"REP-{active_sid}")

            # 1. Save Markdown report artifact
            art_path = repo.artifacts.save_artifact(
                data=formatted_md,
                scan_id=active_sid,
                category="reports",
                filename_prefix=f"report_{r_id}",
                extension=".md",
            )
            repo.record_report(
                scan_id=active_sid,
                report_path=art_path or "storage_data/artifacts/reports",
                report_name=f"Security_Report_{active_sid}.md",
                report_format="markdown",
                summary_metrics={
                    "risk_level": report_dict.get("risk_level"),
                    "vulnerability_score": report_dict.get("vulnerability_score"),
                },
            )

            # 2. Automatically save certified PDF report artifact
            if pdf_bytes:
                pdf_art_path = repo.artifacts.save_artifact(
                    data=pdf_bytes,
                    scan_id=active_sid,
                    category="reports",
                    filename_prefix=f"Security_Report_{active_sid}",
                    extension=".pdf",
                )
                repo.record_report(
                    scan_id=active_sid,
                    report_path=pdf_art_path or f"reports/{active_sid}/Security_Report_{active_sid}.pdf",
                    report_name=f"Security_Report_{active_sid}.pdf",
                    report_format="pdf",
                    summary_metrics={
                        "risk_level": report_dict.get("risk_level"),
                        "vulnerability_score": report_dict.get("vulnerability_score"),
                    },
                )

            repo.update_assessment_status(
                scan_id=active_sid,
                status="completed",
                overall_summary={
                    "risk_level": report_dict.get("risk_level"),
                    "vulnerability_score": report_dict.get("vulnerability_score"),
                },
            )
        except Exception:
            pass

    # 1. Executive Certification Banner
    rep_id = report_dict.get("report_id", "REP-UNKNOWN")
    rep_time = report_dict.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    final_risk = str(report_dict.get("risk_level") or "LOW").upper()
    raw_score = report_dict.get("vulnerability_score")
    final_score = float(raw_score) if raw_score is not None else None
    score_display = f"{final_score:.1f} / 100" if final_score is not None else "N/A"

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
                        <div style="font-size: 1.1rem; font-weight: 700; color: var(--stat-val-cyan);">{score_display}</div>
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
            - **Vulnerability Score**: `{score_display}`
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
            - **Overall Vulnerability Score**: `{score_display}`
            - **Risk Severity Classification**: `{final_risk}`
            """
        )
        for vk, vv in v_metrics.items():
            assess = vv.get("assessment", {})
            asr_raw = assess.get("attack_success_rate")
            asr_str = f"{float(asr_raw) * 100:.2f}%" if asr_raw is not None else "N/A"
            acc_d_raw = assess.get("accuracy_drop")
            acc_d_str = f"-{float(acc_d_raw) * 100:.2f}%" if acc_d_raw is not None else "N/A"
            st.markdown(
                f"• **Attack [{vk.upper()}]**: ASR = **{asr_str}**, Accuracy Drop = **{acc_d_str}**"
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
            tech = xai_info.get("technique") or "attribution"
            st.markdown(f"**Explainability Method**: `{tech.upper()}`")
            if "status" in xai_info:
                st.caption(f"Status: {xai_info.get('status')} • {xai_info.get('message', '')}")

            local_exp = xai_info.get("local_explanation", {})
            global_exp = xai_info.get("global_explanation", {})

            if local_exp:
                st.markdown("##### 📍 Local Feature Attribution (Sample Level)")
                clean_top = local_exp.get("clean_top_features", [])
                adv_top = local_exp.get("adversarial_top_features", [])
                if clean_top or adv_top:
                    col_l1, col_l2 = st.columns(2)
                    with col_l1:
                        st.markdown("**Clean Prediction Top Features:**")
                        for f in clean_top[:5]:
                            st.markdown(f"- `{f.get('feature', 'feat')}`: {f.get('weight', 0.0):.4f}")
                    with col_l2:
                        st.markdown("**Adversarial Prediction Top Features:**")
                        for f in adv_top[:5]:
                            st.markdown(f"- `{f.get('feature', 'feat')}`: {f.get('weight', 0.0):.4f}")

            if global_exp:
                glob_imp = global_exp.get("global_feature_importance", [])
                vuln_freq = global_exp.get("vulnerability_frequency", []) or global_exp.get("global_top_features", [])

                if glob_imp:
                    st.markdown("##### 🌐 Global Feature Importance (Aggregated SHAP / Attribution)")
                    for g in glob_imp[:5]:
                        imp_val = g.get("mean_abs_importance", g.get("importance", 0.0))
                        st.markdown(f"- **{g.get('feature', 'feat')}**: mean |attribution| `{imp_val:.4f}` (sample coverage: {g.get('sample_count', 0)})")

                if vuln_freq:
                    st.markdown("##### 🎯 Adversarial Vulnerability Frequency")
                    for g in vuln_freq[:5]:
                        succ_f = g.get("success_frequency", g.get("frequency", 0))
                        tot_f = g.get("frequency", g.get("total_frequency", 0))
                        st.markdown(f"- **{g.get('feature', 'feat')}**: exploited in `{succ_f}` attacks (total occurrences: `{tot_f}`)")

            st.json(xai_info)
        else:
            st.info(
                "No XAI explanation generated for this run (technique was not enabled or model unsupported). "
                "Explanations are displayed only when genuinely computed by the M6 Explainability Engine."
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
                b_asr = c_data.get('before_asr')
                a_asr = c_data.get('after_asr')
                d_asr = c_data.get('delta_asr')
                b_acc = c_data.get('before_accuracy')
                a_acc = c_data.get('after_accuracy')
                d_acc = c_data.get('delta_accuracy')

                b_asr_str = f"{float(b_asr) * 100:.1f}%" if b_asr is not None else "N/A"
                a_asr_str = f"{float(a_asr) * 100:.1f}%" if a_asr is not None else "N/A"
                d_asr_str = f"-{float(d_asr) * 100:.1f}%" if d_asr is not None else "N/A"
                b_acc_str = f"{float(b_acc) * 100:.1f}%" if b_acc is not None else "N/A"
                a_acc_str = f"{float(a_acc) * 100:.1f}%" if a_acc is not None else "N/A"
                d_acc_str = f"+{float(d_acc) * 100:.1f}%" if (d_acc is not None and float(d_acc) >= 0) else (f"{float(d_acc) * 100:.1f}%" if d_acc is not None else "N/A")

                comp_rows.append(
                    {
                        "Vector": atk_k.upper(),
                        "ASR Before": b_asr_str,
                        "ASR After": a_asr_str,
                        "ASR Delta": d_asr_str,
                        "Acc Before": b_acc_str,
                        "Acc Defended": a_acc_str,
                        "Acc Recovery": d_acc_str,
                        "Robustness Status": "Improved ✅" if c_data.get("is_improved") else "Stable ⚠️",
                    }
                )
            st.table(pd.DataFrame(comp_rows))
        else:
            st.info("No comparative re-test data available.")

    with st.expander("⏱️ Section 13: Execution Performance & Hardware Telemetry (M9)", expanded=False):
        exec_perf = report_dict.get("execution_summary", {}) or {}
        tot_time = exec_perf.get("total_elapsed_seconds") or exec_perf.get("total_time_seconds", 0.0)
        pipe_status = exec_perf.get("overall_status") or exec_perf.get("pipeline_status", "SUCCESS")
        st.markdown(
            f"""
            - **Total Pipeline Execution Time**: `{float(tot_time):.2f}s`
            - **Pipeline Status**: `{pipe_status}`
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

    report_csv = (
        report_res.to_csv()
        if hasattr(report_res, "to_csv")
        else ""
    )

    col_dl0, col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(5)

    with col_dl0:
        if pdf_bytes:
            st.download_button(
                label="⬇️ PDF (.pdf)",
                data=pdf_bytes,
                file_name=f"adverscan_report_{rep_id}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
        else:
            st.button("⬇️ PDF Unavailable", disabled=True, use_container_width=True)

    with col_dl1:
        st.download_button(
            label="⬇️ Markdown (.md)",
            data=formatted_md,
            file_name=f"adverscan_report_{rep_id}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with col_dl2:
        st.download_button(
            label="⬇️ JSON (.json)",
            data=report_json,
            file_name=f"adverscan_report_{rep_id}.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_dl3:
        st.download_button(
            label="⬇️ Plaintext (.txt)",
            data=formatted_md,
            file_name=f"adverscan_report_{rep_id}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with col_dl4:
        st.download_button(
            label="⬇️ Timing CSV (.csv)",
            data=report_csv,
            file_name=f"adverscan_report_{rep_id}_timing.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    col_nav1, col_nav_hist, col_nav2 = st.columns([1, 1, 1])

    with col_nav1:
        if st.button("⬅️ Back to Defense Dashboard", use_container_width=True):
            set_workflow_stage("defense_results")
            st.rerun()

    with col_nav_hist:
        if st.button("📜 View in Assessment History", use_container_width=True):
            set_workflow_stage("history")
            st.rerun()

    with col_nav2:
        if st.button("🔄 Start New Assessment (Reset)", type="primary", use_container_width=True):
            reset_workflow_state()
            set_workflow_stage("dashboard")
            st.rerun()


# ==============================================================================
# STAGE 8: ASSESSMENT HISTORY & AUDIT
# ==============================================================================
elif stage == "history":
    render_history_view()


# ==============================================================================
# STAGE 9: COMPARE ASSESSMENTS
# ==============================================================================
elif stage == "comparison":
    render_comparison_view()


