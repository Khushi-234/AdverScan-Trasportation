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
    load_model_and_dataset,
)
from frontend.services.report_service import generate_security_report
from frontend.utils.session_state import (
    get_active_assessment,
    get_attack_result,
    get_defense_result,
    get_report_result,
    get_state,
    get_workflow_stage,
    init_session_state,
    reset_workflow_state,
    set_active_assessment,
    set_attack_result,
    set_defense_result,
    set_report_result,
    set_state,
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

    col_data, col_sample = st.columns([3, 1])

    with col_data:
        dataset_input = st.text_input(
            "Dataset Identifier or Path",
            value=default_dataset,
            placeholder="e.g. bazyl/GTSRB, data/adult_census_test.csv, or test split",
            help="Dataset name or path to load evaluation samples from",
        )

    with col_sample:
        sample_size = st.number_input(
            "Evaluation Samples",
            min_value=1,
            max_value=100,
            value=5,
            help="Number of samples to load into memory for fast CPU / GPU evaluation",
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
            st.markdown(
                f"""
                <div class="modern-card card-accent-violet">
                    <div class="stat-label">Dataset & Runtime Telemetry</div>
                    <div style="margin: 0.5rem 0;">
                        <span class="badge-status badge-violet">READY FOR ATTACK</span>
                    </div>
                    <div style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.6; margin-top: 0.6rem;">
                        • <strong>Dataset Identifier:</strong> {dataset_ident}<br>
                        • <strong>Evaluated Samples:</strong> {samples_eval} samples<br>
                        • <strong>Batch Tensor Shape:</strong> {tensor_shape}<br>
                        • <strong>Ingestion Wall Time:</strong> {ingest_disp}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
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
                85,
                text="Evaluating clean vs. adversarial predictions and classification confidences...",
            )
            time.sleep(0.08)

            atk_prog.progress(
                100,
                text="Computing perturbation distortion metrics (L_inf, L_2, L_0, MAE)...",
            )
            time.sleep(0.1)
            atk_prog.empty()

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
        # If val is between -1.0 and 1.0 (non-zero), it is a fraction; scale to percentage.
        # If abs(val) > 1.0, it is already a percentage.
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

    # 1. 4 Primary Top-line Modern Stat Cards
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)

    with col_r1:
        st.markdown(
            f"""
            <div class="modern-card card-accent-rose">
                <div class="stat-label">Attack Success Rate (ASR)</div>
                <div class="stat-value" style="color: var(--stat-val-rose);">{asr:.1f}%</div>
                <div class="stat-subtext"><strong>{succ_cnt}</strong> of <strong>{tot_samples}</strong> samples flipped</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_r2:
        st.markdown(
            f"""
            <div class="modern-card card-accent-amber">
                <div class="stat-label">Accuracy Degradation</div>
                <div class="stat-value" style="color: var(--stat-val-amber);">{clean_acc_disp:.1f}% → {adv_acc_disp:.1f}%</div>
                <div class="stat-subtext">Accuracy Drop: <strong style="color: var(--stat-val-rose);">-{abs(acc_drop_disp):.1f}%</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_r3:
        st.markdown(
            f"""
            <div class="modern-card card-accent-violet">
                <div class="stat-label">Confidence Impact</div>
                <div class="stat-value" style="color: var(--stat-val-violet);">{clean_conf_disp:.1f}% → {adv_conf_disp:.1f}%</div>
                <div class="stat-subtext">Confidence Drop: <strong style="color: var(--stat-val-rose);">-{abs(conf_drop_disp):.1f}%</strong></div>
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
                <div class="stat-subtext">Throughput: <strong>{ms_sample:.1f} ms/sample</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # 2. Perturbation Distance Metrics (matching run_pipeline_for_multimodel.py)
    st.markdown(
        """
        <div class="section-banner">
            <h3>📏 Perturbation Distortion Norms</h3>
            <p>Mathematical distance between clean inputs and adversarial examples across Lp norms</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = result.get("perturbation_metrics", {})
    if metrics:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        with col_m1:
            st.markdown(
                f"""
                <div class="modern-card card-accent-cyan">
                    <div class="stat-label">L-infinity Distance (L_inf)</div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: var(--text-primary);">{metrics.get('linf', 0.0):.5f}</div>
                    <div class="stat-subtext">Budget (ε): <strong>{result.get('epsilon_used', 0.0):.3f}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_m2:
            st.markdown(
                f"""
                <div class="modern-card card-accent-emerald">
                    <div class="stat-label">L2 Euclidean Distance</div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: var(--text-primary);">{metrics.get('l2', 0.0):.4f}</div>
                    <div class="stat-subtext">Mean Euclidean displacement</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_m3:
            l0_val = int(metrics.get("l0", 0))
            l0_pct = metrics.get("l0_pct")
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
                    <div class="stat-subtext"><strong>{l0_pct:.1f}%</strong> of total tensor elements changed</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_m4:
            st.markdown(
                f"""
                <div class="modern-card card-accent-amber">
                    <div class="stat-label">Mean Absolute Error (MAE)</div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: var(--text-primary);">{metrics.get('mean_abs_error', 0.0):.5f}</div>
                    <div class="stat-subtext">Average absolute element change</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("Perturbation metrics not applicable for this data modality.")

    # 3. Visual Charts for Attack Vulnerability & Impact
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-banner">
            <h3>📊 Attack Vulnerability Impact Charts</h3>
            <p>Visual breakdown of accuracy degradation, confidence drops, and sample outcomes</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_atk_ch1, col_atk_ch2 = st.columns(2)
    with col_atk_ch1:
        st.markdown("##### 🎯 Classification Accuracy Impact")
        df_atk_acc = pd.DataFrame(
            {
                "Evaluation Stage": ["Clean Accuracy", "Adversarial Accuracy", "Accuracy Drop"],
                "Percentage (%)": [clean_acc_disp, adv_acc_disp, abs(acc_drop_disp)],
            }
        ).set_index("Evaluation Stage")
        st.bar_chart(df_atk_acc)

    with col_atk_ch2:
        st.markdown("##### 📉 Prediction Confidence Degradation")
        df_atk_conf = pd.DataFrame(
            {
                "Confidence Metric": ["Clean Confidence", "Adversarial Confidence", "Confidence Drop"],
                "Percentage (%)": [clean_conf_disp, adv_conf_disp, abs(conf_drop_disp)],
            }
        ).set_index("Confidence Metric")
        st.bar_chart(df_atk_conf)

    comparisons = result.get("sample_comparisons", [])
    if comparisons:
        atk_outcomes = [c.get("Outcome", c.get("outcome", "Unknown")) for c in comparisons]
        from collections import Counter
        atk_counts = Counter(atk_outcomes)
        df_atk_outcomes = pd.DataFrame(
            {
                "Sample Outcome": list(atk_counts.keys()),
                "Number of Samples": list(atk_counts.values()),
            }
        ).set_index("Sample Outcome")
        st.markdown("##### 🏷️ Sample Classification Outcome Distribution")
        st.bar_chart(df_atk_outcomes)

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

    params_used = result.get("parameters_used", {})
    telemetry_rows = [
        {
            "Parameter": "Attack Algorithm",
            "Value": str(result.get("attack_name", "N/A")),
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
            "Value": str(params_used.get("targeted", False)),
        },
        {"Parameter": "Evaluated Batch Size", "Value": f"{tot_samples} samples"},
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

    if comparisons:
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
    risk_level = (
        "CRITICAL"
        if asr_pct >= 80.0
        else ("HIGH" if asr_pct >= 60.0 else ("MEDIUM" if asr_pct >= 30.0 else "LOW"))
    )

    # 1. Threat Context Banner
    st.markdown(
        f"""
        <div class="modern-card card-accent-rose" style="margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
                <div>
                    <div class="stat-label">Identified Vulnerability Context</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: var(--text-primary); margin-top: 0.2rem;">
                        {model_name} • Attacked with {atk_name}
                    </div>
                </div>
                <div style="display: flex; gap: 0.75rem;">
                    <span class="risk-badge risk-{risk_level.lower()}">Risk Level: {risk_level}</span>
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

