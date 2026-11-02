"""
AdverScan — Adversarial Robustness and Security Testing Platform
Continuous Assessment Pipeline:
Dashboard → Model Ingestion → Attack Configuration → Attack Execution
"""

from __future__ import annotations

import sys
import time
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
from frontend.services.ingestion_service import (
    get_available_presets,
    load_model_and_dataset,
)
from frontend.utils.session_state import (
    get_active_assessment,
    get_attack_result,
    get_state,
    get_workflow_stage,
    init_session_state,
    reset_workflow_state,
    set_active_assessment,
    set_attack_result,
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
                <div class="modern-card">
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
                <div class="modern-card">
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
                <div class="modern-card">
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
                <div class="modern-card">
                    <div class="stat-label">Mean Absolute Error (MAE)</div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: var(--text-primary);">{metrics.get('mean_abs_error', 0.0):.5f}</div>
                    <div class="stat-subtext">Average absolute element change</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("Perturbation metrics not applicable for this data modality.")

    # 3. Hyperparameter Configuration & Execution Telemetry Table
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
    st.dataframe(
        pd.DataFrame(telemetry_rows), use_container_width=True, hide_index=True
    )

    # 4. Sample-by-Sample Prediction & Confidence Table
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
        st.dataframe(df_comp, use_container_width=True, hide_index=True)
    else:
        st.info("No per-sample prediction comparisons available.")

    st.markdown("---")

    # Workflow Actions
    col_act1, col_act2 = st.columns([1, 1])

    with col_act1:
        if st.button("⚡ Test Another Attack on this Model", use_container_width=True):
            set_workflow_stage("attack_config")
            st.rerun()

    with col_act2:
        if st.button(
            "🔄 Start New Assessment (Reset)", type="primary", use_container_width=True
        ):
            reset_workflow_state()
            set_workflow_stage("dashboard")
            st.rerun()
