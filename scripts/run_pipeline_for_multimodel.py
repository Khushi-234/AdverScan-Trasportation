"""
AdverScan — End-to-End Multi-Modal & Multi-Domain Security Assessment CLI & Demo.

Supports interactive domain selection (Image, Text, Time-Series, Tabular),
domain-filtered model selection, automated dataset resolution (.csv, .pt, HuggingFace),
and domain-specific attack execution (single, multiple, or all attacks) through AdverScanOrchestrator:
M1 Model Ingestion -> M2 Baseline Evaluation -> M3 Attack Engine
-> M2 Adversarial Evaluation -> M5 Vulnerability Analysis -> M6 XAI -> M7 Hardening
-> M8 Re-Test & Comparison -> M9 Security Report Generator.
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from dotenv import load_dotenv

# Filter common non-critical third-party warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.jit")

# Ensure repository root directory is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.attack_engine import discover_attacks, get_attack, list_attacks
from app.evaluation.dataset_loader.factory import get_dataset_loader, resolve_dataset_file
from app.ingestion.model_registry import ModelPresetRegistry
from app.ingestion.presets import DATASET_TO_RECOMMENDED_MODEL, MODEL_PRESETS
from app.orchestration import AdverScanOrchestrator, PipelineConfig
from app.utils import get_execution_device_info, resolve_device

load_dotenv()

# =====================================================================
# Canonical Attacks Metadata Catalog per Domain
# =====================================================================
DOMAIN_ATTACKS_CATALOG: Dict[str, List[Dict[str, str]]] = {
    "image": [
        {
            "id": "fgsm",
            "name": "FGSM",
            "full_name": "Fast Gradient Sign Method",
            "category": "Gradient-based (One-step)",
            "description": "Fast single-step perturbation along gradient sign",
        },
        {
            "id": "pgd",
            "name": "PGD",
            "full_name": "Projected Gradient Descent",
            "category": "Gradient-based (Iterative)",
            "description": "Standard multi-step iterative Linf/L2 projected gradient attack",
        },
        {
            "id": "deepfool",
            "name": "DeepFool",
            "full_name": "DeepFool Minimal Perturbation",
            "category": "Boundary-based",
            "description": "Finds closest decision boundary via linearized hyperplane projections",
        },
        {
            "id": "cw",
            "name": "Carlini-Wagner (CW)",
            "full_name": "Carlini & Wagner L2 Optimization",
            "category": "Optimization-based",
            "description": "High-confidence targeted/untargeted L2 optimization attack",
        },
        {
            "id": "fab",
            "name": "FAB",
            "full_name": "Fast Adaptive Boundary Attack",
            "category": "Boundary-based",
            "description": "Fast adaptive boundary projection minimizing Lp distance",
        },
    ],
    "text": [
        {
            "id": "embedding_pgd",
            "name": "Embedding PGD",
            "full_name": "Embedding-Space Projected Gradient Descent",
            "category": "Gradient-based",
            "description": "Iterative perturbation in continuous word/subword embedding space",
        },
        {
            "id": "hotflip",
            "name": "HotFlip",
            "full_name": "HotFlip Gradient Substitution",
            "category": "Gradient-based (Token-level)",
            "description": "Single-token substitution using directional gradient approximation",
        },
        {
            "id": "textbugger_wb",
            "name": "TextBugger WB",
            "full_name": "TextBugger White-Box Attack",
            "category": "Gradient-based / Saliency",
            "description": "Character and word-level perturbations guided by token saliency",
        },
        {
            "id": "universal_trigger",
            "name": "Universal Trigger",
            "full_name": "Universal Adversarial Trigger",
            "category": "Optimization / Trigger",
            "description": "Optimizes static token sequences that induce targeted misclassification",
        },
        {
            "id": "gradient_substitution",
            "name": "Gradient Substitution",
            "full_name": "First-Order Gradient Token Substitution",
            "category": "Gradient-based",
            "description": "Token-level replacement guided by first-order Taylor expansion loss gradients",
        },
    ],
    "time-series": [
        {
            "id": "time_series_pgd",
            "name": "Time-Series PGD",
            "full_name": "Temporal Projected Gradient Descent",
            "category": "Gradient-based",
            "description": "Iterative Linf/L2 gradient perturbation constrained to temporal signal bounds",
        },
        {
            "id": "smooth_pgd",
            "name": "Smooth PGD",
            "full_name": "Smooth Projected Gradient Descent",
            "category": "Gradient-based / TV Regularized",
            "description": "PGD with Total Variation (TV) penalization to preserve signal waveform smoothness",
        },
        {
            "id": "wavelet",
            "name": "Wavelet Attack",
            "full_name": "Wavelet Spectral Perturbation Attack",
            "category": "Spectral / Frequency-Domain",
            "description": "Injects adversarial perturbations into discrete wavelet frequency sub-bands",
        },
        {
            "id": "soft_dtw",
            "name": "Soft-DTW Attack",
            "full_name": "Soft Dynamic Time Warping Attack",
            "category": "Temporal Warping",
            "description": "Adversarial time-series distortion minimizing differentiable soft-DTW distance",
        },
        {
            "id": "time_series_jsma",
            "name": "Time-Series JSMA",
            "full_name": "Jacobian Saliency Map for Time-Series",
            "category": "Saliency-based",
            "description": "Greedy saliency-guided perturbation of critical temporal timestamps",
        },
    ],
    "tabular": [
        {
            "id": "cap_pgd",
            "name": "CapPGD",
            "full_name": "Constraint-Aware Projected Gradient Descent",
            "category": "Gradient-based / Constrained",
            "description": "Iterative PGD projecting perturbations onto continuous, integer, and categorical bounds",
        },
        {
            "id": "lowprofool",
            "name": "LowProFool",
            "full_name": "LowProFool Feature-Weighted Saliency",
            "category": "Gradient-based / Weighted",
            "description": "Perturbs tabular features while penalizing distance according to feature mutability weights",
        },
        {
            "id": "tabular_cw",
            "name": "Tabular CW",
            "full_name": "Carlini & Wagner for Tabular Data",
            "category": "Optimization-based",
            "description": "L2 optimization attack respecting tabular feature boundaries and immutability",
        },
        {
            "id": "tabular_jsma",
            "name": "Tabular JSMA",
            "full_name": "Jacobian Saliency Map for Tabular",
            "category": "Saliency-based",
            "description": "Iteratively perturbs the most influential mutable tabular feature columns",
        },
        {
            "id": "vae_attack",
            "name": "VAE Attack",
            "full_name": "On-Manifold VAE Latent Attack",
            "category": "Manifold-based",
            "description": "Generates realistic adversarial tabular records by perturbing in latent VAE space",
        },
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="AdverScan End-to-End Multi-Modal Interactive Security Assessment CLI"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        choices=["image", "text", "time-series", "tabular"],
        help="Target data domain (image, text, time-series, tabular)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model preset ID, Hugging Face Hub model ID, or local checkpoint path (.pt / .pth)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Evaluation dataset identifier or file path (.csv, .pt, or HF repo)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="100",
        help="Number of samples to evaluate (e.g. 100, 500, or 'all'/'full')",
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=None,
        help="Attacks to execute (e.g. 'all', or specific attack IDs)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["baseline_only", "attack_assessment", "full"],
        help="Execution mode (baseline_only, attack_assessment, full)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target execution device ('auto', 'cpu' or 'cuda')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for baseline and attack evaluation (default: 32)",
    )
    parser.add_argument(
        "--enable-xai",
        action="store_true",
        default=False,
        help="Enable M6 Explainable AI (XAI) feature attribution",
    )
    parser.add_argument(
        "--xai-techniques",
        nargs="+",
        default=["shap"],
        help="XAI techniques to apply (shap, lime)",
    )
    parser.add_argument(
        "--enable-hardening",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable M7 Hardening & Defense Engine",
    )
    parser.add_argument(
        "--defense",
        type=str,
        default="auto",
        help="Hardening defense technique (auto, randomized_smoothing, spatial_smoothing, iterative)",
    )
    parser.add_argument(
        "--enable-retest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable M8 Re-Test & Comparison Engine (default: True)",
    )
    parser.add_argument(
        "--enable-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable M9 Security Report Generator (default: True)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/orchestration",
        help="Output directory for generated artifacts and reports",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Force interactive step-by-step terminal prompt menu",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        default=False,
        help="Disable interactive terminal menu and execute CLI arguments directly",
    )
    return parser.parse_args()


# =====================================================================
# Interactive Prompt Menus
# =====================================================================

def prompt_domain_selection() -> str:
    """Prompt user to select the target model domain."""
    print("\n[Step 1/5] Select Data Modality / Domain:")
    print("  1. Image        (Computer Vision: GTSRB, CIFAR-10, Intel Scenes, ImageNet)")
    print("  2. Text         (Natural Language Processing: SST-2, Twitter Sentiment, IMDb, AG News)")
    print("  3. Time-Series  (Temporal Sequences: ECG 1D-CNN, PatchTST, PatchTSMixer)")
    print("  4. Tabular      (Structured Data: Adult Census Income MLP, Credit Card Fraud MLP)")
    print("  5. All Domains  (Browse all 14 pre-configured models across all modalities)")

    choice_map = {
        "1": "image",
        "2": "text",
        "3": "time-series",
        "4": "tabular",
        "5": "all",
        "image": "image",
        "text": "text",
        "time-series": "time-series",
        "timeseries": "time-series",
        "tabular": "tabular",
        "all": "all",
    }
    while True:
        raw_choice = input("Select Domain [1-5] [1]: ").strip().lower() or "1"
        if raw_choice in choice_map:
            return choice_map[raw_choice]
        print("  ❌ Invalid choice. Please enter 1, 2, 3, 4, or 5.")


def prompt_model_selection(domain: str) -> Dict[str, Any]:
    """Prompt user to select a model registered in MODEL_PRESETS or provide a custom one."""
    domain_clean = domain.lower().replace("_", "-")

    # Extract presets directly from MODEL_PRESETS in app/ingestion/presets.py
    domain_presets: List[Dict[str, Any]] = []
    for preset_key, p in MODEL_PRESETS.items():
        p_domain = str(p.get("data_domain", "")).lower().replace("_", "-")
        if domain_clean in ("all", "all domains", "5") or p_domain == domain_clean:
            entry = dict(p)
            entry["preset_key"] = preset_key
            domain_presets.append(entry)

    title_domain = "ALL DOMAINS" if domain_clean in ("all", "all domains", "5") else domain.upper()
    print(f"\n[Step 2/5] Available Model Presets for Domain [{title_domain}] (from app/ingestion/presets.py):")
    print("-" * 75)
    for idx, preset in enumerate(domain_presets, 1):
        m_name = preset.get("name", preset.get("model_id"))
        p_key = preset.get("preset_key", "")
        m_arch = preset.get("model_type", "Standard")
        m_id = preset.get("model_id")
        rec_data = preset.get("dataset_name", "N/A")
        p_dom = preset.get("data_domain", "").upper()

        # Check if local dataset file exists on disk
        resolved_local = None
        if preset.get("data_domain") == "tabular":
            resolved_local = resolve_dataset_file(rec_data, [".csv", ".tsv"], model_identifier=m_id)
        elif preset.get("data_domain") == "time-series":
            resolved_local = resolve_dataset_file(rec_data, [".pt", ".pth", ".npz"], model_identifier=m_id)

        data_desc = f"{rec_data} (found file: '{resolved_local}')" if resolved_local else rec_data

        print(f"  {idx}. {m_name}")
        print(f"     └─ Preset Key:       {p_key}")
        print(f"     └─ Checkpoint / ID:  {m_id}")
        print(f"     └─ Domain / Arch:    [{p_dom}] {m_arch}")
        print(f"     └─ Default Dataset:  {data_desc}")

    opt_hf = len(domain_presets) + 1
    opt_local = len(domain_presets) + 2
    print(f"  {opt_hf}. Custom Hugging Face Hub Model ID")
    print(f"  {opt_local}. Custom Local Model Checkpoint (.pt / .pth / TorchScript)")
    print("-" * 75)

    while True:
        raw_choice = input(f"Select Model by number [1-{opt_local}] or preset key [1]: ").strip() or "1"
        if raw_choice.isdigit():
            val = int(raw_choice)
            if 1 <= val <= len(domain_presets):
                selected = domain_presets[val - 1]
                return {
                    "model_id": selected["model_id"],
                    "model_name": selected.get("name", selected["model_id"]),
                    "processor_name": selected.get("processor_name", selected["model_id"]),
                    "dataset_name": selected.get("dataset_name"),
                    "data_domain": selected.get("data_domain"),
                    "num_classes": selected.get("num_classes", 2),
                    "preset_info": selected,
                    "is_custom": False,
                }
            elif val == opt_hf:
                m_id = input("Enter Hugging Face Model ID: ").strip()
                p_id = input(f"Enter Processor / Tokenizer ID [{m_id}]: ").strip() or m_id
                num_classes_str = input("Number of Classes [2]: ").strip() or "2"
                num_classes = int(num_classes_str) if num_classes_str.isdigit() else 2
                return {
                    "model_id": m_id,
                    "model_name": m_id.split("/")[-1],
                    "processor_name": p_id,
                    "dataset_name": None,
                    "data_domain": domain if domain != "all" else "image",
                    "num_classes": num_classes,
                    "preset_info": {},
                    "is_custom": True,
                }
            elif val == opt_local:
                m_path = input("Enter Local Model Checkpoint Path (.pt / .pth): ").strip()
                num_classes_str = input("Number of Classes [2]: ").strip() or "2"
                num_classes = int(num_classes_str) if num_classes_str.isdigit() else 2
                return {
                    "model_id": m_path,
                    "model_name": Path(m_path).stem,
                    "processor_name": None,
                    "dataset_name": None,
                    "data_domain": domain if domain != "all" else "tabular",
                    "num_classes": num_classes,
                    "preset_info": {},
                    "is_custom": True,
                }
        else:
            # Check match by preset_key or model_id string
            choice_clean = raw_choice.lower()
            for p in domain_presets:
                if choice_clean in (p.get("preset_key", "").lower(), p.get("model_id", "").lower(), p.get("name", "").lower()):
                    return {
                        "model_id": p["model_id"],
                        "model_name": p.get("name", p["model_id"]),
                        "processor_name": p.get("processor_name", p["model_id"]),
                        "dataset_name": p.get("dataset_name"),
                        "data_domain": p.get("data_domain"),
                        "num_classes": p.get("num_classes", 2),
                        "preset_info": p,
                        "is_custom": False,
                    }

        print(f"  ❌ Invalid input. Please enter a number between 1 and {opt_local}, or a valid preset key.")


def prompt_dataset_selection(domain: str, model_info: Dict[str, Any]) -> Tuple[str, int]:
    """Prompt user to confirm or specify the dataset and sample count."""
    preset_data = model_info.get("preset_info", {})
    raw_dataset = preset_data.get("dataset_name") or model_info.get("dataset_name")
    model_id = model_info.get("model_id", "")

    # Look up in DATASET_TO_RECOMMENDED_MODEL or resolve local file on disk
    resolved_file = None
    if domain == "tabular":
        resolved_file = resolve_dataset_file(raw_dataset, [".csv", ".tsv"], model_identifier=model_id) or resolve_dataset_file(None, [".csv", ".tsv"], model_identifier=model_id)
    elif domain == "time-series":
        resolved_file = resolve_dataset_file(raw_dataset, [".pt", ".pth", ".npz", ".npy"], model_identifier=model_id) or resolve_dataset_file(None, [".pt", ".pth", ".npz"], model_identifier=model_id)
    elif domain == "text":
        resolved_file = raw_dataset or "sst2"
    else:
        resolved_file = raw_dataset or "bazyl/GTSRB"

    recommended_dataset = resolved_file or raw_dataset or "data/test.csv"

    print(f"\n[Step 3/5] Dataset Selection for [{domain.upper()}]:")
    print(f"  • Recommended Dataset from presets.py: '{raw_dataset or 'N/A'}'")
    if resolved_file and Path(resolved_file).is_file():
        print(f"  • Resolved Local File on Disk:         '{resolved_file}'")
    user_ds = input(f"Enter Dataset identifier or file path [{recommended_dataset}]: ").strip() or recommended_dataset

    while True:
        samples_str = input("Number of samples to evaluate [100] (enter 'all' or '0' for full dataset): ").strip().lower() or "100"
        if samples_str in ("0", "all", "full"):
            sample_count = 0
            break
        try:
            val = int(samples_str)
            if val > 0:
                sample_count = val
                break
            print("  ❌ Please enter a positive integer, or 'all'.")
        except ValueError:
            print("  ❌ Invalid number format. Try again.")

    return user_ds, sample_count


def prompt_attack_selection(domain: str) -> List[str]:
    """
    Dynamically discover all attacks, display the ones compatible with the chosen domain,
    and prompt the user to select specific attacks or ALL attacks.
    """
    discover_attacks()
    catalog_list = DOMAIN_ATTACKS_CATALOG.get(domain, [])

    print(f"\n[Step 4/5] Available Adversarial Attacks for Domain [{domain.upper()}]:")
    print("----------------------------------------------------------------------")
    for idx, atk in enumerate(catalog_list, 1):
        print(f"  {idx}. [{atk['name']}] — {atk['full_name']}")
        print(f"     └─ Category: {atk['category']}")
        print(f"     └─ Description: {atk['description']}")
    print(f"  A. ALL Attacks (Execute all {len(catalog_list)} attacks for this domain)")
    print("----------------------------------------------------------------------")

    while True:
        raw_choice = input(f"Select attack(s) (e.g. '1', '1 2', or 'all') [all]: ").strip().lower() or "all"
        if raw_choice in ("a", "all", "all attacks", "m"):
            return [atk["id"] for atk in catalog_list]

        selected: List[str] = []
        tokens = raw_choice.split()
        for tok in tokens:
            if tok.isdigit():
                num = int(tok)
                if 1 <= num <= len(catalog_list):
                    selected.append(catalog_list[num - 1]["id"])
            else:
                for atk in catalog_list:
                    if tok in (atk["id"].lower(), atk["name"].lower()):
                        selected.append(atk["id"])
                        break

        if selected:
            # Deduplicate while preserving selection order
            return list(dict.fromkeys(selected))
        print("  ❌ No recognized attacks selected. Please enter valid attack numbers or 'all'.")


def prompt_pipeline_options() -> Dict[str, Any]:
    """Prompt for pipeline execution mode, XAI, hardening, re-testing, and device."""
    print("\n[Step 5/5] Pipeline & Security Assessment Options:")
    print("  1. Baseline Only               (M1 Ingestion → M2 Baseline Clean Evaluation)")
    print("  2. Attack Assessment           (M1 → M2 → M3 Attack Engine → M2_Adv → M5 Vulnerability Scoring)")
    print("  3. Full Security Assessment    (Full Pipeline: M1 → M2 → M3 → M5 → M6 XAI → M7 Hardening → M8 ReTest → M9 Report)")

    mode_choice = input("Select Execution Mode [3]: ").strip() or "3"
    mode_map = {"1": "baseline_only", "2": "attack_assessment", "3": "full"}
    selected_mode = mode_map.get(mode_choice, "full")

    enable_xai = False
    xai_techniques = ["shap"]
    enable_hardening = False
    defense = "auto"
    enable_retest = True
    enable_report = True

    if selected_mode == "full":
        xai_in = input("Enable M6 Explainable AI (XAI) feature attributions? [y/N]: ").strip().lower()
        enable_xai = xai_in in ("y", "yes", "true", "1")
        if enable_xai:
            xai_tech_choice = input("Select XAI Technique: 1. SHAP, 2. LIME, 3. Both [1]: ").strip() or "1"
            if xai_tech_choice == "2":
                xai_techniques = ["lime"]
            elif xai_tech_choice == "3":
                xai_techniques = ["shap", "lime"]
            else:
                xai_techniques = ["shap"]

        hard_in = input("Enable M7 Defense & Hardening Engine? [y/N]: ").strip().lower()
        enable_hardening = hard_in in ("y", "yes", "true", "1")
        if enable_hardening:
            print("  Available Defenses: 1. Auto, 2. Randomized Smoothing, 3. Spatial Smoothing, 4. Iterative Loop")
            d_choice = input("Select Defense [1]: ").strip() or "1"
            d_map = {"1": "auto", "2": "randomized_smoothing", "3": "spatial_smoothing", "4": "iterative"}
            defense = d_map.get(d_choice, "auto")

        retest_in = input("Enable M8 Re-Test & Comparison Engine? [Y/n]: ").strip().lower() or "y"
        enable_retest = retest_in in ("y", "yes", "true", "1")

        report_in = input("Enable M9 Security Report Generation? [Y/n]: ").strip().lower() or "y"
        enable_report = report_in in ("y", "yes", "true", "1")

    # Execution Device
    dev_info = get_execution_device_info("auto")
    print(f"\nTarget Execution Device:")
    if dev_info["gpu_available"]:
        print(f"  1. Auto-detect [CUDA / GPU ({dev_info['gpu_model']})] (Default)")
        print("  2. CPU")
    else:
        print("  1. Auto-detect [CPU (CUDA unavailable)] (Default)")
        print("  2. CPU")
    dev_choice = input("Select Device [1]: ").strip() or "1"
    target_device = "cpu" if dev_choice == "2" else resolve_device("auto")

    return {
        "mode": selected_mode,
        "enable_xai": enable_xai,
        "xai_techniques": xai_techniques,
        "enable_hardening": enable_hardening,
        "defense": defense,
        "enable_retest": enable_retest,
        "enable_report": enable_report,
        "device": target_device,
    }


def print_configuration_summary(opts: Dict[str, Any]):
    """Print visually organized configuration summary before running."""
    print("\n" + "=" * 70)
    print("            ADVERSCAN MULTI-MODAL PIPELINE CONFIGURATION            ")
    print("=" * 70)
    print(f"Data Modality / Domain:     {opts['domain'].upper()}")
    print(f"Model Identifier:           {opts['model_id']}")
    print(f"Model Name:                 {opts['model_name']}")
    print(f"Dataset Identifier:         {opts['dataset_name']}")
    print(f"Evaluation Samples:         {'Full Dataset' if opts['samples'] == 0 else opts['samples']}")
    print(f"Selected Attacks:           {', '.join(opts['attacks']).upper()}")
    print(f"Execution Mode:             {opts['mode'].upper()}")
    print(f"Target Device:              {opts['device'].upper()}")
    print(f"Batch Size:                 {opts['batch_size']}")
    print(f"XAI Enabled (M6):           {opts['enable_xai']} {opts['xai_techniques'] if opts['enable_xai'] else ''}")
    print(f"Hardening Enabled (M7):     {opts['enable_hardening']} (Defense: {opts['defense']})")
    print(f"Re-Test Enabled (M8):       {opts['enable_retest']}")
    print(f"Report Generator (M9):      {opts['enable_report']}")
    print(f"Output Directory:           {opts['output_dir']}")
    print("=" * 70 + "\n")


# =====================================================================
# Dataset Loader & Sample Input Builders
# =====================================================================

def build_dataset_and_sample_input(
    domain: str,
    model_id: str,
    dataset_name: str,
    processor_name: Optional[str],
    samples: int,
    batch_size: int,
    preset_info: Dict[str, Any],
) -> Tuple[Any, Any]:
    """
    Construct domain-specific DatasetLoader and synthetic sample_input for validation.
    Handles Tabular (.csv), Time-Series (.pt / .npz), Image, and Text datasets seamlessly.
    """
    split_str = f"test[:{samples}]" if samples > 0 else "test"
    domain_clean = domain.lower().replace("_", "-")

    if domain_clean == "tabular":
        # Resolve CSV dataset file
        resolved_csv = resolve_dataset_file(dataset_name, [".csv", ".tsv"], model_identifier=model_id)
        if resolved_csv:
            print(f"  ✓ Resolved tabular dataset file: '{resolved_csv}'")
            dataset_loader = get_dataset_loader(
                dataset_name=resolved_csv,
                data_domain="tabular",
                batch_size=batch_size,
                data=resolved_csv,
                target_col="target",
            )
        else:
            print(f"  [INFO] No local CSV dataset file found; using synthetic tabular samples.")
            num_feats = preset_info.get("num_features", 14)
            num_classes = preset_info.get("num_classes", 2)
            n_eval = samples if samples > 0 else 100
            x_syn = torch.randn(n_eval, num_feats)
            y_syn = torch.randint(0, num_classes, (n_eval,))
            dataset_loader = get_dataset_loader(
                dataset_name=dataset_name,
                data_domain="tabular",
                batch_size=batch_size,
                data=(x_syn, y_syn),
            )
        num_feats = preset_info.get("num_features", 14)
        sample_input = torch.randn(1, num_feats)
        return dataset_loader, sample_input

    elif domain_clean == "time-series":
        # Resolve Time-Series tensor file
        resolved_ts = resolve_dataset_file(dataset_name, [".pt", ".pth", ".npz", ".npy"], model_identifier=model_id)
        seq_len = preset_info.get("sequence_length", 50)
        num_ch = preset_info.get("num_channels", 1)
        num_classes = preset_info.get("num_classes", 2)

        if resolved_ts and Path(resolved_ts).is_file():
            print(f"  ✓ Resolved time-series dataset file: '{resolved_ts}'")
            loaded = torch.load(resolved_ts, map_location="cpu", weights_only=False)
            if isinstance(loaded, dict):
                x_ts = loaded.get("inputs", loaded.get("X"))
                y_ts = loaded.get("targets", loaded.get("y"))
            elif isinstance(loaded, tuple) and len(loaded) == 2:
                x_ts, y_ts = loaded
            else:
                x_ts, y_ts = loaded, None

            if samples > 0 and len(x_ts) > samples:
                x_ts = x_ts[:samples]
                if y_ts is not None:
                    y_ts = y_ts[:samples]

            dataset_loader = get_dataset_loader(
                dataset_name=resolved_ts,
                data_domain="time-series",
                batch_size=batch_size,
                inputs=x_ts,
                targets=y_ts,
            )
        else:
            print(f"  [INFO] No local time-series tensor file found; using synthetic sequence samples.")
            n_eval = samples if samples > 0 else 100
            x_ts = torch.randn(n_eval, num_ch, seq_len)
            y_ts = torch.randint(0, num_classes, (n_eval,))
            dataset_loader = get_dataset_loader(
                dataset_name=dataset_name,
                data_domain="time-series",
                batch_size=batch_size,
                inputs=x_ts,
                targets=y_ts,
            )

        # Build sample input for time-series
        if "patchtst" in model_id.lower() or "patchtsmixer" in model_id.lower():
            sample_input = torch.randn(1, seq_len, num_ch)
        else:
            sample_input = torch.randn(1, num_ch, seq_len)
        return dataset_loader, sample_input

    elif domain_clean == "text":
        dataset_loader = get_dataset_loader(
            dataset_name=dataset_name,
            data_domain="text",
            processor_name=processor_name or model_id,
            split=split_str,
            batch_size=batch_size,
        )
        sample_input = "AdverScan end-to-end security assessment evaluation sentence."
        return dataset_loader, sample_input

    else:
        # Default: Computer Vision / Image
        res = preset_info.get("input_resolution", (3, 224, 224))
        dataset_loader = get_dataset_loader(
            dataset_name=dataset_name,
            data_domain="image",
            processor_name=processor_name or model_id,
            split=split_str,
            batch_size=batch_size,
        )
        sample_input = torch.randn(1, *res)
        return dataset_loader, sample_input


# =====================================================================
# Main Orchestration Routine
# =====================================================================

def main():
    total_start_time = time.time()
    args = parse_args()

    # Determine interactive vs CLI mode
    is_interactive = args.interactive or (
        not args.no_interactive and args.domain is None and args.model is None
    )

    print("=" * 70)
    print("      ADVERSCAN — MULTI-MODAL ADVERSARIAL SECURITY ASSESSMENT       ")
    print("=" * 70)

    if is_interactive:
        selected_domain = prompt_domain_selection()
        model_info = prompt_model_selection(selected_domain)
        if selected_domain in ("all", "all domains", "5") or not selected_domain:
            selected_domain = model_info.get("data_domain") or "image"
        selected_dataset, sample_count = prompt_dataset_selection(selected_domain, model_info)
        selected_attacks = prompt_attack_selection(selected_domain)
        pipeline_opts = prompt_pipeline_options()

        opts = {
            "domain": selected_domain,
            "model_id": model_info["model_id"],
            "model_name": model_info["model_name"],
            "processor_name": model_info.get("processor_name"),
            "num_classes": model_info.get("num_classes", 2),
            "preset_info": model_info.get("preset_info", {}),
            "dataset_name": selected_dataset,
            "samples": sample_count,
            "attacks": selected_attacks,
            "mode": pipeline_opts["mode"],
            "device": pipeline_opts["device"],
            "batch_size": args.batch_size,
            "enable_xai": pipeline_opts["enable_xai"],
            "xai_techniques": pipeline_opts["xai_techniques"],
            "enable_hardening": pipeline_opts["enable_hardening"],
            "defense": pipeline_opts["defense"],
            "enable_retest": pipeline_opts["enable_retest"],
            "enable_report": pipeline_opts["enable_report"],
            "output_dir": args.output_dir,
        }
    else:
        # Resolve non-interactive CLI arguments matching MODEL_PRESETS
        model_input = args.model or "bazyl/gtsrb-model"
        preset = None
        if model_input in MODEL_PRESETS:
            preset = {"preset_key": model_input, **MODEL_PRESETS[model_input]}
        else:
            for k, v in MODEL_PRESETS.items():
                if model_input in (v.get("model_id"), v.get("name"), k):
                    preset = {"preset_key": k, **v}
                    break

        domain = args.domain or (preset.get("data_domain") if preset else "image")
        domain = domain.lower().replace("_", "-")
        model_id = preset.get("model_id", model_input) if preset else model_input
        model_name = preset.get("name", Path(model_id).stem) if preset else Path(model_id).stem

        # Default dataset from preset or resolved local file
        dataset = args.dataset or (preset.get("dataset_name") if preset else None) or "bazyl/GTSRB"
        if domain == "tabular" and not args.dataset:
            resolved_csv = resolve_dataset_file(dataset, [".csv", ".tsv"], model_identifier=model_id)
            if resolved_csv:
                dataset = resolved_csv
        elif domain == "time-series" and not args.dataset:
            resolved_ts = resolve_dataset_file(dataset, [".pt", ".pth", ".npz", ".npy"], model_identifier=model_id)
            if resolved_ts:
                dataset = resolved_ts

        samples_str = str(args.samples).lower()
        samples_count = 0 if samples_str in ("0", "all", "full") else int(samples_str)

        # Resolve attacks for domain if none given
        if args.attacks:
            if "all" in args.attacks:
                attacks_list = [a["id"] for a in DOMAIN_ATTACKS_CATALOG.get(domain, [])]
            else:
                attacks_list = [a.lower() for a in args.attacks]
        else:
            # Default to first canonical attack of domain
            default_catalog = DOMAIN_ATTACKS_CATALOG.get(domain, [])
            attacks_list = [default_catalog[0]["id"]] if default_catalog else ["fgsm"]

        opts = {
            "domain": domain,
            "model_id": model_id,
            "model_name": model_name,
            "processor_name": preset.get("processor_name", model_id) if preset else model_id,
            "num_classes": preset.get("num_classes", 2) if preset else 2,
            "preset_info": preset or {},
            "dataset_name": dataset,
            "samples": samples_count,
            "attacks": attacks_list,
            "mode": args.mode.lower(),
            "device": resolve_device(args.device),
            "batch_size": args.batch_size,
            "enable_xai": args.enable_xai,
            "xai_techniques": [t.lower() for t in args.xai_techniques],
            "enable_hardening": args.enable_hardening,
            "defense": args.defense.lower(),
            "enable_retest": args.enable_retest,
            "enable_report": args.enable_report,
            "output_dir": args.output_dir,
        }

    print_configuration_summary(opts)

    if is_interactive:
        confirm = input("▶ Proceed and launch security assessment? [Y/n]: ").strip().lower()
        if confirm in ("n", "no"):
            print("  Assessment aborted by user.")
            sys.exit(0)

    # 1. Prepare Dataset Loader and Sample Input
    print(f"\n[1/3] Preparing dataset loader for '{opts['dataset_name']}' ({opts['domain'].upper()})...")
    try:
        dataset_loader, sample_input = build_dataset_and_sample_input(
            domain=opts["domain"],
            model_id=opts["model_id"],
            dataset_name=opts["dataset_name"],
            processor_name=opts.get("processor_name"),
            samples=opts["samples"],
            batch_size=opts["batch_size"],
            preset_info=opts.get("preset_info", {}),
        )
    except Exception as e:
        print(f"  ❌ Error initializing dataset loader: {e}")
        sys.exit(1)

    # 2. Build PipelineConfig
    print(f"[2/3] Configuring AdverScan Pipeline (Mode: {opts['mode'].upper()})...")
    pipeline_config = PipelineConfig(
        model_path=opts["model_id"],
        sample_input=sample_input,
        device=opts["device"],
        model_name=opts["model_name"],
        num_classes=opts["num_classes"],
        dataset_name=opts["dataset_name"],
        processor_name=opts.get("processor_name") or opts["model_id"],
        mode=opts["mode"],
        attacks=opts["attacks"],
        sample_count=opts["samples"] if opts["samples"] > 0 else None,
        batch_size=opts["batch_size"],
        custom_dataset_loader=dataset_loader,
        enable_xai=opts["enable_xai"],
        xai_techniques=opts["xai_techniques"],
        enable_hardening=opts["enable_hardening"],
        defense=opts["defense"],
        enable_retest=opts["enable_retest"],
        enable_report=opts["enable_report"],
        output_dir=opts["output_dir"],
    )

    # 3. Execute Orchestrator
    print(f"[3/3] Executing AdverScanOrchestrator across modules...\n")
    orchestrator = AdverScanOrchestrator()
    result = orchestrator.run(pipeline_config)

    # Execution Time Calculation
    total_time = time.time() - total_start_time
    mins, secs = divmod(total_time, 60)
    time_fmt = f"{int(mins)}m {secs:.2f}s" if mins > 0 else f"{total_time:.2f}s"

    # 4. Display Comprehensive Security Assessment Results
    print("\n" + "=" * 70)
    print("             ADVERSCAN SECURITY ASSESSMENT RESULTS SUMMARY          ")
    print("=" * 70)
    print(f"Overall Status:            {result.status}")
    print(f"Execution Mode:            {result.execution_mode.upper()}")
    print(f"Total Pipeline Runtime:    {time_fmt}")
    print(f"Target Modality / Domain:  {opts['domain'].upper()}")
    print(f"Evaluated Model:           {opts['model_name']} ({opts['model_id']})")
    print(f"Evaluated Dataset:         {opts['dataset_name']}")

    # Clean Baseline Results
    print("\n" + "-" * 70)
    print("M2 — CLEAN BASELINE EVALUATION METRICS")
    print("-" * 70)
    if result.baseline_evaluation:
        acc = result.baseline_evaluation.get("accuracy", 0.0)
        conf = result.baseline_evaluation.get("average_confidence", 0.0)
        f1 = result.baseline_evaluation.get("f1_macro", 0.0)
        print(f"  • Clean Accuracy:           {acc * 100:.2f}%")
        print(f"  • Clean Average Confidence: {conf * 100:.2f}%")
        print(f"  • Macro F1-Score:           {f1 * 100:.2f}%")
    else:
        print("  Baseline evaluation results unavailable.")

    # Adversarial Attack Results
    if result.attack_results:
        print("\n" + "-" * 70)
        print("M3 — ADVERSARIAL ATTACK ENGINE RESULTS")
        print("-" * 70)
        for atk_name, atk_data in result.attack_results.items():
            print(f"  ▶ Attack: [{atk_name.upper()}]")
            print(f"    Class:            {atk_data.get('attack_class')}")
            print(f"    Execution Time:   {atk_data.get('execution_time_seconds', 0.0):.2f}s")
            params = atk_data.get("parameters", {})
            if params:
                print(f"    Parameters:       {params}")

    # Vulnerability Scoring Results
    if result.vulnerability_analysis:
        print("\n" + "-" * 70)
        print("M5 — VULNERABILITY ANALYSIS & RISK SCORING")
        print("-" * 70)
        for atk_name, vuln in result.vulnerability_analysis.items():
            assessment = vuln.get("assessment", {})
            scoring = vuln.get("scoring", {})
            asr = assessment.get("attack_success_rate")
            asr_str = f"{asr * 100:.2f}%" if asr is not None else "N/A"
            adv_acc = assessment.get("adversarial_accuracy")
            adv_acc_str = f"{adv_acc * 100:.2f}%" if adv_acc is not None else "N/A"
            v_score = scoring.get("vulnerability_score")
            v_score_str = f"{v_score:.2f} / 100" if v_score is not None else "N/A"
            risk = scoring.get("risk_level", "UNKNOWN").upper()

            print(f"  ▶ Attack: [{atk_name.upper()}]")
            print(f"    Adversarial Accuracy:    {adv_acc_str}")
            print(f"    Attack Success Rate:     {asr_str}")
            print(f"    Vulnerability Score:     {v_score_str}")
            print(f"    Assessed Risk Level:     {risk}")

    # Hardening & Re-Test Results (if available)
    if result.hardening_results and result.hardening_results.get("selected_defense"):
        print("\n" + "-" * 70)
        print("M7 / M8 — HARDENING DEFENSE & RE-TEST COMPARISON")
        print("-" * 70)
        print(f"  • Selected Defense:        {result.hardening_results.get('selected_defense')}")
        if result.retest_results:
            rob_gain = result.retest_results.get("overall_robustness_gain")
            gain_str = f"{rob_gain * 100:+.2f}%" if rob_gain is not None else "N/A"
            print(f"  • Overall Robustness Gain: {gain_str}")

    # Generated Reports
    if result.report_result:
        print("\n" + "-" * 70)
        print("M9 — SECURITY ASSESSMENT REPORT")
        print("-" * 70)
        print(f"  ✓ Report ID:               {result.report_result.get('report_id')}")
        print(f"  ✓ Final Risk Level:        {result.report_result.get('risk_level')}")
        print(f"  ✓ Vulnerability Score:     {result.report_result.get('vulnerability_score')}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
