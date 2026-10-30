"""
AdverScan — End-to-End Multi-Modal & Multi-Domain Security Assessment CLI & Demo.

Supports interactive domain selection (Image, Text, Time-Series, Tabular),
domain-filtered model selection, automated dataset resolution (.csv, .pt, HuggingFace),
and domain-specific attack execution (single, multiple, or all attacks) through AdverScanOrchestrator:
M1 Model Ingestion -> M2 Baseline Evaluation -> M3 Attack Engine
-> M2 Adversarial Evaluation -> M5 Vulnerability Analysis -> M6 XAI -> M7 Hardening
-> M8 Re-Test & Comparison -> M9 Security Report Generator.
"""

import numpy as np
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
from app.ingestion.presets import DATASET_TO_RECOMMENDED_MODEL, KNOWN_DATASET_SAMPLE_COUNTS, MODEL_PRESETS
from app.orchestration import AdverScanOrchestrator, PipelineConfig
from app.utils import get_execution_device_info, resolve_device
from app.core.device_manager import DeviceManager

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
        help="Number of samples to evaluate ('full'/'0' for full dataset, 'half' for 50%%, or custom integer like 100)",
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
        "--xai-mode",
        type=str,
        default="standard",
        choices=["fast", "standard", "thorough"],
        help="Operational mode preset for XAI evaluation ('fast', 'standard', 'thorough')",
    )
    parser.add_argument(
        "--xai-techniques",
        nargs="+",
        default=["auto"],
        help="XAI techniques to apply ('auto' for model-adaptive routing, 'shap', 'lime', 'layer_ig', 'integrated_gradients', 'treeshap', 'saliency', 'gradcam')",
    )
    parser.add_argument(
        "--xai-budget",
        type=int,
        default=5,
        help="Maximum number of sample explanations to generate per attack (default: 5)",
    )
    parser.add_argument(
        "--xai-figures",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate rich multi-modal visual explanation figures & plots (default: False)",
    )
    parser.add_argument(
        "--xai-plan",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Preview the formatted XAI execution plan before execution",
    )
    parser.add_argument(
        "--xai-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ExplanationCache to reuse clean input attributions across attacks (default: True)",
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


# =====================================================================
# Total Sample Inspector (using KNOWN_DATASET_SAMPLE_COUNTS from presets.py)
# =====================================================================


def get_dataset_total_samples(
    domain: str,
    dataset_name: str,
    model_id: Optional[str] = None,
) -> Optional[int]:
    """
    Inspect local dataset file or lookup known dataset to determine total samples available.
    Supports CSV/TSV, PT/PTH tensors, NPZ/NPY files, and known benchmark datasets.
    """
    domain_clean = domain.lower().replace("_", "-")
    resolved_file: Optional[str] = None

    if domain_clean == "tabular":
        resolved_file = resolve_dataset_file(dataset_name, [".csv", ".tsv"], model_identifier=model_id) or resolve_dataset_file(None, [".csv", ".tsv"], model_identifier=model_id)
    elif domain_clean == "time-series":
        resolved_file = resolve_dataset_file(dataset_name, [".pt", ".pth", ".npz", ".npy"], model_identifier=model_id) or resolve_dataset_file(None, [".pt", ".pth", ".npz"], model_identifier=model_id)

    target_path = Path(resolved_file) if resolved_file else Path(dataset_name)

    if target_path.is_file():
        suffix = target_path.suffix.lower()
        if suffix in (".csv", ".tsv"):
            try:
                with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
                    cnt = sum(1 for line in f if line.strip()) - 1
                    if cnt > 0:
                        return cnt
            except Exception:
                pass
        elif suffix in (".pt", ".pth"):
            try:
                loaded = torch.load(target_path, map_location="cpu", weights_only=False)
                if isinstance(loaded, dict):
                    data = loaded.get("inputs", loaded.get("X", loaded.get("data")))
                    if data is not None:
                        return len(data)
                elif isinstance(loaded, (tuple, list)) and len(loaded) > 0:
                    return len(loaded[0])
                elif hasattr(loaded, "__len__"):
                    return len(loaded)
            except Exception:
                pass
        elif suffix in (".npz", ".npy"):
            try:
                import numpy as np
                arr = np.load(target_path)
                if isinstance(arr, np.ndarray):
                    return len(arr)
                elif hasattr(arr, "files") and len(arr.files) > 0:
                    return len(arr[arr.files[0]])
            except Exception:
                pass

    ds_key = dataset_name.lower().strip()
    if ds_key in KNOWN_DATASET_SAMPLE_COUNTS:
        return KNOWN_DATASET_SAMPLE_COUNTS[ds_key]
    for k, v in KNOWN_DATASET_SAMPLE_COUNTS.items():
        if k in ds_key or ds_key in k:
            return v

    return None


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
        p_classes = preset.get("num_classes", 2)
        p_task = preset.get("task_type", "classification").title()
        p_fw = preset.get("framework", "PyTorch").upper()
        p_proc = preset.get("processor_name", m_id)

        # Dimension / Specs
        specs = []
        if "input_resolution" in preset:
            specs.append(f"Input: {preset['input_resolution']}")
        if "num_features" in preset:
            specs.append(f"Features: {preset['num_features']}")
        if "sequence_length" in preset:
            specs.append(f"SeqLen: {preset['sequence_length']}, Channels: {preset.get('num_channels', 1)}")
        specs_str = f" | {', '.join(specs)}" if specs else ""

        # Check if local dataset file exists on disk
        resolved_local = None
        if preset.get("data_domain") == "tabular":
            resolved_local = resolve_dataset_file(rec_data, [".csv", ".tsv"], model_identifier=m_id)
        elif preset.get("data_domain") == "time-series":
            resolved_local = resolve_dataset_file(rec_data, [".pt", ".pth", ".npz"], model_identifier=m_id)

        data_desc = f"{rec_data} (found file: '{resolved_local}')" if resolved_local else rec_data

        print(f"  {idx}. {m_name}")
        print(f"     └─ Preset Key:       {p_key}")
        print(f"     └─ Model ID / Path:  {m_id}")
        print(f"     └─ Architecture:     {m_arch} ({p_fw})")
        print(f"     └─ Task & Classes:   {p_task} ({p_classes} classes){specs_str}")
        print(f"     └─ Preprocessing:    {p_proc}")
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
                    "preset_info": {"framework": "huggingface"},
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

    # Detect total samples present in the dataset
    total_samples = get_dataset_total_samples(domain, user_ds, model_id=model_id)
    half_samples = (total_samples // 2) if total_samples else 50
    default_sample = min(100, total_samples) if total_samples else 100

    print("\nSelect Number of Samples to Evaluate:")
    if total_samples is not None:
        print(f"  Total samples present in dataset: {total_samples:,}")
        print(f"  1. Full Dataset     ({total_samples:,} samples)")
        print(f"  2. Half Dataset     ({half_samples:,} samples)")
        print(f"  3. Custom Count     (write any custom number, e.g. 50, 100, 200)")
    else:
        print("  Total samples in dataset: Dynamic stream / Unknown")
        print("  1. Full Dataset     (Evaluate entire test split)")
        print("  2. Half Dataset     (Evaluate 50% split / 50 samples)")
        print("  3. Custom Count     (write any custom number, e.g. 100)")

    while True:
        prompt_text = f"Select sample option [1: Full, 2: Half, or write custom number] [{default_sample}]: "
        samples_str = input(prompt_text).strip().lower() or str(default_sample)

        if samples_str in ("1", "full", "all"):
            sample_count = 0  # 0 indicates full dataset in AdverScan
            total_disp = f"{total_samples:,}" if total_samples else "all"
            print(f"  ✓ Selected: Full Dataset ({total_disp} samples)")
            break
        elif samples_str in ("2", "half"):
            sample_count = half_samples
            print(f"  ✓ Selected: Half Dataset ({half_samples:,} samples)")
            break
        elif samples_str == "3":
            custom_in = input(f"Write custom number of samples [{default_sample}]: ").strip() or str(default_sample)
            try:
                val = int(custom_in)
                if val > 0:
                    sample_count = val
                    print(f"  ✓ Selected: Custom count ({sample_count:,} samples)")
                    break
                print("  ❌ Please enter a positive integer.")
            except ValueError:
                print("  ❌ Invalid number format.")
        else:
            try:
                val = int(samples_str)
                if val == 0:
                    sample_count = 0
                    total_disp = f"{total_samples:,}" if total_samples else "all"
                    print(f"  ✓ Selected: Full Dataset ({total_disp} samples)")
                    break
                elif val > 0:
                    sample_count = val
                    print(f"  ✓ Selected: {sample_count:,} samples")
                    break
                print("  ❌ Please enter a positive integer, '1' (Full), or '2' (Half).")
            except ValueError:
                print("  ❌ Invalid input. Enter '1' for Full, '2' for Half, or write a custom number.")

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
    xai_mode = "standard"
    xai_techniques = ["auto"]
    xai_budget = 5
    xai_figures = False
    xai_plan = False
    xai_cache = True
    enable_hardening = False
    defense = "auto"
    enable_retest = True
    enable_report = True

    if selected_mode == "full":
        xai_in = input("Enable M6 Explainable AI (XAI) feature attributions? [Y/n]: ").strip().lower() or "y"
        enable_xai = xai_in in ("y", "yes", "true", "1")
        if enable_xai:
            print("\n  Select XAI Evaluation Mode:")
            print("    1. Fast       (Rapid screening, fewer sampling steps)")
            print("    2. Standard   (Balanced speed & attribution fidelity) [Default]")
            print("    3. Thorough   (High-precision evaluations, maximum sampling steps)")
            xai_mode_choice = input("  Select Mode [2]: ").strip() or "2"
            xai_mode_map = {"1": "fast", "2": "standard", "3": "thorough"}
            xai_mode = xai_mode_map.get(xai_mode_choice, "standard")

            print("\n  Select XAI Explanation Technique:")
            print("    1. Auto (Model-Adaptive Routing via Rules Table) [Recommended]")
            print("    2. SHAP (Kernel / Sampling SHAP)")
            print("    3. LIME (Local Interpretable Model-agnostic Explanations)")
            print("    4. Integrated Gradients / Layer IG (Neural Networks / Transformers)")
            print("    5. TreeSHAP (Tree Ensembles: XGBoost, Random Forest)")
            print("    6. Saliency Map (Fast Gradient Saliency)")
            print("    7. Grad-CAM (Convolutional Networks)")
            print("    8. Multiple / Custom (Enter space-separated technique names)")
            xai_tech_choice = input("  Select Technique [1]: ").strip() or "1"
            if xai_tech_choice == "2":
                xai_techniques = ["shap"]
            elif xai_tech_choice == "3":
                xai_techniques = ["lime"]
            elif xai_tech_choice == "4":
                xai_techniques = ["integrated_gradients"]
            elif xai_tech_choice == "5":
                xai_techniques = ["treeshap"]
            elif xai_tech_choice == "6":
                xai_techniques = ["saliency"]
            elif xai_tech_choice == "7":
                xai_techniques = ["gradcam"]
            elif xai_tech_choice == "8":
                custom_techs = input("  Enter technique names (e.g. 'shap lime'): ").strip()
                xai_techniques = custom_techs.split() if custom_techs else ["auto"]
            else:
                xai_techniques = ["auto"]

            budget_in = input("  Sample budget for XAI explanations [5]: ").strip() or "5"
            try:
                xai_budget = max(1, int(budget_in))
            except ValueError:
                xai_budget = 5

            fig_in = input("  Generate multi-modal XAI visualization figures & plots? [y/N]: ").strip().lower()
            xai_figures = fig_in in ("y", "yes", "true", "1")

            plan_in = input("  Preview formatted XAI execution plan before running? [Y/n]: ").strip().lower() or "y"
            xai_plan = plan_in in ("y", "yes", "true", "1")

            cache_in = input("  Enable ExplanationCache to reuse clean attributions across attacks? [Y/n]: ").strip().lower() or "y"
            xai_cache = cache_in in ("y", "yes", "true", "1")

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
    target_device = "cpu" if dev_choice == "2" else str(DeviceManager.resolve_device(requested_device="auto"))

    return {
        "mode": selected_mode,
        "enable_xai": enable_xai,
        "xai_mode": xai_mode,
        "xai_techniques": xai_techniques,
        "xai_budget": xai_budget,
        "xai_figures": xai_figures,
        "xai_plan": xai_plan,
        "xai_cache": xai_cache,
        "enable_hardening": enable_hardening,
        "defense": defense,
        "enable_retest": enable_retest,
        "enable_report": enable_report,
        "device": target_device,
    }


def print_configuration_summary(opts: Dict[str, Any]):
    """Print visually organized configuration summary before running."""
    preset = opts.get("preset_info", {})
    print("\n" + "=" * 70)
    print("            ADVERSCAN MULTI-MODAL PIPELINE CONFIGURATION            ")
    print("=" * 70)
    print(f"Data Modality / Domain:     {opts['domain'].upper()}")
    print(f"Model Identifier:           {opts['model_id']}")
    print(f"Model Name:                 {opts['model_name']}")
    if preset.get("model_type"):
        print(f"Model Architecture:         {preset.get('model_type')}")
    if preset.get("task_type"):
        print(f"Task / Number of Classes:   {preset.get('task_type', '').title()} ({opts['num_classes']} classes)")
    if opts.get("processor_name"):
        print(f"Preprocessor / Config:      {opts['processor_name']}")
    samples_display = "Full Dataset" if opts["samples"] == 0 else f"{opts['samples']:,}"
    print(f"Evaluation Samples:         {samples_display}")
    print(f"Selected Attacks:           {', '.join(opts['attacks']).upper()}")
    print(f"Execution Mode:             {opts['mode'].upper()}")
    print(f"Target Device:              {opts['device'].upper()}")
    print(f"Batch Size:                 {opts['batch_size']}")
    if opts.get("enable_xai"):
        techs_str = ", ".join(opts["xai_techniques"]).upper()
        mode_str = str(opts.get("xai_mode", "standard")).upper()
        budget_str = f"{opts.get('xai_budget', 5)} samples/attack"
        figs_str = "Enabled" if opts.get("xai_figures") else "Disabled"
        cache_str = "Enabled" if opts.get("xai_cache") else "Disabled"
        print(f"XAI Enabled (M6):           True (Mode: {mode_str}, Techniques: {techs_str}, Budget: {budget_str}, Figures: {figs_str}, Cache: {cache_str})")
    else:
        print(f"XAI Enabled (M6):           False")
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

    # Guard: Detect text dataset inadvertently routed to image domain
    known_text_datasets = {"sst2", "stanfordnlp/sst2", "glue/sst2", "tweet_eval", "cardiffnlp/tweet_eval", "ag_news", "imdb"}
    if dataset_name.lower().strip() in known_text_datasets and domain_clean not in ("text", "nlp"):
        print(f"  [INFO] Auto-adjusting execution domain from '{domain}' to 'text' for NLP dataset '{dataset_name}'.")
        domain_clean = "text"

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

        # Detect channel-last (Transformer / PatchTST) vs channel-first (1D-CNN)
        is_channel_last = (
            "patchtst" in model_id.lower()
            or "patchtsmixer" in model_id.lower()
            or "transformer" in str(preset_info.get("model_type", "")).lower()
        )

        # Introspect Hugging Face config if sequence_length was not explicitly declared
        if "sequence_length" not in preset_info and ("huggingface" in str(preset_info.get("framework", "")).lower() or "/" in model_id):
            try:
                from transformers import AutoConfig
                cfg = AutoConfig.from_pretrained(model_id)
                seq_len = getattr(cfg, "context_length", getattr(cfg, "sequence_length", seq_len))
                num_ch = getattr(cfg, "num_input_channels", getattr(cfg, "in_channels", num_ch))
                num_cls_val = getattr(cfg, "num_classes", getattr(cfg, "num_targets", getattr(cfg, "num_labels", None)))
                if num_cls_val is not None:
                    num_classes = int(num_cls_val)
            except Exception:
                pass

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

            if isinstance(x_ts, np.ndarray):
                x_ts = torch.from_numpy(x_ts).float()
            elif isinstance(x_ts, torch.Tensor):
                x_ts = x_ts.float()

            if isinstance(x_ts, torch.Tensor):
                if x_ts.ndim == 2:
                    x_ts = x_ts.unsqueeze(-1) if is_channel_last else x_ts.unsqueeze(1)
                elif x_ts.ndim == 3:
                    if is_channel_last and x_ts.shape[1] == num_ch and x_ts.shape[2] == seq_len and num_ch != seq_len:
                        x_ts = x_ts.transpose(1, 2)
                    elif not is_channel_last and x_ts.shape[1] == seq_len and x_ts.shape[2] == num_ch and num_ch != seq_len:
                        x_ts = x_ts.transpose(1, 2)

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
            if is_channel_last:
                x_ts = torch.randn(n_eval, seq_len, num_ch)
            else:
                x_ts = torch.randn(n_eval, num_ch, seq_len)
            y_ts = torch.randint(0, num_classes, (n_eval,))
            dataset_loader = get_dataset_loader(
                dataset_name=dataset_name,
                data_domain="time-series",
                batch_size=batch_size,
                inputs=x_ts,
                targets=y_ts,
            )

        # Build sample input for time-series matching data shape
        sample_input = torch.randn(1, *tuple(x_ts.shape[1:]))
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
            "xai_mode": pipeline_opts.get("xai_mode", "standard"),
            "xai_techniques": pipeline_opts["xai_techniques"],
            "xai_budget": pipeline_opts.get("xai_budget", 5),
            "xai_figures": pipeline_opts.get("xai_figures", False),
            "xai_plan": pipeline_opts.get("xai_plan", False),
            "xai_cache": pipeline_opts.get("xai_cache", True),
            "enable_hardening": pipeline_opts["enable_hardening"],
            "defense": pipeline_opts["defense"],
            "enable_retest": pipeline_opts["enable_retest"],
            "enable_report": pipeline_opts["enable_report"],
            "output_dir": args.output_dir,
        }
    else:
        known_text = {"sst2", "stanfordnlp/sst2", "glue/sst2", "tweet_eval", "cardiffnlp/tweet_eval", "ag_news", "imdb"}
        raw_model_input = (args.model or "").strip()
        raw_dataset_input = (args.dataset or "").strip()

        inferred_domain = args.domain
        if not inferred_domain:
            if raw_dataset_input.lower() in known_text or raw_model_input.lower() in known_text:
                inferred_domain = "text"

        domain = (inferred_domain or "image").lower().replace("_", "-")
        default_models = {
            "image": "bazyl/gtsrb-model",
            "text": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            "time-series": "time-series-ecg-classifier",
            "tabular": "tabular-mlp-adult-census",
        }
        model_input = args.model or default_models.get(domain, "bazyl/gtsrb-model")
        preset = None
        if model_input in MODEL_PRESETS:
            preset = {"preset_key": model_input, **MODEL_PRESETS[model_input]}
        elif model_input.lower() in DATASET_TO_RECOMMENDED_MODEL:
            rec_model = DATASET_TO_RECOMMENDED_MODEL[model_input.lower()]
            if rec_model in MODEL_PRESETS:
                preset = {"preset_key": rec_model, **MODEL_PRESETS[rec_model]}
        else:
            for k, v in MODEL_PRESETS.items():
                if model_input in (v.get("model_id"), v.get("name"), k):
                    preset = {"preset_key": k, **v}
                    break

        if preset and not args.domain:
            domain = str(preset.get("data_domain", domain)).lower().replace("_", "-")
        model_id = preset.get("model_id", model_input) if preset else model_input
        model_name = preset.get("name", Path(model_id).stem) if preset else Path(model_id).stem

        # Default dataset from preset or resolved local file
        default_ds = "stanfordnlp/sst2" if domain == "text" else "bazyl/GTSRB"
        dataset = args.dataset or (preset.get("dataset_name") if preset else None) or default_ds
        if dataset.lower() in ("sst2", "glue/sst2"):
            dataset = "stanfordnlp/sst2"
        if domain == "tabular" and not args.dataset:
            resolved_csv = resolve_dataset_file(dataset, [".csv", ".tsv"], model_identifier=model_id)
            if resolved_csv:
                dataset = resolved_csv
        elif domain == "time-series" and not args.dataset:
            resolved_ts = resolve_dataset_file(dataset, [".pt", ".pth", ".npz", ".npy"], model_identifier=model_id)
            if resolved_ts:
                dataset = resolved_ts

        samples_str = str(args.samples).lower()
        if samples_str in ("0", "all", "full"):
            samples_count = 0
        elif samples_str == "half":
            total_s = get_dataset_total_samples(domain, dataset, model_id=model_id)
            samples_count = (total_s // 2) if total_s else 50
        else:
            try:
                samples_count = int(samples_str)
            except ValueError:
                samples_count = 100

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
            "xai_mode": str(args.xai_mode).lower(),
            "xai_techniques": [t.lower() for t in args.xai_techniques],
            "xai_budget": args.xai_budget,
            "xai_figures": args.xai_figures,
            "xai_plan": args.xai_plan,
            "xai_cache": args.xai_cache,
            "enable_hardening": args.enable_hardening,
            "defense": args.defense.lower(),
            "enable_retest": args.enable_retest,
            "enable_report": args.enable_report,
            "output_dir": args.output_dir,
        }

    print_configuration_summary(opts)

    # Unpack strongly-typed pipeline configuration parameters
    domain_val: str = str(opts.get("domain", "image"))
    model_id_val: str = str(opts.get("model_id", ""))
    model_name_val: str = str(opts.get("model_name", "TargetModel"))
    dataset_name_val: str = str(opts.get("dataset_name", ""))
    proc_name_val: Optional[str] = str(opts["processor_name"]) if opts.get("processor_name") else None
    num_classes_val: int = int(opts["num_classes"]) if "num_classes" in opts and isinstance(opts["num_classes"], (int, float, str)) else 2
    preset_info_val: Dict[str, Any] = opts["preset_info"] if isinstance(opts.get("preset_info"), dict) else {}
    samples_val: int = int(opts["samples"]) if "samples" in opts and isinstance(opts["samples"], (int, float, str)) else 0
    attacks_val: List[str] = [str(a) for a in opts["attacks"]] if isinstance(opts.get("attacks"), list) else ["fgsm"]
    mode_val: str = str(opts.get("mode", "full"))
    device_val: str = str(opts.get("device", "cpu"))
    batch_size_val: int = int(opts["batch_size"]) if "batch_size" in opts and isinstance(opts["batch_size"], (int, float, str)) else 32
    enable_xai_val: bool = bool(opts.get("enable_xai", False))
    xai_techniques_val: List[str] = [str(t) for t in opts["xai_techniques"]] if isinstance(opts.get("xai_techniques"), list) else []
    xai_mode_val: str = str(opts.get("xai_mode", "standard"))
    raw_xai_budget = opts.get("xai_budget", 5)
    xai_budget_val: int = int(raw_xai_budget) if isinstance(raw_xai_budget, (int, str)) and not isinstance(raw_xai_budget, bool) else 5
    xai_figures_val: bool = bool(opts.get("xai_figures", False))
    xai_plan_val: bool = bool(opts.get("xai_plan", False))
    xai_cache_val: bool = bool(opts.get("xai_cache", True))
    enable_hardening_val: bool = bool(opts.get("enable_hardening", False))
    defense_val: str = str(opts.get("defense", "auto"))
    enable_retest_val: bool = bool(opts.get("enable_retest", True))
    enable_report_val: bool = bool(opts.get("enable_report", True))
    output_dir_val: Optional[str] = str(opts["output_dir"]) if opts.get("output_dir") else None

    # Display XAI Execution Plan if enabled
    if enable_xai_val and xai_plan_val:
        try:
            from app.explainability.pipeline import display_xai_plan
            from app.explainability.techniques.registry import RULES_TABLE, detect_model_type, normalize_model_type
            m_arch = detect_model_type(model_id_val, domain_val)
            canon_arch = normalize_model_type(m_arch)
            chosen_tech = xai_techniques_val[0] if xai_techniques_val else "auto"
            if chosen_tech == "auto":
                chosen_tech = str(RULES_TABLE.get(canon_arch, {}).get(xai_mode_val, "integrated_gradients"))
                reason = f"{m_arch} + {xai_mode_val} mode (routing table)"
            else:
                reason = f"Manual override: {chosen_tech}"
            display_xai_plan(
                model_name=model_name_val,
                model_type=m_arch,
                modality=domain_val,
                device=device_val,
                mode=xai_mode_val,
                technique=chosen_tech,
                selection_reason=reason,
                budget=xai_budget_val,
                total_samples=samples_val if samples_val > 0 else 100,
            )
        except Exception as plan_err:
            print(f"  [XAI Plan Preview Error: {plan_err}]")

    if is_interactive:
        confirm = input("▶ Proceed and launch security assessment? [Y/n]: ").strip().lower()
        if confirm in ("n", "no"):
            print("  Assessment aborted by user.")
            sys.exit(0)

    # 1. Prepare Dataset Loader and Sample Input
    print(f"\n[1/3] Preparing dataset loader for '{dataset_name_val}' ({domain_val.upper()})...")
    try:
        dataset_loader, sample_input = build_dataset_and_sample_input(
            domain=domain_val,
            model_id=model_id_val,
            dataset_name=dataset_name_val,
            processor_name=proc_name_val,
            samples=samples_val,
            batch_size=batch_size_val,
            preset_info=preset_info_val,
        )
    except Exception as e:
        print(f"  ❌ Error initializing dataset loader: {e}")
        sys.exit(1)

    # Prepare attack-specific configs if required (e.g. VAE for tabular manifold attacks)
    attack_configs = {}
    if domain_val == "tabular" and any(a.lower() == "vae_attack" for a in attacks_val):
        class _SimpleTabularVAE(torch.nn.Module):
            def __init__(self, num_features: int, latent_dim: int = 4):
                super().__init__()
                latent = max(2, min(latent_dim, num_features // 2))
                self.encoder = torch.nn.Linear(num_features, latent)
                self.decoder = torch.nn.Linear(latent, num_features)

            def encode(self, x: torch.Tensor):
                z = self.encoder(x)
                return z, z

            def decode(self, z: torch.Tensor):
                return self.decoder(z)

        n_feats = sample_input.shape[-1] if hasattr(sample_input, "shape") else 30
        fallback_vae = _SimpleTabularVAE(num_features=n_feats).to(device_val)
        attack_configs["vae_attack"] = {
            "epsilon": 0.5,
            "params": {"vae": fallback_vae, "num_steps": 5, "alpha": 0.05},
        }

    # 2. Build PipelineConfig
    print(f"[2/3] Configuring AdverScan Pipeline (Mode: {mode_val.upper()})...")
    sample_cnt = samples_val if samples_val > 0 else None

    # Hardware Guard: Enforce strict GPU binding or safe CPU workload scaling
    resolved_dev = DeviceManager.resolve_device(device_val)
    device_val = str(resolved_dev)

    est_params = preset_info_val.get("parameters", 0) if isinstance(preset_info_val, dict) else 0
    if not est_params:
        mid_low = model_id_val.lower()
        if any(k in mid_low for k in ("vit", "resnet", "gtsrb")):
            est_params = 85_800_000
        elif any(k in mid_low for k in ("bert", "roberta", "t5")):
            est_params = 110_000_000

    if sample_cnt is not None:
        sample_cnt, safe_batch = DeviceManager.adjust_workload_for_hardware(
            sample_count=sample_cnt,
            model_param_count=est_params,
            device=resolved_dev,
        )
        batch_size_val = min(batch_size_val, safe_batch)
    pipeline_config = PipelineConfig(
        attack_configs=attack_configs,
        model_path=model_id_val,
        sample_input=sample_input,
        device=device_val,
        model_name=model_name_val,
        num_classes=num_classes_val,
        dataset_name=dataset_name_val,
        processor_name=proc_name_val or model_id_val,
        mode=mode_val,
        attacks=attacks_val,
        sample_count=sample_cnt,
        batch_size=batch_size_val,
        custom_dataset_loader=dataset_loader,
        enable_xai=enable_xai_val,
        xai_techniques=xai_techniques_val,
        xai_mode=xai_mode_val,
        xai_budget=xai_budget_val,
        xai_figures=xai_figures_val,
        xai_cache=xai_cache_val,
        enable_hardening=enable_hardening_val,
        defense=defense_val,
        enable_retest=enable_retest_val,
        enable_report=enable_report_val,
        output_dir=output_dir_val,
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
    print(f"Target Modality / Domain:  {str(opts['domain']).upper()}")
    print(f"Evaluated Model:           {opts['model_name']} ({opts['model_id']})")
    print(f"Evaluated Dataset:         {opts['dataset_name']}")

    if result.errors:
        print("\n" + "-" * 70)
        print("PIPELINE ERRORS & DIAGNOSTICS")
        print("-" * 70)
        for err in result.errors:
            print(f"  ❌ [{err.get('module')}]: {err.get('error_type')} — {err.get('message')}")

    # M1 — Ingested Model Specifications & Parameter Details
    print("\n" + "-" * 70)
    print("M1 — MODEL SPECIFICATIONS & PARAMETERS")
    print("-" * 70)
    meta = result.model_metadata or {}
    print(f"  • Model Identifier:        {meta.get('model_name') or opts['model_name']} ({opts['model_id']})")
    if meta.get("model_architecture"):
        print(f"  • Architecture:            {meta.get('model_architecture')}")
    total_p = meta.get("total_parameters")
    if total_p is not None:
        train_p = meta.get("trainable_parameters") if meta.get("trainable_parameters") is not None else total_p
        train_pct = (train_p / total_p * 100) if total_p > 0 else 100.0
        print(f"  • Total Parameters:        {total_p:,} (~{total_p / 1e6:.2f}M)")
        print(f"  • Trainable Parameters:    {train_p:,} ({train_pct:.1f}%)")
        print(f"  • Estimated Model Size:    ~{total_p * 4 / (1024 * 1024):.1f} MB (FP32)")
    if meta.get("input_shape"):
        print(f"  • Input Shape:             {meta.get('input_shape')}")
    if meta.get("num_classes"):
        print(f"  • Total Classes:           {meta.get('num_classes')}")
    if meta.get("task_type"):
        print(f"  • Task Type:               {meta.get('task_type')}")
    if meta.get("framework"):
        print(f"  • Ingestion Framework:     {meta.get('framework')}")
    print(f"  • Execution Device:        {meta.get('device') or opts['device']}")

    # Dataset & Evaluation Details
    print("\n" + "-" * 70)
    print("DATASET & EVALUATION DETAILS")
    print("-" * 70)
    print(f"  • Dataset Identifier:      {dataset_name_val}")
    print(f"  • Target Domain:           {domain_val.upper()}")
    samples_disp = f"{samples_val:,}" if samples_val > 0 else "Full Dataset"
    print(f"  • Evaluated Samples:       {samples_disp}")
    print(f"  • Evaluation Batch Size:   {batch_size_val}")

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
        total_attack_time = sum(atk_data.get("execution_time_seconds", 0.0) for atk_data in result.attack_results.values())
        print(f"Total Attack Execution Time: {total_attack_time:.2f}s across {len(result.attack_results)} attack(s)")
        for atk_name, atk_data in result.attack_results.items():
            print(f"\n  ▶ Attack: [{atk_name.upper()}]")
            print(f"    • Attack Class:          {atk_data.get('attack_class')}")
            print(f"    • Execution Time:        {atk_data.get('execution_time_seconds', 0.0):.2f}s")
            params = atk_data.get("parameters", {})
            if params:
                param_strs = [f"{k}={v}" for k, v in params.items()]
                print(f"    • Configuration:         {', '.join(param_strs)}")

    # Vulnerability Scoring & Security Assessment Results
    if result.vulnerability_analysis:
        print("\n" + "-" * 70)
        print("M5 — VULNERABILITY ANALYSIS & SECURITY RISK SCORING")
        print("-" * 70)

        # Display Model-Level Roll-up if available
        model_rob = result.vulnerability_analysis.get("__model_robustness__")
        if model_rob:
            m_vscore = model_rob.get("vulnerability_score")
            m_vscore_str = f"{m_vscore:.2f} / 100" if m_vscore is not None else "INCONCLUSIVE"
            m_rscore = model_rob.get("robustness_score")
            m_rscore_str = f"{m_rscore:.2f} / 100" if m_rscore is not None else "N/A"
            m_risk = model_rob.get("risk_level", "UNKNOWN").upper()
            driving = model_rob.get("driving_attack") or "None"

            print(f"  ★ MODEL-LEVEL UNIFIED SECURITY ASSESSMENT:")
            print(f"    • Model Vulnerability Score: {m_vscore_str}")
            print(f"    • Model Robustness Score:    {m_rscore_str}")
            print(f"    • Unified Security Risk:     {m_risk}")
            print(f"    • Primary Driving Threat:    {driving.upper()}")
            print(f"    • Conclusive Evaluations:    {model_rob.get('valid_attacks_count', 0)} / {model_rob.get('evaluated_attacks_count', 0)}")
            print("-" * 70)

        # Iterate over individual attack assessments
        for atk_name, vuln in result.vulnerability_analysis.items():
            if atk_name.startswith("__"):
                continue
            assessment = vuln.get("assessment", {})
            scoring = vuln.get("scoring", {})

            # Performance metrics
            clean_acc = assessment.get("clean_accuracy", 0.0)
            adv_acc = assessment.get("adversarial_accuracy", 0.0)
            acc_drop = assessment.get("accuracy_drop", 0.0)

            clean_conf = assessment.get("clean_confidence", 0.0)
            adv_conf = assessment.get("adversarial_confidence", 0.0)
            conf_drop = assessment.get("confidence_drop", 0.0)

            clean_f1 = assessment.get("clean_f1", 0.0)
            adv_f1 = assessment.get("adversarial_f1", 0.0)
            f1_drop = assessment.get("f1_drop", 0.0)

            model_deg = assessment.get("model_degradation", 0.0)

            # ASR and efficacy
            asr = assessment.get("attack_success_rate")
            asr_str = f"{asr * 100:.2f}%" if asr is not None else "N/A"
            asr_reason = assessment.get("attack_success_rate_reason")
            asr_display = f"{asr_str} ({asr_reason})" if asr_reason and asr is None else asr_str

            atk_meta = assessment.get("metadata") or assessment.get("extra_metadata") or {}
            eff = atk_meta.get("attack_effective")
            eff_str = "Effective (Perturbations Applied)" if eff is True else ("No-op / Ineffective" if eff is False else "Estimated")

            # Perturbations
            pert = assessment.get("perturbation", {})
            linf = pert.get("linf_mean") if isinstance(pert, dict) else getattr(pert, "linf_mean", None)
            l2 = pert.get("l2_mean") if isinstance(pert, dict) else getattr(pert, "l2_mean", None)
            l0 = pert.get("l0_mean") if isinstance(pert, dict) else getattr(pert, "l0_mean", None)
            tok_changed = atk_meta.get("tokens_changed_rate") or l0

            # Scoring
            v_score = scoring.get("vulnerability_score")
            v_score_str = f"{v_score:.2f} / 100" if v_score is not None else "INCONCLUSIVE"
            risk = scoring.get("risk_level", "UNKNOWN").upper()
            sub = scoring.get("sub_scores", {})

            # Execution time
            atk_exec_time = result.attack_results.get(atk_name, {}).get("execution_time_seconds", 0.0)

            print(f"\n  ▶ Attack: [{atk_name.upper()}]")
            print(f"    • Risk Severity Level:   {risk}")
            print(f"    • Vulnerability Score:   {v_score_str}")
            print(f"    • Attack Success Rate:   {asr_display}")
            print(f"    • Attack Efficacy:       {eff_str}")
            print(f"    • Attack Execution Time: {atk_exec_time:.2f}s")
            print(f"    • Accuracy Impact:       {clean_acc * 100:.2f}% → {adv_acc * 100:.2f}% (Drop: -{acc_drop * 100:.2f}%)")
            print(f"    • Confidence Impact:     {clean_conf * 100:.2f}% → {adv_conf * 100:.2f}% (Drop: -{conf_drop * 100:.2f}%)")
            print(f"    • Macro F1 Impact:       {clean_f1 * 100:.2f}% → {adv_f1 * 100:.2f}% (Drop: -{f1_drop * 100:.2f}%)")
            print(f"    • Model Degradation:     {model_deg * 100:.2f}%")

            pert_details = []
            if linf is not None:
                pert_details.append(f"L_inf: {linf:.4f}")
            if l2 is not None:
                pert_details.append(f"L_2: {l2:.4f}")
            if tok_changed is not None:
                pert_details.append(f"Tokens/L_0: {tok_changed * 100:.2f}%")
            if pert_details:
                print(f"    • Perturbation Norms:    {', '.join(pert_details)}")

            if sub:
                sub_parts = []
                if "asr_score" in sub and sub["asr_score"] is not None:
                    sub_parts.append(f"ASR={sub['asr_score']:.1f}")
                if "degradation_score" in sub and sub["degradation_score"] is not None:
                    sub_parts.append(f"Degradation={sub['degradation_score']:.1f}")
                if "stealth_score" in sub and sub["stealth_score"] is not None:
                    sub_parts.append(f"Stealth={sub['stealth_score']:.1f}")
                if sub_parts:
                    print(f"    • Score Components:      {', '.join(sub_parts)}")

    # M6 — Explainable AI (XAI) Attribution & Feature Shift Analysis
    if result.xai_results:
        print("\n" + "-" * 70)
        print("M6 — EXPLAINABLE AI (XAI) ATTRIBUTION & SHIFT ANALYSIS")
        print("-" * 70)
        for key, xai_data in result.xai_results.items():
            atk_n = xai_data.get("attack_name", key).upper()
            tech = xai_data.get("technique", "XAI").upper()
            status = xai_data.get("attribution_status", "success").upper()
            modality = xai_data.get("modality", "unknown").upper()

            print(f"\n  ▶ Explanation: [{atk_n} — {tech}] (Modality: {modality}, Status: {status})")

            # XAI Decision & Model-Adaptive Routing
            x_dec = xai_data.get("xai_decision") or {}
            if x_dec:
                sel_tech = str(x_dec.get("technique_selected", tech)).upper()
                reason = x_dec.get("selection_reason", "Adaptive routing")
                m_type = x_dec.get("model_type", "neural_network")
                dev_str = str(x_dec.get("device", "cpu")).upper()
                print(f"    • Routing Decision:      {sel_tech} ({reason})")
                print(f"    • Inferred Architecture: {m_type} | Device: {dev_str}")
                if x_dec.get("fallback"):
                    print(f"    • Fallback Triggered:    YES (Execution timeout reached; gracefully fell back)")
                if "explain_seconds" in x_dec:
                    print(f"    • XAI Execution Time:    {float(x_dec.get('explain_seconds', 0.0)):.4f}s")

            clean_p = xai_data.get("clean_prediction")
            adv_p = xai_data.get("adversarial_prediction")
            clean_c = xai_data.get("clean_confidence")
            adv_c = xai_data.get("adversarial_confidence")
            pred_changed = xai_data.get("prediction_changed")
            caused_fail = xai_data.get("attack_caused_failure")

            if clean_p is not None and adv_p is not None:
                chg_str = "YES (Adversarial Flip Induced)" if pred_changed else "NO (Prediction Preserved)"
                clean_c_str = f"{clean_c * 100:.2f}%" if clean_c is not None else "N/A"
                adv_c_str = f"{adv_c * 100:.2f}%" if adv_c is not None else "N/A"
                print(f"    • Prediction Shift:      Clean: {clean_p} (Conf: {clean_c_str}) → Adv: {adv_p} (Conf: {adv_c_str})")
                print(f"    • Decision Flipped:      {chg_str}")
            if caused_fail is not None:
                print(f"    • Attack Caused Failure: {'YES' if caused_fail else 'NO'}")

            comp = xai_data.get("comparison", {})
            if comp:
                cos_sim = comp.get("cosine_similarity")
                pearson = comp.get("pearson_correlation")
                attr_shift = comp.get("attribution_shift")
                topk_ov = comp.get("top_k_overlap")

                metrics = []
                if cos_sim is not None:
                    metrics.append(f"Cosine Sim={cos_sim:.4f}")
                if pearson is not None:
                    metrics.append(f"Pearson r={pearson:.4f}")
                if attr_shift is not None:
                    metrics.append(f"Attr Shift={attr_shift:.4f}")
                if topk_ov is not None:
                    metrics.append(f"Top-K Overlap={topk_ov * 100:.1f}%")
                if metrics:
                    print(f"    • Attribution Alignment: {', '.join(metrics)}")

            feat_imp = xai_data.get("feature_importance", {})
            clean_top = feat_imp.get("clean_top_features") or []
            adv_top = feat_imp.get("adversarial_top_features") or []
            shifted_top = feat_imp.get("highest_shifted_features") or []

            def _format_feats(feats):
                if not feats:
                    return "None"
                items = []
                for f in feats[:5]:
                    if isinstance(f, dict):
                        f_name = f.get("feature", f.get("token", f.get("name", "?")))
                        f_score = f.get("score", f.get("weight", f.get("attribution", 0.0)))
                        items.append(f"'{f_name}' ({f_score:+.3f})")
                    elif isinstance(f, (list, tuple)) and len(f) >= 2:
                        items.append(f"'{f[0]}' ({float(f[1]):+.3f})")
                    else:
                        items.append(str(f))
                return ", ".join(items)

            if clean_top:
                print(f"    • Clean Top Features:    {_format_feats(clean_top)}")
            if adv_top:
                print(f"    • Adv Top Features:      {_format_feats(adv_top)}")
            if shifted_top:
                print(f"    • Most Shifted Features: {_format_feats(shifted_top)}")

            # Failure Analysis, Root-Cause & Defense Suggestions
            fail_data = xai_data.get("failure_analysis", {})
            top_contrib = fail_data.get("contributing_features", [])
            if top_contrib:
                print(f"\n    [Why This Result Came — Diagnostic Feature Shifts]")
                for feat in top_contrib[:4]:
                    name = feat.get("feature", "feature")
                    c_imp = feat.get("clean_importance", 0.0)
                    a_imp = feat.get("adversarial_importance", 0.0)
                    shift = feat.get("importance_shift", 0.0)
                    rank = feat.get("rank", 1)
                    print(f"      • #{rank} '{name}': Clean={c_imp:.4f} → Adv={a_imp:.4f} (Shift: {shift:+.4f})")

            attack_reason = fail_data.get("attack_reason")
            if attack_reason:
                print(f"\n    [Why This Attack Succeeded — Root-Cause Narrative]")
                print(f"      {attack_reason}")

            hints = xai_data.get("defense_hints") or fail_data.get("defense_hints", [])
            if hints:
                print(f"\n    [Defense Hints — Quantitative Evidence & Targeted Mitigations]")
                for h in hints:
                    f_text = h.get("finding", "")
                    fam = h.get("suggested_family", "defense")
                    stg = h.get("stage", "training")
                    conf = h.get("confidence", "high")
                    print(f"      • Finding:     \"{f_text}\"")
                    print(f"        └─ Suggested Family: {fam} (Stage: {stg}, Confidence: {conf})")
                    if "evidence" in h and isinstance(h["evidence"], dict):
                        ev_str = ", ".join(f"{k}={v}" for k, v in h["evidence"].items())
                        print(f"        └─ Evidence:         {ev_str}")

            pipeline = xai_data.get("defense_pipeline") or fail_data.get("defense_pipeline", {})
            if pipeline:
                print(f"\n    [What Defense We Can Apply — 6-Stage Defense Pipeline Architecture]")
                print("      [1. Preprocessing] → [2. Transformation] → [3. Detection] → [4. Smoothing] → [5. Feature-level] → [6. Ensemble]")
                for stg_key in sorted(pipeline.keys()):
                    stg_data = pipeline[stg_key]
                    if isinstance(stg_data, dict):
                        stg_name = stg_data.get("stage", stg_key)
                        stg_focus = stg_data.get("focus", "")
                        techs = stg_data.get("techniques", [])
                        print(f"      • {stg_name} ({stg_focus}): {', '.join(techs)}")

            if "batch_results" in xai_data:
                b_res = xai_data["batch_results"]
                print(f"\n    • Batch Explanations:    {len(b_res)} sample(s) evaluated")

            # Check if artifacts exist on disk
            if pipeline_config and pipeline_config.output_dir:
                xai_dir = Path(pipeline_config.output_dir) / "xai"
                expected_json = xai_dir / f"{key}_explanation.json"
                if expected_json.exists():
                    print(f"    • Exported JSON Report:  {expected_json}")
                figs_dir = xai_dir / "figures"
                if figs_dir.exists():
                    fig_files = list(figs_dir.glob("*.png")) + list(figs_dir.glob("*.pdf"))
                    if fig_files:
                        print(f"    • Visual Figures:        {len(fig_files)} plots generated in {figs_dir}")

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
