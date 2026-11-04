"""
AdverScan — Dedicated Adversarial Attack Engine Evaluation & Perturbation Assessment CLI.

Performs End-to-End Attack Engine Evaluation through Pipeline Stages:
M1: Model Ingestion -> M2: Clean Baseline Evaluation -> M3: Attack Engine Execution
-> M2: Adversarial Evaluation -> M5: Vulnerability Analysis & Risk Scoring.

Evaluates Multi-Objective Optimization Targets:
1. High Attack Success Rate (ASR)
2. Low L-infinity (L_inf) Perturbation Norm
3. Low L-2 (L_2) Euclidean Perturbation Norm
4. Low L-0 (L_0) Sparsity / Modified Elements
5. Minimal Degradation in Visual Quality (Peak Signal-to-Noise Ratio PSNR & Structural Similarity SSIM)

Supports interactive terminal wizard for attack constraints (epsilon, alpha, steps, norm, clipping)
as well as full scriptable CLI execution.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
from dotenv import load_dotenv

# Suppress non-critical third-party warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.jit")

# Ensure repository root directory is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.attack_engine import discover_attacks
from app.attack_engine.config import AttackConfig
from app.core.device_manager import DeviceManager
from app.ingestion.model_registry import ModelPresetRegistry
from app.ingestion.presets import DATASET_TO_RECOMMENDED_MODEL, KNOWN_DATASET_SAMPLE_COUNTS, MODEL_PRESETS
from app.orchestration import AdverScanOrchestrator, PipelineConfig
from app.utils import get_execution_device_info
from scripts.run_pipeline_for_multimodel import (
    DOMAIN_ATTACKS_CATALOG,
    build_dataset_and_sample_input,
    get_dataset_total_samples,
    prompt_dataset_selection,
    prompt_domain_selection,
    prompt_model_selection,
)

load_dotenv()

# Check for skimage visual quality metrics
try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


# =====================================================================
# CLI Arguments Parser
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="AdverScan Dedicated Adversarial Attack Engine & Perturbation Evaluation CLI"
    )
    # Core Model & Dataset Configuration
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        choices=["image", "text", "time-series", "tabular"],
        help="Target model domain / data modality (default: auto-detected or image)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model preset key, HuggingFace repository ID, or local file path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name, HuggingFace dataset ID, or local file path",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="10",
        help="Number of samples to evaluate (e.g. 10, 50, 'all', 'half') (default: 10)",
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=None,
        help="Adversarial attack(s) to evaluate (e.g. 'fgsm', 'pgd', 'cw', 'all')",
    )

    # Attack Constraints & Hyperparameters
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Perturbation budget / magnitude epsilon (e.g. 0.03137 ~ 8/255 for Linf image attacks)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of iterations / steps for iterative attacks (e.g. PGD, CW) (default: 20)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Iteration step size alpha (default: auto 2*eps/steps or 0.0078)",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="Linf",
        choices=["Linf", "L2", "L0"],
        help="Perturbation distance norm constraint ('Linf' | 'L2' | 'L0') (default: Linf)",
    )
    parser.add_argument(
        "--clip-min",
        type=float,
        default=None,
        help="Minimum tensor clipping bound (default: None / unclipped to prevent mangling normalized inputs)",
    )
    parser.add_argument(
        "--clip-max",
        type=float,
        default=None,
        help="Maximum tensor clipping bound (default: None / unclipped)",
    )
    parser.add_argument(
        "--targeted",
        action="store_true",
        help="Enable targeted attack mode (default: False / untargeted)",
    )

    # Hardware & Performance
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Evaluation batch size (default: 16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device ('cuda', 'cuda:0', 'mps', or 'cpu')",
    )

    # Artifacts & Reporting
    parser.add_argument(
        "--save-visuals",
        action="store_true",
        help="Save comparison image grids (clean vs adversarial vs diff) to output directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save evaluation reports and generated artifacts (default: results)",
    )

    # Interaction Mode
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force interactive terminal prompts",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Disable interactive terminal prompts and use CLI flags / defaults",
    )

    return parser.parse_args()


# =====================================================================
# Interactive Attack Selection & Constraints Prompts
# =====================================================================

def prompt_attack_selection(domain: str) -> List[str]:
    """Display available attacks for the domain and prompt user to select."""
    discover_attacks()
    catalog_list = DOMAIN_ATTACKS_CATALOG.get(domain, [])

    print(f"\n[Step 4/6] Select Adversarial Attack(s) for Domain [{domain.upper()}]:")
    print("-" * 72)
    for idx, atk in enumerate(catalog_list, 1):
        print(f"  {idx}. [{atk['name']}] — {atk['full_name']}")
        print(f"     └─ Category: {atk['category']}")
        print(f"     └─ Description: {atk['description']}")
    print(f"  A. ALL Attacks (Benchmark all {len(catalog_list)} attacks for this domain)")
    print("-" * 72)

    while True:
        raw_choice = input(f"Select attack(s) (e.g. '1', '1 2', or 'all') [1]: ").strip().lower() or "1"
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
            return list(dict.fromkeys(selected))
        print("  ❌ No recognized attacks selected. Please enter valid attack numbers or 'all'.")


def prompt_attack_constraints(
    domain: str,
    selected_attacks: List[str],
    cli_args: argparse.Namespace,
) -> Dict[str, AttackConfig]:
    """
    Prompt user for specific attack constraints (epsilon, alpha, steps, norm, clipping)
    aiming for high ASR with minimal perturbation and high visual quality.
    """
    print("\n" + "=" * 72)
    print("  [Step 5/6] ATTACK CONSTRAINTS & PERTURBATION OBJECTIVES CONFIGURATION  ")
    print("=" * 72)
    print("  Goal: High ASR • Low L_inf • Low L_2 • Low L_0 • High Visual Quality")
    print("  Tip : Smaller epsilon keeps perturbation imperceptible (higher PSNR/SSIM).")
    print("        Iterative attacks (PGD/CW) achieve higher ASR at smaller epsilons.")
    print("-" * 72)

    # 1. Perturbation Budget (Epsilon)
    if cli_args.epsilon is not None:
        chosen_eps = float(cli_args.epsilon)
        print(f"  • Epsilon (CLI override): {chosen_eps}")
    else:
        default_eps_map = {
            "image": 0.03137,  # 8 / 255
            "text": 0.1,
            "tabular": 0.1,
            "time-series": 0.05,
        }
        rec_eps = default_eps_map.get(domain, 0.03137)
        if domain == "image":
            print(f"  Recommended Epsilon for Image Domain:")
            print(f"    - 0.0157 (~4/255) : Ultra-stealthy / Near-zero visual perturbation")
            print(f"    - 0.0314 (~8/255) : Standard benchmarking default (Recommended)")
            print(f"    - 0.0627 (~16/255): Aggressive perturbation (Higher ASR, noticeable distortion)")
        prompt_text = f"  Enter perturbation budget epsilon (ε) [{rec_eps}]: "
        eps_input = input(prompt_text).strip()
        try:
            chosen_eps = float(eps_input) if eps_input else rec_eps
        except ValueError:
            chosen_eps = rec_eps
        print(f"  ✓ Epsilon set to: {chosen_eps:.5f}")

    # 2. Iteration Steps (for iterative attacks like PGD, CW, FAB)
    has_iterative = any(a.lower() in ("pgd", "cw", "deepfool", "fab", "embedding_pgd") for a in selected_attacks)
    if cli_args.steps is not None:
        chosen_steps = cli_args.steps
    elif has_iterative:
        print(f"\n  Iterative Steps Configuration:")
        print(f"  Tip: More iterations (e.g. 20-40) allow smaller step sizes to find minimal perturbations.")
        steps_input = input(f"  Enter number of attack iterations / steps [20]: ").strip()
        try:
            chosen_steps = int(steps_input) if steps_input else 20
        except ValueError:
            chosen_steps = 20
        print(f"  ✓ Iteration steps set to: {chosen_steps}")
    else:
        chosen_steps = 10

    # 3. Step size Alpha
    if cli_args.alpha is not None:
        chosen_alpha = cli_args.alpha
    elif has_iterative:
        default_alpha = max(0.001, round(2.0 * chosen_eps / chosen_steps, 5))
        alpha_input = input(f"  Enter step size alpha (α) [{default_alpha:.5f}]: ").strip()
        try:
            chosen_alpha = float(alpha_input) if alpha_input else default_alpha
        except ValueError:
            chosen_alpha = default_alpha
        print(f"  ✓ Step size alpha set to: {chosen_alpha:.5f}")
    else:
        chosen_alpha = chosen_eps

    # 4. Norm Constraint
    chosen_norm = cli_args.norm or "Linf"
    if domain == "image" and cli_args.norm == "Linf":
        norm_input = input(f"  Enter perturbation norm constraint ('Linf' | 'L2') [Linf]: ").strip()
        if norm_input.lower() in ("l2", "2"):
            chosen_norm = "L2"
        else:
            chosen_norm = "Linf"
    print(f"  ✓ Norm constraint set to: {chosen_norm}")

    # 5. Clipping bounds
    if cli_args.clip_min is not None or cli_args.clip_max is not None:
        clip_min = cli_args.clip_min
        clip_max = cli_args.clip_max
        print(f"  • Clipping bounds: [{clip_min}, {clip_max}]")
    else:
        clip_prompt = input("  Enable tensor clipping? (enter 'y' for [0.0, 1.0], or hit Enter for unclipped/auto) [auto]: ").strip().lower()
        if clip_prompt in ("y", "yes", "1"):
            clip_min = 0.0
            clip_max = 1.0
            print("  ✓ Clipping bounds set to: [0.0, 1.0]")
        else:
            clip_min = None
            clip_max = None
            print("  ✓ Clipping bounds: Unclipped / Preserves dataset normalization")

    # 6. Targeting mode
    is_targeted = cli_args.targeted

    # Construct AttackConfig per selected attack
    attack_configs: Dict[str, AttackConfig] = {}
    for atk in selected_attacks:
        atk_id = atk.lower()
        params: Dict[str, Any] = {
            "norm": chosen_norm,
            "targeted": is_targeted,
        }

        if atk_id == "pgd":
            params["num_steps"] = chosen_steps
            params["steps"] = chosen_steps
            params["alpha"] = chosen_alpha
            params["step_size"] = chosen_alpha
            params["random_start"] = True
        elif atk_id == "cw":
            params["max_iterations"] = chosen_steps
            params["learning_rate"] = chosen_alpha
            params["confidence"] = 0.0
            params["initial_const"] = 1.0
        elif atk_id == "deepfool":
            params["max_iterations"] = chosen_steps
            params["overshoot"] = 0.02
        elif atk_id == "fab":
            params["n_iter"] = chosen_steps
            params["alpha_max"] = 0.1
        elif atk_id == "embedding_pgd":
            params["steps"] = chosen_steps
            params["alpha"] = chosen_alpha

        attack_configs[atk_id] = AttackConfig(
            epsilon=chosen_eps,
            clip_min=clip_min,
            clip_max=clip_max,
            params=params,
        )

    return attack_configs


# =====================================================================
# Visual Quality & Perturbation Metrics Calculator
# =====================================================================

def compute_perturbation_and_visual_metrics(
    clean_tensors: Union[torch.Tensor, np.ndarray],
    adv_tensors: Union[torch.Tensor, np.ndarray],
    domain: str = "image",
) -> Dict[str, Any]:
    """
    Compute rigorous perturbation distance norms and perceptual visual quality metrics.
    Includes L-infinity, L-2, L-0 sparsity, MAE, PSNR (dB), SSIM, and Visual Degradation Score.
    """
    if isinstance(clean_tensors, torch.Tensor):
        clean_np = clean_tensors.detach().cpu().float().numpy()
    else:
        clean_np = np.asarray(clean_tensors, dtype=np.float32)

    if isinstance(adv_tensors, torch.Tensor):
        adv_np = adv_tensors.detach().cpu().float().numpy()
    else:
        adv_np = np.asarray(adv_tensors, dtype=np.float32)

    batch_size = clean_np.shape[0]
    diff = adv_np - clean_np
    diff_flat = diff.reshape(batch_size, -1)

    # 1. Mathematical Perturbation Norms per Sample
    linf_samples = np.max(np.abs(diff_flat), axis=-1)
    l2_samples = np.linalg.norm(diff_flat, ord=2, axis=-1)
    l0_samples_count = np.sum(np.abs(diff_flat) > 1e-5, axis=-1)
    total_elements = diff_flat.shape[-1]
    l0_samples_pct = (l0_samples_count / total_elements * 100.0) if total_elements > 0 else np.zeros(batch_size)
    mae_samples = np.mean(np.abs(diff_flat), axis=-1)

    metrics: Dict[str, Any] = {
        "batch_size": batch_size,
        "linf_mean": float(np.mean(linf_samples)),
        "linf_max": float(np.max(linf_samples)),
        "linf_min": float(np.min(linf_samples)),
        "l2_mean": float(np.mean(l2_samples)),
        "l2_max": float(np.max(l2_samples)),
        "l2_min": float(np.min(l2_samples)),
        "l0_mean_count": float(np.mean(l0_samples_count)),
        "l0_mean_pct": float(np.mean(l0_samples_pct)),
        "mae_mean": float(np.mean(mae_samples)),
        "sample_metrics": [],
    }

    # 2. Visual Quality Metrics (for images with C, H, W or H, W, C)
    is_image = domain == "image" and clean_np.ndim == 4
    psnr_samples: List[float] = []
    ssim_samples: List[float] = []

    if is_image:
        # Normalize clean & adv into [0.0, 1.0] domain for standardized PSNR/SSIM calculation
        c_min, c_max = float(clean_np.min()), float(clean_np.max())
        val_range = max(1e-5, c_max - c_min)

        for i in range(batch_size):
            # Scale sample to [0, 1]
            c_sample = np.clip((clean_np[i] - c_min) / val_range, 0.0, 1.0)
            a_sample = np.clip((adv_np[i] - c_min) / val_range, 0.0, 1.0)

            # Reshape to (H, W, C) for SSIM
            if c_sample.shape[0] in (1, 3, 4):  # (C, H, W)
                c_img = np.transpose(c_sample, (1, 2, 0))
                a_img = np.transpose(a_sample, (1, 2, 0))
            else:
                c_img = c_sample
                a_img = a_sample

            # Compute PSNR
            mse_val = np.mean((c_img - a_img) ** 2)
            if mse_val < 1e-10:
                psnr_val = 100.0
            else:
                psnr_val = float(-10.0 * np.log10(mse_val))
            psnr_samples.append(psnr_val)

            # Compute SSIM
            if SKIMAGE_AVAILABLE:
                try:
                    ch_axis = -1 if c_img.ndim == 3 and c_img.shape[-1] in (1, 3, 4) else None
                    if c_img.shape[-1] == 1:
                        c_img_2d = c_img.squeeze(-1)
                        a_img_2d = a_img.squeeze(-1)
                        ssim_val = float(structural_similarity(c_img_2d, a_img_2d, data_range=1.0))
                    else:
                        ssim_val = float(structural_similarity(c_img, a_img, data_range=1.0, channel_axis=ch_axis))
                except Exception:
                    # Fallback structural approximation
                    ssim_val = float(max(0.0, 1.0 - np.sqrt(mse_val)))
            else:
                # Fast vectorized SSIM approximation
                ssim_val = float(max(0.0, 1.0 - np.sqrt(mse_val)))

            ssim_samples.append(ssim_val)

            metrics["sample_metrics"].append({
                "index": i,
                "linf": float(linf_samples[i]),
                "l2": float(l2_samples[i]),
                "l0_pct": float(l0_samples_pct[i]),
                "l0_count": int(l0_samples_count[i]),
                "mae": float(mae_samples[i]),
                "psnr": float(psnr_val),
                "ssim": float(ssim_val),
                "visual_degradation_pct": float(max(0.0, (1.0 - ssim_val) * 100.0)),
            })

        metrics["psnr_mean"] = float(np.mean(psnr_samples))
        metrics["psnr_min"] = float(np.min(psnr_samples))
        metrics["psnr_max"] = float(np.max(psnr_samples))
        metrics["ssim_mean"] = float(np.mean(ssim_samples))
        metrics["ssim_min"] = float(np.min(ssim_samples))
        metrics["ssim_max"] = float(np.max(ssim_samples))
        metrics["visual_degradation_pct_mean"] = float(np.mean([(1.0 - s) * 100.0 for s in ssim_samples]))
    else:
        for i in range(batch_size):
            metrics["sample_metrics"].append({
                "index": i,
                "linf": float(linf_samples[i]),
                "l2": float(l2_samples[i]),
                "l0_pct": float(l0_samples_pct[i]),
                "l0_count": int(l0_samples_count[i]),
                "mae": float(mae_samples[i]),
                "psnr": None,
                "ssim": None,
                "visual_degradation_pct": None,
            })
        metrics["psnr_mean"] = None
        metrics["ssim_mean"] = None
        metrics["visual_degradation_pct_mean"] = None

    return metrics


# =====================================================================
# Multi-Objective Scorecard & Tradeoff Index
# =====================================================================

def evaluate_multi_objective_tradeoff(
    asr_pct: float,
    pert_metrics: Dict[str, Any],
    domain: str = "image",
) -> Dict[str, Any]:
    """
    Score the multi-objective goal:
    - High ASR (aim for >= 80%)
    - Low L_inf (aim for <= 0.0314 ~ 8/255)
    - Low L_2 (aim for minimal Euclidean norm)
    - Low L_0 (aim for minimal elements altered)
    - Minimal degradation in visual quality (SSIM >= 0.95, PSNR >= 35 dB)
    """
    linf = pert_metrics.get("linf_mean", 0.0)
    l2 = pert_metrics.get("l2_mean", 0.0)
    l0_pct = pert_metrics.get("l0_mean_pct", 0.0)
    ssim = pert_metrics.get("ssim_mean")
    psnr = pert_metrics.get("psnr_mean")

    # Target checklist
    targets = {
        "high_asr": {
            "name": "High Attack Success Rate (ASR >= 80%)",
            "target": ">= 80.0%",
            "actual": f"{asr_pct:.1f}%",
            "passed": asr_pct >= 80.0,
            "weight": 0.35,
        },
        "low_linf": {
            "name": "Low L-infinity Norm (L_inf <= 0.0314 / 8/255)",
            "target": "<= 0.0314",
            "actual": f"{linf:.4f}",
            "passed": linf <= 0.0314,
            "weight": 0.20,
        },
        "low_l2": {
            "name": "Low L-2 Euclidean Norm (L_2 <= 2.5)",
            "target": "<= 2.5000",
            "actual": f"{l2:.4f}",
            "passed": l2 <= 2.5,
            "weight": 0.15,
        },
        "low_l0": {
            "name": "Sparse Element Modifications (L_0 <= 50% or low count)",
            "target": "Minimal",
            "actual": f"{l0_pct:.1f}% altered",
            "passed": l0_pct < 80.0 or linf <= 0.0314,
            "weight": 0.10,
        },
    }

    if ssim is not None and psnr is not None:
        targets["high_visual_ssim"] = {
            "name": "High Structural Similarity (SSIM >= 0.95)",
            "target": ">= 0.9500",
            "actual": f"{ssim:.4f}",
            "passed": ssim >= 0.95,
            "weight": 0.10,
        }
        targets["high_visual_psnr"] = {
            "name": "High Peak Signal-to-Noise Ratio (PSNR >= 35.0 dB)",
            "target": ">= 35.0 dB",
            "actual": f"{psnr:.1f} dB",
            "passed": psnr >= 35.0,
            "weight": 0.10,
        }

    # Calculate Stealth & Quality Tradeoff Index (SQTI) in [0, 100]
    # Balance between: ASR effectiveness (40%) + Visual Fidelity (30%) + Perturbation Stealth (30%)
    asr_component = min(100.0, asr_pct)
    if ssim is not None:
        visual_component = min(100.0, max(0.0, ssim * 100.0))
    else:
        visual_component = 100.0

    # Perturbation score: 100 when linf <= 0.01, 0 when linf >= 0.20
    pert_component = max(0.0, min(100.0, 100.0 - (linf / 0.15) * 100.0))

    sqti_score = (0.40 * asr_component) + (0.30 * visual_component) + (0.30 * pert_component)

    # Qualitative Verdict
    passed_cnt = sum(1 for t in targets.values() if t["passed"])
    total_cnt = len(targets)

    if sqti_score >= 82.0 and asr_pct >= 75.0 and (ssim is None or ssim >= 0.94):
        verdict = "OPTIMAL STEALTH ATTACK (High ASR with Imperceptible Perturbation)"
        verdict_badge = "🟢 OPTIMAL STEALTH & QUALITY"
        recommendation = (
            "Excellent multi-objective balance! The attack achieves high misclassification "
            "rate while keeping perturbation visually imperceptible."
        )
    elif asr_pct >= 80.0 and (ssim is not None and ssim < 0.92 or linf > 0.05):
        verdict = "EFFECTIVE BUT OVER-PERTURBED (High ASR with Noticeable Distortion)"
        verdict_badge = "🟡 OVER-PERTURBED"
        recommendation = (
            f"The attack successfully flips predictions ({asr_pct:.1f}%), but perturbation magnitude "
            f"(L_inf={linf:.4f}) degrades visual quality (SSIM={ssim:.3f} if image). "
            f"Recommendation: Lower epsilon by 30-50% and increase iteration steps (e.g. 20-30 steps) "
            f"with a smaller alpha step size to find closer decision boundary points."
        )
    elif asr_pct < 60.0 and (ssim is not None and ssim >= 0.97 and linf <= 0.02):
        verdict = "HIGH VISUAL QUALITY BUT LOW EFFICACY (Stealthy but Under-effective)"
        verdict_badge = "🟠 UNDER-EFFECTIVE"
        recommendation = (
            f"Perturbations are extremely subtle (L_inf={linf:.4f}, SSIM={ssim:.3f}), but attack success "
            f"rate is insufficient ({asr_pct:.1f}%). "
            f"Recommendation: Slightly increase epsilon budget (e.g. from {linf:.4f} to {linf*1.5:.4f}) "
            f"or switch to optimization-based attacks (C&W or DeepFool) to reach the nearest boundary."
        )
    else:
        verdict = "MODERATE TRADEOFF (Balanced Performance)"
        verdict_badge = "🔵 MODERATE TRADEOFF"
        recommendation = (
            f"Attack achieved {asr_pct:.1f}% ASR with L_inf={linf:.4f}. "
            f"Consider fine-tuning epsilon and step size alpha to maximize stealth score."
        )

    return {
        "sqti_score": round(sqti_score, 2),
        "verdict": verdict,
        "verdict_badge": verdict_badge,
        "passed_targets_count": passed_cnt,
        "total_targets_count": total_cnt,
        "targets": targets,
        "recommendation": recommendation,
    }


# =====================================================================
# Main Execution Pipeline
# =====================================================================

def main():
    total_start_time = time.time()
    args = parse_args()

    # Determine interactive mode
    is_interactive = args.interactive or (
        not args.no_interactive and args.domain is None and args.model is None and args.attacks is None
    )

    print("=" * 76)
    print("      ADVERSCAN — ADVERSARIAL ATTACK ENGINE & PERTURBATION EVALUATION       ")
    print("=" * 76)
    print("  Scope: Ingestion (M1) -> Baseline (M2) -> Attack (M3) -> Adv Eval -> Vulnerability (M5)")
    print("  Focus: High ASR • Low L_inf • Low L_2 • Low L_0 • Minimal Visual Degradation")
    print("=" * 76)

    # 1. Domain & Model Selection
    if is_interactive:
        selected_domain = prompt_domain_selection()
        model_info = prompt_model_selection(selected_domain)
        if selected_domain in ("all", "all domains", "5") or not selected_domain:
            selected_domain = model_info.get("data_domain") or "image"
        selected_dataset, sample_count = prompt_dataset_selection(selected_domain, model_info)
        selected_attacks = prompt_attack_selection(selected_domain)
        attack_configs = prompt_attack_constraints(selected_domain, selected_attacks, args)
    else:
        selected_domain = (args.domain or "image").lower().replace("_", "-")
        default_models = {
            "image": "bazyl/gtsrb-model",
            "text": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            "time-series": "time-series-ecg-classifier",
            "tabular": "tabular-mlp-adult-census",
        }
        model_input = args.model or default_models.get(selected_domain, "bazyl/gtsrb-model")

        preset = None
        if model_input in MODEL_PRESETS:
            preset = {"preset_key": model_input, **MODEL_PRESETS[model_input]}
        else:
            for k, v in MODEL_PRESETS.items():
                if model_input in (v.get("model_id"), v.get("name"), k):
                    preset = {"preset_key": k, **v}
                    break

        model_id = preset.get("model_id", model_input) if preset else model_input
        model_name = preset.get("name", Path(model_id).stem) if preset else Path(model_id).stem
        proc_name = preset.get("processor_name") if preset else model_id
        num_classes = preset.get("num_classes", 2) if preset else 2

        model_info = {
            "model_id": model_id,
            "model_name": model_name,
            "processor_name": proc_name,
            "num_classes": num_classes,
            "preset_info": preset or {},
        }

        default_ds = "stanfordnlp/sst2" if selected_domain == "text" else "bazyl/GTSRB"
        selected_dataset = args.dataset or (preset.get("dataset_name") if preset else None) or default_ds

        # Sample count parsing
        samples_str = str(args.samples).lower()
        if samples_str in ("0", "all", "full"):
            sample_count = 0
        elif samples_str == "half":
            total_s = get_dataset_total_samples(selected_domain, selected_dataset, model_id=model_id)
            sample_count = (total_s // 2) if total_s else 10
        else:
            try:
                sample_count = int(samples_str)
            except ValueError:
                sample_count = 10

        if args.attacks:
            if "all" in args.attacks:
                selected_attacks = [a["id"] for a in DOMAIN_ATTACKS_CATALOG.get(selected_domain, [])]
            else:
                selected_attacks = [a.lower() for a in args.attacks]
        else:
            selected_attacks = ["pgd"] if selected_domain == "image" else ["fgsm"]

        # Build attack configs from CLI args
        attack_configs = {}
        for atk in selected_attacks:
            atk_id = atk.lower()
            eps = args.epsilon if args.epsilon is not None else (0.03137 if selected_domain == "image" else 0.1)
            steps = args.steps or 20
            alpha = args.alpha or (2.0 * eps / steps)
            attack_configs[atk_id] = AttackConfig(
                epsilon=eps,
                clip_min=args.clip_min,
                clip_max=args.clip_max,
                params={
                    "num_steps": steps,
                    "steps": steps,
                    "alpha": alpha,
                    "step_size": alpha,
                    "norm": args.norm,
                    "targeted": args.targeted,
                    "random_start": True,
                },
            )

    # 2. Hardware Device Resolution
    dev_info = get_execution_device_info(args.device)
    device_str = dev_info["device_str"]
    print(f"\n[Hardware] Target Device: {device_str} (GPU Available: {dev_info['gpu_available']})")

    # Interactive confirmation
    if is_interactive:
        confirm = input("\n▶ Launch attack engine evaluation & analysis? [Y/n]: ").strip().lower()
        if confirm in ("n", "no"):
            print("  Evaluation aborted by user.")
            sys.exit(0)

    # 3. Dataset Loader Preparation
    print(f"\n[1/3] Preparing dataset loader for '{selected_dataset}' ({selected_domain.upper()})...")
    dataset_loader, sample_input = build_dataset_and_sample_input(
        domain=selected_domain,
        model_id=model_info["model_id"],
        dataset_name=selected_dataset,
        processor_name=model_info.get("processor_name"),
        samples=sample_count,
        batch_size=args.batch_size,
        preset_info=model_info.get("preset_info", {}),
    )

    # 4. Configure AdverScan Pipeline with mode="attack_assessment"
    print(f"[2/3] Configuring Attack Assessment Pipeline...")
    sample_cnt_val = sample_count if sample_count > 0 else None

    pipeline_config = PipelineConfig(
        model_path=model_info["model_id"],
        sample_input=sample_input,
        device=device_str,
        model_name=model_info["model_name"],
        num_classes=model_info.get("num_classes", 2),
        dataset_name=selected_dataset,
        processor_name=model_info.get("processor_name") or model_info["model_id"],
        mode="attack_assessment",
        attacks=selected_attacks,
        attack_configs=attack_configs,
        sample_count=sample_cnt_val,
        batch_size=args.batch_size,
        custom_dataset_loader=dataset_loader,
        enable_xai=False,
        enable_hardening=False,
        enable_retest=False,
        enable_report=False,
        output_dir=args.output_dir,
    )

    # 5. Execute Pipeline (M1 -> M2 -> M3 -> M2 -> M5)
    print(f"[3/3] Executing Attack Engine Evaluation across M1 -> M2 -> M3 -> M2 -> M5...\n")
    orchestrator = AdverScanOrchestrator()
    result = orchestrator.run(pipeline_config)

    total_pipeline_time = time.time() - total_start_time

    # =====================================================================
    # Display Comprehensive Evaluation Telemetry & Analysis
    # =====================================================================
    print("\n" + "=" * 76)
    print("      ADVERSCAN ATTACK ENGINE & PERTURBATION ASSESSMENT RESULTS         ")
    print("=" * 76)
    print(f"Overall Status:            {result.status}")
    print(f"Execution Mode:            ATTACK_ASSESSMENT (M1 Ingestion -> M5 Vulnerability)")
    print(f"Total Evaluation Time:     {total_pipeline_time:.2f}s")
    print(f"Target Modality / Domain:  {selected_domain.upper()}")
    print(f"Evaluated Model:           {model_info['model_name']} ({model_info['model_id']})")
    print(f"Evaluated Dataset:         {selected_dataset} ({sample_count if sample_count > 0 else 'Full'} samples)")

    # 1. Clean Baseline Results
    print("\n" + "-" * 76)
    print("M2 — CLEAN BASELINE EVALUATION")
    print("-" * 76)
    clean_acc = 0.0
    clean_conf = 0.0
    if result.baseline_evaluation:
        clean_acc = result.baseline_evaluation.get("accuracy", 0.0) * 100.0
        clean_conf = result.baseline_evaluation.get("average_confidence", 0.0) * 100.0
        clean_f1 = result.baseline_evaluation.get("f1_macro", 0.0) * 100.0
        print(f"  • Clean Accuracy:           {clean_acc:.2f}%")
        print(f"  • Clean Average Confidence: {clean_conf:.2f}%")
        print(f"  • Clean Macro F1-Score:     {clean_f1:.2f}%")
    else:
        print("  Clean baseline results unavailable.")

    # 2. Iterate each evaluated attack for deep perturbation & visual quality analysis
    raw_atk_results = getattr(result, "_raw_attack_results", {})
    raw_adv_evals = getattr(result, "_raw_adv_evaluations", {})
    raw_baseline = getattr(result, "_raw_baseline_result", None)

    evaluation_summary_export: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "domain": selected_domain,
        "model": model_info["model_id"],
        "dataset": selected_dataset,
        "samples_evaluated": sample_count,
        "clean_baseline": result.baseline_evaluation,
        "attacks": {},
    }

    for atk_name, atk_meta in result.attack_results.items():
        atk_key = atk_name.lower()
        print("\n" + "=" * 76)
        print(f"▶ ATTACK EVALUATION: [{atk_name.upper()}] ({atk_meta.get('attack_class')})")
        print("=" * 76)

        # Retrieve vulnerability scoring (M5)
        vuln_item = result.vulnerability_analysis.get(atk_key, {})
        assessment = vuln_item.get("assessment", {})
        scoring = vuln_item.get("scoring", {})

        adv_acc = assessment.get("adversarial_accuracy", 0.0) * 100.0
        acc_drop = assessment.get("accuracy_drop", 0.0) * 100.0
        adv_conf = assessment.get("adversarial_confidence", 0.0) * 100.0
        conf_drop = assessment.get("confidence_drop", 0.0) * 100.0
        asr = assessment.get("attack_success_rate")
        asr_pct = (asr * 100.0) if asr is not None else max(0.0, acc_drop)

        atk_time = atk_meta.get("execution_time_seconds", 0.0)
        v_score = scoring.get("vulnerability_score")
        risk = scoring.get("risk_level", "UNKNOWN").upper()

        print(f"  Execution Time:          {atk_time:.2f}s")
        print(f"  Adversarial Accuracy:    {clean_acc:.1f}% → {adv_acc:.1f}% (Accuracy Drop: -{acc_drop:.1f}%)")
        print(f"  Confidence Impact:       {clean_conf:.1f}% → {adv_conf:.1f}% (Confidence Drop: -{conf_drop:.1f}%)")
        print(f"  Attack Success Rate:     {asr_pct:.2f}%")
        print(f"  Risk Severity Rating:    {risk} (Vulnerability Score: {v_score:.1f} / 100)" if v_score else f"  Risk Severity: {risk}")

        # Extract Raw Tensors for Perturbation & Visual Quality Metrics
        raw_combined = raw_atk_results.get(atk_key)
        pert_metrics = {}
        if raw_combined and raw_combined.adversarial_examples is not None:
            clean_tensors = raw_combined.original_inputs
            adv_tensors = raw_combined.adversarial_examples
            pert_metrics = compute_perturbation_and_visual_metrics(
                clean_tensors=clean_tensors,
                adv_tensors=adv_tensors,
                domain=selected_domain,
            )
        else:
            # Fallback to M5 perturbation metrics
            m5_pert = assessment.get("perturbation", {})
            pert_metrics = {
                "linf_mean": m5_pert.get("linf_mean", 0.0) or 0.0,
                "l2_mean": m5_pert.get("l2_mean", 0.0) or 0.0,
                "l0_mean_pct": m5_pert.get("l0_mean", 0.0) or 0.0,
                "sample_metrics": [],
            }

        # Display Perturbation Norms
        print("\n  📏 Mathematical Perturbation Distortion Norms:")
        print(f"    • L-infinity (L_inf):   Mean: {pert_metrics.get('linf_mean', 0.0):.5f} | Max: {pert_metrics.get('linf_max', 0.0):.5f}")
        print(f"    • L-2 Euclidean (L_2):  Mean: {pert_metrics.get('l2_mean', 0.0):.4f}  | Max: {pert_metrics.get('l2_max', 0.0):.4f}")
        print(f"    • L-0 Sparsity (L_0):   Mean Altered Elements: {pert_metrics.get('l0_mean_pct', 0.0):.1f}%")
        if pert_metrics.get("mae_mean") is not None:
            print(f"    • Mean Absolute Error:  {pert_metrics.get('mae_mean'):.5f}")

        # Display Visual Quality Metrics (for images)
        if pert_metrics.get("ssim_mean") is not None:
            print("\n  👁️ Visual Quality & Perceptual Fidelity Metrics:")
            psnr_m = pert_metrics.get("psnr_mean", 0.0)
            ssim_m = pert_metrics.get("ssim_mean", 0.0)
            deg_m = pert_metrics.get("visual_degradation_pct_mean", 0.0)
            print(f"    • Peak Signal-to-Noise: {psnr_m:.2f} dB (Min: {pert_metrics.get('psnr_min', 0.0):.2f} dB)")
            print(f"    • Structural Similarity: {ssim_m:.4f} (Min: {pert_metrics.get('ssim_min', 0.0):.4f})")
            print(f"    • Visual Degradation:    {deg_m:.2f}%")

            # Perceptual assessment
            if ssim_m >= 0.98 and psnr_m >= 38.0:
                print(f"    • Perceptual Status:     EXCELLENT — Humanly imperceptible perturbation")
            elif ssim_m >= 0.94 and psnr_m >= 32.0:
                print(f"    • Perceptual Status:     VERY GOOD — Minimal degradation, subtle artifacting")
            elif ssim_m >= 0.88:
                print(f"    • Perceptual Status:     MODERATE — Detectable noise on close inspection")
            else:
                print(f"    • Perceptual Status:     HIGH DEGRADATION — Noticeable human-visible distortion")

        # Multi-Objective Goal Assessment
        tradeoff = evaluate_multi_objective_tradeoff(
            asr_pct=asr_pct,
            pert_metrics=pert_metrics,
            domain=selected_domain,
        )

        print("\n  🎯 MULTI-OBJECTIVE ATTACK OPTIMIZATION ASSESSMENT:")
        print(f"    • Stealth & Quality Index: {tradeoff['sqti_score']} / 100 [{tradeoff['verdict_badge']}]")
        print(f"    • Objective Targets Met:   {tradeoff['passed_targets_count']} of {tradeoff['total_targets_count']}")
        for t_key, t_info in tradeoff["targets"].items():
            status_icon = "✅" if t_info["passed"] else "❌"
            print(f"      {status_icon} {t_info['name']}: Actual={t_info['actual']} (Target: {t_info['target']})")

        print(f"\n  💡 Hyperparameter Tuning Recommendation:")
        print(f"    {tradeoff['recommendation']}")

        # Per-Sample Inspection Table
        sample_metrics = pert_metrics.get("sample_metrics", [])
        if sample_metrics:
            print("\n  🔬 Per-Sample Prediction Shift & Perturbation Telemetry:")
            adv_eval_obj = raw_adv_evals.get(atk_key)
            adv_preds = adv_eval_obj.predictions if adv_eval_obj else []
            adv_confs = (adv_eval_obj.sample_confidences or []) if adv_eval_obj else []
            clean_preds = raw_baseline.predictions if raw_baseline else []
            clean_confs = (raw_baseline.sample_confidences or []) if raw_baseline else []
            true_labels = raw_combined.labels if raw_combined else []
            if isinstance(true_labels, torch.Tensor):
                true_labels = true_labels.detach().cpu().tolist()

            header = (
                f"  {'Idx':<4} | {'True':<6} | {'Clean Pred (Conf)':<18} | {'Adv Pred (Conf)':<18} | "
                f"{'Status':<11} | {'L_inf':<7} | {'L_2':<7} | {'L_0 (%)':<7}"
            )
            if pert_metrics.get("ssim_mean") is not None:
                header += f" | {'PSNR (dB)':<9} | {'SSIM':<6}"

            print("-" * len(header))
            print(header)
            print("-" * len(header))

            max_display = min(15, len(sample_metrics))
            for i in range(max_display):
                sm = sample_metrics[i]
                y_t = str(true_labels[i]) if i < len(true_labels) else "?"
                c_conf = f" ({clean_confs[i]*100:.0f}%)" if i < len(clean_confs) and clean_confs[i] is not None else ""
                a_conf = f" ({adv_confs[i]*100:.0f}%)" if i < len(adv_confs) and adv_confs[i] is not None else ""
                c_p = f"{clean_preds[i]}{c_conf}" if i < len(clean_preds) else "?"
                a_p = f"{adv_preds[i]}{a_conf}" if i < len(adv_preds) else "?"

                # Determine flip status
                flipped = (i < len(clean_preds) and i < len(adv_preds) and clean_preds[i] != adv_preds[i])
                status_str = "FLIPPED ⚡" if flipped else "RESISTANT 🛡️"

                row = (
                    f"  {i:<4} | {y_t:<6} | {c_p:<18} | {a_p:<18} | "
                    f"{status_str:<11} | {sm['linf']:<7.4f} | {sm['l2']:<7.4f} | {sm['l0_pct']:<7.1f}"
                )
                if sm.get("psnr") is not None and sm.get("ssim") is not None:
                    row += f" | {sm['psnr']:<9.1f} | {sm['ssim']:<6.4f}"
                print(row)

            if len(sample_metrics) > max_display:
                print(f"  ... [and {len(sample_metrics) - max_display} additional samples evaluated]")
            print("-" * len(header))

        # Save Visual Comparison Images if requested
        if (args.save_visuals or (is_interactive and selected_domain == "image")) and selected_domain == "image":
            if raw_combined and raw_combined.adversarial_examples is not None:
                try:
                    visuals_dir = Path(args.output_dir) / "visuals" / atk_name.lower()
                    visuals_dir.mkdir(parents=True, exist_ok=True)
                    import matplotlib.pyplot as plt

                    c_np = raw_combined.original_inputs.detach().cpu().float().numpy()
                    a_np = raw_combined.adversarial_examples.detach().cpu().float().numpy()
                    c_min, c_max = float(c_np.min()), float(c_np.max())
                    v_range = max(1e-5, c_max - c_min)

                    num_to_save = min(5, c_np.shape[0])
                    for s_idx in range(num_to_save):
                        c_img = np.clip((c_np[s_idx] - c_min) / v_range, 0.0, 1.0)
                        a_img = np.clip((a_np[s_idx] - c_min) / v_range, 0.0, 1.0)
                        if c_img.shape[0] in (1, 3, 4):
                            c_img = np.transpose(c_img, (1, 2, 0))
                            a_img = np.transpose(a_img, (1, 2, 0))

                        diff_amp = np.clip(np.abs(a_img - c_img) * 10.0, 0.0, 1.0)

                        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
                        axes[0].imshow(c_img.squeeze() if c_img.shape[-1] == 1 else c_img)
                        axes[0].set_title(f"Clean (Pred: {clean_preds[s_idx] if s_idx < len(clean_preds) else '?'})")
                        axes[0].axis("off")

                        axes[1].imshow(a_img.squeeze() if a_img.shape[-1] == 1 else a_img)
                        axes[1].set_title(f"Adversarial (Pred: {adv_preds[s_idx] if s_idx < len(adv_preds) else '?'})")
                        axes[1].axis("off")

                        axes[2].imshow(diff_amp.squeeze() if diff_amp.shape[-1] == 1 else diff_amp)
                        axes[2].set_title("Perturbation (|Adv-Clean| x10)")
                        axes[2].axis("off")

                        plt.tight_layout()
                        save_p = visuals_dir / f"sample_{s_idx}_comparison.png"
                        plt.savefig(save_p, dpi=120)
                        plt.close(fig)

                    print(f"\n  📸 Saved {num_to_save} visual comparison figures to: '{visuals_dir}'")
                except Exception as vis_err:
                    print(f"  [Notice: Could not save visual figures: {vis_err}]")

        # Populate export summary
        evaluation_summary_export["attacks"][atk_name] = {
            "attack_class": atk_meta.get("attack_class"),
            "execution_time_seconds": atk_time,
            "adversarial_accuracy": adv_acc,
            "accuracy_drop": acc_drop,
            "adversarial_confidence": adv_conf,
            "confidence_drop": conf_drop,
            "attack_success_rate": asr_pct,
            "vulnerability_score": v_score,
            "risk_severity": risk,
            "perturbation_norms": {
                "linf_mean": pert_metrics.get("linf_mean"),
                "linf_max": pert_metrics.get("linf_max"),
                "l2_mean": pert_metrics.get("l2_mean"),
                "l2_max": pert_metrics.get("l2_max"),
                "l0_mean_pct": pert_metrics.get("l0_mean_pct"),
                "mae_mean": pert_metrics.get("mae_mean"),
            },
            "visual_quality": {
                "psnr_mean": pert_metrics.get("psnr_mean"),
                "ssim_mean": pert_metrics.get("ssim_mean"),
                "visual_degradation_pct_mean": pert_metrics.get("visual_degradation_pct_mean"),
            },
            "multi_objective_tradeoff": tradeoff,
        }

    # Export JSON Summary
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / f"attack_evaluation_{int(time.time())}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_summary_export, f, indent=2)

    print("\n" + "=" * 76)
    print(f"✔ Attack Engine Evaluation Complete! Report saved to: '{report_file}'")
    print("=" * 76)


if __name__ == "__main__":
    main()

