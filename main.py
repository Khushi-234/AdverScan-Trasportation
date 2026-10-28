#!/usr/bin/env python3
"""
AdverScan Transportation - Full Pipeline Demo (main.py)
========================================================
Runs the complete end-to-end adversarial robustness assessment pipeline:

  Step 1 | Model Ingestion         - Load & standardise the GTSRB ViT model
  Step 2 | Baseline Evaluation     - Clean accuracy metrics on GTSRB test set
  Step 3 | Adversarial Attack      - Run FGSM / PGD / DeepFool / CW / FAB
  Step 4 | Adversarial Evaluation  - Measure accuracy under attack
  Step 5 | Vulnerability Analysis  - Assess + score vulnerability
  Step 6 | Summary Report          - Print final results to terminal

Usage:
    python main.py [--samples N] [--attack ATTACK] [--epsilon E] [--device DEVICE]
    python main.py --interactive   # or simply `python main.py` in terminal for interactive mode
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from dotenv import load_dotenv

from app.ingestion import ingest_model
from app.evaluation import evaluate_baseline
from app.evaluation.dataset_loader import HFVisionDatasetLoader
from app.evaluation.metrics import MetricsCalculator
from app.evaluation.results import EvaluationResult
from app.attack_engine import AttackConfig, execute_attack, get_attack
from app.attack_engine.models import AttackResult
from app.vulnerability_analysis import VulnerabilityEngine, VulnerabilityScorer
from app.vulnerability_analysis.assessment_result import AssessmentResult
from app.vulnerability_analysis.scoring_result import ScoringResult

load_dotenv()

MODEL_ID      = "bazyl/gtsrb-model"
DATASET_NAME  = "bazyl/GTSRB"
NUM_CLASSES   = 43
SEPARATOR     = "=" * 70
SUPPORTED_ATTACKS = ["fgsm", "pgd", "deepfool", "cw", "fab"]
ATTACK_CHOICES = SUPPORTED_ATTACKS + ["all"]


def banner():
    print(SEPARATOR)
    print("  AdverScan - Transportation AI Adversarial Robustness System")
    print("  Full Pipeline Demo  |  GTSRB Traffic Sign Classification")
    print(SEPARATOR)


def detect_system_device() -> Tuple[str, str]:
    """Auto-detect whether CUDA acceleration is available on the system."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return "cuda", f"CUDA (GPU: {gpu_name})"
    return "cpu", "CPU"


def resolve_device(requested: Optional[str] = None) -> str:
    """Resolve hardware device with automatic fallback."""
    detected_device, _ = detect_system_device()
    if not requested or requested.strip().lower() in ("auto", "none", ""):
        return detected_device
    req = requested.strip().lower()
    if req.startswith("cuda"):
        if torch.cuda.is_available():
            return "cuda"
        print("  WARNING: CUDA requested but unavailable - falling back to CPU.")
        return "cpu"
    if req == "cpu":
        return "cpu"
    return detected_device


def prompt_interactive_config(
    default_samples: int = 10,
    default_attack: str = "fgsm",
    default_epsilon: float = 8 / 255,
    default_device: Optional[str] = None,
) -> Tuple[str, str, int, float]:
    """Interactively prompt user for pipeline options."""
    detected_dev, dev_desc = detect_system_device()
    initial_device = default_device or detected_dev

    print("\n" + SEPARATOR)
    print("  INTERACTIVE PIPELINE CONFIGURATION")
    print(SEPARATOR)

    # 1. Device Auto-Detection
    print(f"\n[1/4] Device Auto-Detection:")
    print(f"      Detected Hardware: {dev_desc}")
    dev_in = input(f"      Use '{initial_device}'? [Y/n/cuda/cpu] (default: {initial_device}): ").strip().lower()
    if dev_in in ("n", "no"):
        selected_device = "cpu" if initial_device == "cuda" else "cuda"
    elif dev_in in ("cuda", "cpu"):
        selected_device = dev_in
    else:
        selected_device = initial_device
    selected_device = resolve_device(selected_device)

    # 2. Attack Selection
    print(f"\n[2/4] Adversarial Attack Selection:")
    attack_descriptions = [
        ("fgsm", "FGSM", "Fast Gradient Sign Method (Fast, 1-step gradient attack)"),
        ("pgd", "PGD", "Projected Gradient Descent (Iterative multi-step gradient attack)"),
        ("deepfool", "DeepFool", "DeepFool (Minimal perturbation boundary attack)"),
        ("cw", "CW", "Carlini & Wagner (Optimization-based L2/Linf attack)"),
        ("fab", "FAB", "Fast Adaptive Boundary (Adaptive step boundary attack)"),
        ("all", "ALL", "Run all supported adversarial attacks in sequence"),
    ]
    for idx, (_, name, desc) in enumerate(attack_descriptions, 1):
        print(f"      [{idx}] {name:<10} - {desc}")

    while True:
        atk_in = input(f"      Select attack [1-6 or name] (default: 1 [{default_attack.upper()}]): ").strip().lower()
        if not atk_in:
            selected_attack = default_attack
            break
        atk_map = {"1": "fgsm", "2": "pgd", "3": "deepfool", "4": "cw", "5": "fab", "6": "all"}
        if atk_in in atk_map:
            selected_attack = atk_map[atk_in]
            break
        elif atk_in in ATTACK_CHOICES:
            selected_attack = atk_in
            break
        print("      Invalid attack choice. Enter 1-6 or an attack name (or 'all').")

    # 3. Number of Samples
    print(f"\n[3/4] Evaluation Samples:")
    while True:
        samples_in = input(f"      Enter number of test samples [1-12630] (default: {default_samples}): ").strip()
        if not samples_in:
            selected_samples = default_samples
            break
        try:
            val = int(samples_in)
            if val <= 0:
                print("      Please enter a positive integer.")
                continue
            selected_samples = val
            break
        except ValueError:
            print("      Invalid input. Please enter a valid integer.")

    # 4. Epsilon Budget
    print(f"\n[4/4] Perturbation Budget (Epsilon):")
    while True:
        eps_in = input(f"      Enter perturbation epsilon (default: {default_epsilon:.5f} [~8/255]): ").strip()
        if not eps_in:
            selected_epsilon = default_epsilon
            break
        try:
            val = float(eps_in)
            if val < 0:
                print("      Epsilon must be non-negative.")
                continue
            selected_epsilon = val
            break
        except ValueError:
            print("      Invalid input. Please enter a float number.")

    print(f"\n  Pipeline Configuration Confirmed:")
    print(f"  Device: {selected_device}  |  Attack: {selected_attack.upper()}  |  Samples: {selected_samples}  |  Epsilon: {selected_epsilon:.5f}")
    print(SEPARATOR + "\n")

    return selected_device, selected_attack, selected_samples, selected_epsilon


def parse_args():
    parser = argparse.ArgumentParser(
        description="AdverScan full pipeline demo for GTSRB model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--samples",         type=int,   default=10,       help="Number of GTSRB test samples")
    parser.add_argument("--attack",          type=str,   default="fgsm",   choices=ATTACK_CHOICES, help="Adversarial attack (or 'all' to evaluate all attacks)")
    parser.add_argument("--epsilon",         type=float, default=8/255,    help="Perturbation budget epsilon")
    parser.add_argument("--batch-size",      type=int,   default=32,       help="Batch size")
    parser.add_argument("--device",          type=str,   default=None,     help="'cpu', 'cuda', or 'auto'")
    parser.add_argument("--output-dir",      type=str,   default="results",help="Root results directory")
    parser.add_argument("--interactive", "-i", action="store_true",        help="Force interactive prompt mode")
    parser.add_argument("--non-interactive", action="store_true",          help="Disable interactive prompt mode")
    return parser.parse_args()


def section(title):
    print(f"\n{SEPARATOR}\n  {title}\n{SEPARATOR}")


# -- STEP 1: Model Ingestion ----------------------------------------------------
def step1_ingest_model(device):
    section("STEP 1 - Model Ingestion (M1)")
    print(f"  Model:   {MODEL_ID}")
    print(f"  Device:  {device}\n")
    t0 = time.time()
    try:
        from transformers import AutoModelForImageClassification
        raw_model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"  ERROR: Could not load model from HuggingFace Hub: {e}")
        print("  Make sure you have internet access and 'transformers' installed.")
        sys.exit(1)
    sample_input = torch.randn(1, 3, 224, 224)
    adapter, metadata = ingest_model(
        model_path=raw_model,
        sample_input=sample_input,
        model_name="GTSRB ViT",
        device=device,
    )
    metadata.model_architecture = raw_model.__class__.__name__
    metadata.framework = "huggingface"
    if not metadata.total_parameters and hasattr(raw_model, "parameters"):
        try:
            metadata.total_parameters = sum(p.numel() for p in raw_model.parameters())
            metadata.trainable_parameters = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        except Exception:
            pass

    print(f"  Ingestion complete in {time.time() - t0:.1f}s")
    print(f"  Framework:   {metadata.framework}")
    params = getattr(metadata, "total_parameters", None)
    if params:
        print(f"  Parameters:  {params:,} (~{params / 1e6:.1f}M)")
    print(f"  Device:      {adapter.device}")
    return adapter, metadata


# -- STEP 2: Baseline Evaluation -----------------------------------------------
def step2_baseline_evaluation(adapter, samples, batch_size, output_dir):
    section("STEP 2 - Baseline Evaluation (M2)")
    split_str = f"test[:{samples}]" if samples > 0 else "test"
    print(f"  Dataset: {DATASET_NAME}  split={split_str}  batch_size={batch_size}\n")
    t0 = time.time()
    result = evaluate_baseline(
        adapter=adapter,
        dataset_name=DATASET_NAME,
        processor_name=MODEL_ID,
        split=split_str,
        batch_size=batch_size,
        num_classes=NUM_CLASSES,
        model_name="GTSRB_ViT",
        output_dir=str(output_dir / "baseline"),
        log_mlflow=False,
    )
    print(f"  Baseline complete in {time.time() - t0:.1f}s")
    print(f"  Clean Accuracy:       {result.accuracy * 100:.2f}%")
    print(f"  Macro F1:             {result.f1_macro * 100:.2f}%")
    print(f"  Avg Confidence:       {result.average_confidence * 100:.2f}%")
    print(f"  Avg Entropy:          {result.average_entropy:.4f} bits")
    print(f"  Samples evaluated:    {result.num_samples:,}")
    return result


# -- STEP 3+4: Adversarial Attack + Evaluation ---------------------------------
def step3_and_4_attack_and_evaluate(adapter, attack_name, epsilon, samples, batch_size, device, output_dir, loader=None):
    section(f"STEP 3 - Adversarial Attack ({attack_name.upper()}, epsilon={epsilon:.5f})")
    split_str = f"test[:{samples}]" if samples > 0 else "test"
    if loader is None:
        loader = HFVisionDatasetLoader(
            dataset_name=DATASET_NAME,
            processor_name=MODEL_ID,
            split=split_str,
            batch_size=batch_size,
        )
    total_samples = len(loader)
    print(f"  Loaded {total_samples} samples. Running {attack_name.upper()} attack...\n")

    attack_config = AttackConfig(epsilon=epsilon, clip_min=0.0, clip_max=1.0)
    attack_cls    = get_attack(attack_name.lower())

    all_targets, clean_preds, adv_preds = [], [], []
    clean_probs_list, adv_probs_list    = [], []
    all_adv_results                     = []
    processed = 0
    t0 = time.time()

    for batch_pixels, batch_targets, _ in loader.iterate_batches():
        batch_pixels  = batch_pixels.to(device)
        batch_targets = batch_targets.to(device)

        with torch.no_grad():
            c_out = adapter.predict(batch_pixels)
            if hasattr(c_out, "logits"):
                c_out = c_out.logits
            c_logits = c_out[:, :NUM_CLASSES] if c_out.shape[-1] > NUM_CLASSES else c_out
            c_probs  = torch.softmax(c_logits, dim=-1)
            c_preds  = torch.argmax(c_probs, dim=-1)

        adv_result = execute_attack(
            model=adapter, attack_cls=attack_cls,
            inputs=batch_pixels, labels=batch_targets, config=attack_config,
        )
        all_adv_results.append(adv_result)

        with torch.no_grad():
            a_out = adapter.predict(adv_result.adversarial_examples)
            if hasattr(a_out, "logits"):
                a_out = a_out.logits
            a_logits = a_out[:, :NUM_CLASSES] if a_out.shape[-1] > NUM_CLASSES else a_out
            a_probs  = torch.softmax(a_logits, dim=-1)
            a_preds  = torch.argmax(a_probs, dim=-1)

        all_targets.extend(batch_targets.cpu().numpy().tolist())
        clean_preds.extend(c_preds.cpu().numpy().tolist())
        adv_preds.extend(a_preds.cpu().numpy().tolist())
        clean_probs_list.append(c_probs.cpu().numpy())
        adv_probs_list.append(a_probs.cpu().numpy())
        processed += len(batch_targets)
        print(f"  Progress: {processed}/{total_samples}", end="\r", flush=True)

    print()
    print(f"  Attack complete in {time.time() - t0:.1f}s")

    section(f"STEP 4 - Adversarial Evaluation Metrics ({attack_name.upper()})")
    y_true        = np.array(all_targets,  dtype=np.int64)
    y_clean_pred  = np.array(clean_preds,  dtype=np.int64)
    y_adv_pred    = np.array(adv_preds,    dtype=np.int64)
    y_clean_probs = np.concatenate(clean_probs_list, axis=0)
    y_adv_probs   = np.concatenate(adv_probs_list,   axis=0)

    clean_m = MetricsCalculator.compute_metrics(y_true=y_true, y_pred=y_clean_pred, y_probs=y_clean_probs, num_classes=NUM_CLASSES)
    adv_m   = MetricsCalculator.compute_metrics(y_true=y_true, y_pred=y_adv_pred,   y_probs=y_adv_probs,   num_classes=NUM_CLASSES)

    print(f"  Clean  Accuracy:   {clean_m['accuracy'] * 100:.2f}%")
    print(f"  Adv    Accuracy:   {adv_m['accuracy']   * 100:.2f}%")
    print(f"  Accuracy Drop:     {(clean_m['accuracy'] - adv_m['accuracy']) * 100:.2f}%")

    adv_out_dir = output_dir / "adversarial"
    adv_out_dir.mkdir(parents=True, exist_ok=True)
    out_file = adv_out_dir / f"adversarial_{attack_name}_eps{epsilon:.4f}_{total_samples}samples.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL_ID, "dataset": DATASET_NAME, "split": split_str,
            "attack": attack_name, "epsilon": epsilon, "num_samples": total_samples,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "clean_accuracy": clean_m["accuracy"], "adversarial_accuracy": adv_m["accuracy"],
            "accuracy_drop": clean_m["accuracy"] - adv_m["accuracy"],
            "clean_f1_macro": clean_m["f1_macro"], "adversarial_f1_macro": adv_m["f1_macro"],
            "clean_average_confidence": clean_m["average_confidence"],
            "adversarial_average_confidence": adv_m["average_confidence"],
        }, f, indent=2)
    print(f"\n  Results saved -> {out_file}")

    if all_adv_results:
        adv_tensors = [r.adversarial_examples for r in all_adv_results if isinstance(r.adversarial_examples, torch.Tensor)]
        combined_adv = torch.cat(adv_tensors, dim=0) if adv_tensors else all_adv_results[-1].adversarial_examples

        orig_tensors = [r.original_inputs for r in all_adv_results if isinstance(r.original_inputs, torch.Tensor)]
        combined_orig = torch.cat(orig_tensors, dim=0) if orig_tensors else None

        label_tensors = [r.labels for r in all_adv_results if isinstance(r.labels, torch.Tensor)]
        combined_lbl = torch.cat(label_tensors, dim=0) if label_tensors else None

        final_attack_result = AttackResult(
            adversarial_examples=combined_adv,
            metadata=all_adv_results[-1].metadata,
            original_inputs=combined_orig,
            labels=combined_lbl,
        )
    else:
        final_attack_result = None

    return final_attack_result, clean_m, adv_m, y_true, y_clean_pred, y_adv_pred, total_samples


# -- STEP 5: Vulnerability Analysis --------------------------------------------
def step5_vulnerability_analysis(baseline_result, attack_result, clean_m, adv_m,
                                  y_true, y_clean_pred, y_adv_pred, num_samples, attack_name, output_dir):
    section(f"STEP 5 - Vulnerability Analysis ({attack_name.upper()})")
    adv_eval = EvaluationResult(
        dataset_name=DATASET_NAME, model_name="GTSRB_ViT",
        num_samples=num_samples, num_classes=NUM_CLASSES,
        accuracy=adv_m["accuracy"],
        precision_macro=adv_m["precision_macro"], recall_macro=adv_m["recall_macro"],
        f1_macro=adv_m["f1_macro"],
        precision_weighted=adv_m["precision_weighted"], recall_weighted=adv_m["recall_weighted"],
        f1_weighted=adv_m["f1_weighted"],
        average_confidence=adv_m["average_confidence"], average_entropy=adv_m["average_entropy"],
        per_class_metrics=adv_m["per_class_metrics"], confusion_matrix=adv_m["confusion_matrix"],
        batch_size=32, device=str(getattr(attack_result, "device", "cpu")),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    engine   = VulnerabilityEngine()
    analysis = engine.analyze(
        baseline_result=baseline_result, attack_result=attack_result, adversarial_result=adv_eval,
        y_true=y_true, y_clean_pred=y_clean_pred, y_adv_pred=y_adv_pred,
    )
    assessment: AssessmentResult = analysis["assessment"]
    scoring:    ScoringResult    = analysis["scoring"]

    print(f"  Accuracy Drop:       {assessment.accuracy_drop * 100:.2f}%")
    print(f"  F1 Drop:             {assessment.f1_drop * 100:.2f}%")
    print(f"  Confidence Drop:     {assessment.confidence_drop * 100:.2f}%")
    print(f"  Model Degradation:   {assessment.model_degradation * 100:.2f}%")
    asr_str = f"{assessment.attack_success_rate * 100:.2f}%" if assessment.attack_success_rate is not None else "N/A"
    print(f"  Attack Success Rate: {asr_str}")
    pert = assessment.perturbation or {}
    if pert:
        print(f"  Perturbation L-inf:  {pert.get('linf_mean', 'N/A')}")
    print(f"\n  Vulnerability Score: {scoring.vulnerability_score:.2f} / 100")
    print(f"  Risk Level:          {scoring.risk_level}")

    vuln_out_dir = output_dir / "vulnerability"
    vuln_out_dir.mkdir(parents=True, exist_ok=True)
    out_file = vuln_out_dir / f"vulnerability_{attack_name}_{num_samples}samples.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "attack_name": assessment.attack_name, "dataset_name": assessment.dataset_name,
            "num_samples": assessment.num_samples, "timestamp": assessment.timestamp,
            "clean_accuracy": assessment.clean_accuracy, "adversarial_accuracy": assessment.adversarial_accuracy,
            "accuracy_drop": assessment.accuracy_drop, "clean_f1": assessment.clean_f1,
            "adversarial_f1": assessment.adversarial_f1, "f1_drop": assessment.f1_drop,
            "confidence_drop": assessment.confidence_drop, "model_degradation": assessment.model_degradation,
            "attack_success_rate": assessment.attack_success_rate,
            "attack_success_rate_reason": assessment.attack_success_rate_reason,
            "perturbation": assessment.perturbation,
            "vulnerability_score": scoring.vulnerability_score, "risk_level": scoring.risk_level,
            "sub_scores": scoring.sub_scores,
        }, f, indent=2, default=str)
    print(f"\n  Vulnerability report saved -> {out_file}")
    return assessment, scoring


# -- STEP 6: Final Summary -----------------------------------------------------
def step6_final_summary(
    baseline,
    attack_eval_results,
    attacks_to_run,
    metadata,
    device,
    samples,
    batch_size,
    pipeline_errors,
    total_pipeline_runtime,
    output_dir,
):
    LINE_LEN = 70
    SEP_EQUAL = "=" * LINE_LEN
    SEP_DASH = "-" * LINE_LEN

    # 1. Overall Status
    if len(pipeline_errors) == 0 and len(attack_eval_results) == len(attacks_to_run):
        overall_status = "SUCCESS"
    elif len(attack_eval_results) > 0:
        overall_status = "PARTIAL_SUCCESS"
    else:
        overall_status = "FAILED"

    # Ingestion metadata details
    total_params = getattr(metadata, "total_parameters", None)
    trainable_params = getattr(metadata, "trainable_parameters", None)

    total_params_str = f"{total_params:,} (~{total_params / 1e6:.2f}M)" if total_params else "N/A"
    if trainable_params is not None and total_params:
        trainable_pct = (trainable_params / total_params) * 100.0
        trainable_str = f"{trainable_params:,} ({trainable_pct:.1f}%)"
    elif trainable_params is not None:
        trainable_str = f"{trainable_params:,}"
    else:
        trainable_str = "N/A"

    if total_params:
        size_mb = (total_params * 4) / (1024 * 1024)
        size_str = f"~{size_mb:.1f} MB (FP32)"
    else:
        size_str = "N/A"

    model_arch = getattr(metadata, "model_architecture", None) or "ViTForImageClassification"
    input_shape_str = str(getattr(metadata, "input_shape", (1, 3, 224, 224)))
    num_classes = getattr(metadata, "num_classes", None) or NUM_CLASSES
    task_type = getattr(metadata, "task_type", None) or "classification"
    framework = getattr(metadata, "framework", None) or "huggingface"
    if framework == "pytorch":
        framework = "huggingface"

    model_id_display = f"GTSRB ViT ({MODEL_ID})"

    print(f"\n{SEP_EQUAL}")
    print("             ADVERSCAN SECURITY ASSESSMENT RESULTS SUMMARY          ")
    print(SEP_EQUAL)
    print(f"Overall Status:            {overall_status}")
    print(f"Execution Mode:            FULL")
    print(f"Total Pipeline Runtime:    {total_pipeline_runtime:.2f}s")
    print(f"Target Modality / Domain:  IMAGE")
    print(f"Evaluated Model:           {model_id_display}")
    print(f"Evaluated Dataset:         {DATASET_NAME}")

    # PIPELINE ERRORS & DIAGNOSTICS
    if pipeline_errors:
        print(f"\n{SEP_DASH}")
        print("PIPELINE ERRORS & DIAGNOSTICS")
        print(SEP_DASH)
        for err in pipeline_errors:
            print(f"  ❌ [{err['module']}]: {err['error']}")

    # M1 — MODEL SPECIFICATIONS & PARAMETERS
    print(f"\n{SEP_DASH}")
    print("M1 — MODEL SPECIFICATIONS & PARAMETERS")
    print(SEP_DASH)
    print(f"  • Model Identifier:        {model_id_display}")
    print(f"  • Architecture:            {model_arch}")
    print(f"  • Total Parameters:        {total_params_str}")
    print(f"  • Trainable Parameters:    {trainable_str}")
    print(f"  • Estimated Model Size:    {size_str}")
    print(f"  • Input Shape:             {input_shape_str}")
    print(f"  • Total Classes:           {num_classes}")
    print(f"  • Task Type:               {task_type}")
    print(f"  • Ingestion Framework:     {framework}")
    print(f"  • Execution Device:        {device}")

    # DATASET & EVALUATION DETAILS
    print(f"\n{SEP_DASH}")
    print("DATASET & EVALUATION DETAILS")
    print(SEP_DASH)
    print(f"  • Dataset Identifier:      {DATASET_NAME}")
    print(f"  • Target Domain:           IMAGE")
    print(f"  • Evaluated Samples:       {samples}")
    print(f"  • Evaluation Batch Size:   {batch_size}")

    # M2 — CLEAN BASELINE EVALUATION METRICS
    print(f"\n{SEP_DASH}")
    print("M2 — CLEAN BASELINE EVALUATION METRICS")
    print(SEP_DASH)
    print(f"  • Clean Accuracy:           {baseline.accuracy * 100:.2f}%")
    print(f"  • Clean Average Confidence: {baseline.average_confidence * 100:.2f}%")
    print(f"  • Macro F1-Score:           {baseline.f1_macro * 100:.2f}%")

    # M3 — ADVERSARIAL ATTACK ENGINE RESULTS
    total_atk_time = sum(res["execution_time"] for res in attack_eval_results)
    print(f"\n{SEP_DASH}")
    print("M3 — ADVERSARIAL ATTACK ENGINE RESULTS")
    print(SEP_DASH)
    print(f"Total Attack Execution Time: {total_atk_time:.2f}s across {len(attack_eval_results)} attack(s)")
    for res in attack_eval_results:
        print(f"\n  ▶ Attack: [{res['attack_name'].upper()}]")
        print(f"    • Attack Class:          {res['attack_class']}")
        print(f"    • Execution Time:        {res['execution_time']:.2f}s")

    # M5 — VULNERABILITY ANALYSIS & SECURITY RISK SCORING
    print(f"\n{SEP_DASH}")
    print("M5 — VULNERABILITY ANALYSIS & SECURITY RISK SCORING")
    print(SEP_DASH)

    if attack_eval_results:
        # Determine driving threat (highest vulnerability score)
        driving_result = max(attack_eval_results, key=lambda r: r["scoring"].vulnerability_score)
        unified_vuln_score = driving_result["scoring"].vulnerability_score
        unified_robust_score = max(0.0, min(100.0, 100.0 - unified_vuln_score))
        unified_risk = getattr(driving_result["scoring"], "risk_level", None) or VulnerabilityScorer.determine_risk_level(unified_vuln_score)
        primary_threat = driving_result["attack_name"].upper()

        print("  ★ MODEL-LEVEL UNIFIED SECURITY ASSESSMENT:")
        print(f"    • Model Vulnerability Score: {unified_vuln_score:.2f} / 100")
        print(f"    • Model Robustness Score:    {unified_robust_score:.2f} / 100")
        print(f"    • Unified Security Risk:     {unified_risk}")
        print(f"    • Primary Driving Threat:    {primary_threat}")
        print(f"    • Conclusive Evaluations:    {len(attack_eval_results)} / {len(attacks_to_run)}")
        print(SEP_DASH)

        for res in attack_eval_results:
            assessment = res["assessment"]
            scoring = res["scoring"]
            pert = assessment.perturbation or {}

            asr_val = assessment.attack_success_rate
            asr_str = f"{asr_val * 100:.2f}%" if asr_val is not None else "N/A"

            # Efficacy
            linf_val = pert.get("linf_mean", 0.0)
            if (asr_val and asr_val > 0) or assessment.accuracy_drop > 0 or linf_val > 0:
                efficacy_str = "Effective (Perturbations Applied)"
            else:
                efficacy_str = "Ineffective (No Significant Impact)"

            # Perturbation norms
            linf_s = f"{pert.get('linf_mean', 0.0):.4f}"
            l2_s = f"{pert.get('l2_mean', 0.0):.4f}"
            l0_v = pert.get('l0_mean', 0.0)
            l0_s = f"{l0_v * 100:.2f}%" if l0_v <= 1.0 else f"{l0_v:.2f}%"
            pert_str = f"L_inf: {linf_s}, L_2: {l2_s}, L_0: {l0_s}"

            # Score components
            sub = scoring.sub_scores
            sub_asr = f"{sub.get('asr_score', 0.0):.1f}" if sub.get('asr_score') is not None else "N/A"
            sub_deg = f"{sub.get('degradation_score', 0.0):.1f}"
            sub_stl = f"{sub.get('stealth_score', 0.0):.1f}"
            score_comp_str = f"ASR={sub_asr}, Degradation={sub_deg}, Stealth={sub_stl}"

            clean_acc_str = f"{assessment.clean_accuracy * 100:.2f}%"
            adv_acc_str = f"{assessment.adversarial_accuracy * 100:.2f}%"
            acc_drop_str = f"-{assessment.accuracy_drop * 100:.2f}%"

            clean_conf_str = f"{assessment.clean_confidence * 100:.2f}%"
            adv_conf_str = f"{assessment.adversarial_confidence * 100:.2f}%"
            conf_drop_str = f"-{assessment.confidence_drop * 100:.2f}%"

            clean_f1_str = f"{assessment.clean_f1 * 100:.2f}%"
            adv_f1_str = f"{assessment.adversarial_f1 * 100:.2f}%"
            f1_drop_str = f"-{assessment.f1_drop * 100:.2f}%"

            print(f"\n  ▶ Attack: [{res['attack_name'].upper()}]")
            print(f"    • Risk Severity Level:   {scoring.risk_level}")
            print(f"    • Vulnerability Score:   {scoring.vulnerability_score:.2f} / 100")
            print(f"    • Attack Success Rate:   {asr_str}")
            print(f"    • Attack Efficacy:       {efficacy_str}")
            print(f"    • Attack Execution Time: {res['execution_time']:.2f}s")
            print(f"    • Accuracy Impact:       {clean_acc_str} → {adv_acc_str} (Drop: {acc_drop_str})")
            print(f"    • Confidence Impact:     {clean_conf_str} → {adv_conf_str} (Drop: {conf_drop_str})")
            print(f"    • Macro F1 Impact:       {clean_f1_str} → {adv_f1_str} (Drop: {f1_drop_str})")
            print(f"    • Model Degradation:     {assessment.model_degradation * 100:.2f}%")
            print(f"    • Perturbation Norms:    {pert_str}")
            print(f"    • Score Components:      {score_comp_str}")

    print(SEP_EQUAL)

    # Save unified summary JSON
    summary_path = output_dir / f"unified_assessment_{samples}samples.json"
    summary_payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overall_status": overall_status,
        "runtime_seconds": total_pipeline_runtime,
        "model_id": MODEL_ID,
        "dataset_name": DATASET_NAME,
        "num_samples": samples,
        "clean_baseline": {
            "accuracy": baseline.accuracy,
            "f1_macro": baseline.f1_macro,
            "average_confidence": baseline.average_confidence,
        },
        "attacks": [
            {
                "attack_name": r["attack_name"],
                "attack_class": r["attack_class"],
                "execution_time": r["execution_time"],
                "vulnerability_score": r["scoring"].vulnerability_score,
                "risk_level": r["scoring"].risk_level,
                "sub_scores": r["scoring"].sub_scores,
                "accuracy_drop": r["assessment"].accuracy_drop,
                "attack_success_rate": r["assessment"].attack_success_rate,
            }
            for r in attack_eval_results
        ],
        "unified_assessment": {
            "vulnerability_score": unified_vuln_score if attack_eval_results else 0.0,
            "robustness_score": unified_robust_score if attack_eval_results else 100.0,
            "risk_level": unified_risk if attack_eval_results else "LOW",
            "primary_driving_threat": primary_threat if attack_eval_results else "NONE",
            "conclusive_evaluations": f"{len(attack_eval_results)} / {len(attacks_to_run)}",
        },
        "errors": pipeline_errors,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"  Summary assessment report saved -> {summary_path}")


# -- MAIN ----------------------------------------------------------------------
def main():
    pipeline_start_time = time.time()
    banner()
    args = parse_args()

    # Determine whether interactive mode should be triggered
    is_tty = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
    explicit_cli = any(
        arg.startswith(("--samples", "--attack", "--epsilon", "--device"))
        for arg in sys.argv[1:]
    )
    should_prompt = args.interactive or (not args.non_interactive and not explicit_cli and is_tty)

    if should_prompt:
        device, attack, samples, epsilon = prompt_interactive_config(
            default_samples=args.samples,
            default_attack=args.attack,
            default_epsilon=args.epsilon,
            default_device=args.device,
        )
    else:
        device  = resolve_device(args.device)
        attack  = args.attack
        samples = args.samples
        epsilon = args.epsilon

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Active Execution Settings:")
    print(f"  Device: {device}  |  Samples: {samples}  |  Attack: {attack.upper()}  |  Epsilon: {epsilon:.5f}\n")

    pipeline_errors = []

    # Step 1: Model Ingestion
    adapter, metadata = step1_ingest_model(device)

    # Step 2: Baseline Evaluation
    baseline_result = step2_baseline_evaluation(
        adapter=adapter, samples=samples,
        batch_size=args.batch_size, output_dir=output_dir,
    )

    # Determine attacks to evaluate
    if attack.lower() == "all":
        attacks_to_run = list(SUPPORTED_ATTACKS)
    else:
        attacks_to_run = [attack.lower()]

    split_str = f"test[:{samples}]" if samples > 0 else "test"
    loader = HFVisionDatasetLoader(
        dataset_name=DATASET_NAME,
        processor_name=MODEL_ID,
        split=split_str,
        batch_size=args.batch_size,
    )

    attack_eval_results = []
    total_samples = len(loader)

    for atk_name in attacks_to_run:
        atk_t0 = time.time()
        try:
            attack_result, clean_m, adv_m, y_true, y_clean_pred, y_adv_pred, _ = \
                step3_and_4_attack_and_evaluate(
                    adapter=adapter, attack_name=atk_name, epsilon=epsilon,
                    samples=samples, batch_size=args.batch_size,
                    device=device, output_dir=output_dir, loader=loader,
                )
            atk_duration = time.time() - atk_t0

            assessment, scoring = step5_vulnerability_analysis(
                baseline_result=baseline_result, attack_result=attack_result,
                clean_m=clean_m, adv_m=adv_m,
                y_true=y_true, y_clean_pred=y_clean_pred, y_adv_pred=y_adv_pred,
                num_samples=total_samples, attack_name=atk_name, output_dir=output_dir,
            )

            attack_cls = get_attack(atk_name.lower())
            attack_class_name = attack_cls.__name__ if attack_cls else atk_name.upper()

            attack_eval_results.append({
                "attack_name": atk_name,
                "attack_class": attack_class_name,
                "execution_time": atk_duration,
                "assessment": assessment,
                "scoring": scoring,
                "clean_m": clean_m,
                "adv_m": adv_m,
            })
        except Exception as e:
            atk_duration = time.time() - atk_t0
            pipeline_errors.append({
                "module": f"M3_{atk_name}",
                "error": f"{type(e).__name__} — {str(e)}",
            })
            print(f"  ❌ Error executing attack {atk_name.upper()}: {e}")

    total_pipeline_runtime = time.time() - pipeline_start_time

    step6_final_summary(
        baseline=baseline_result,
        attack_eval_results=attack_eval_results,
        attacks_to_run=attacks_to_run,
        metadata=metadata,
        device=device,
        samples=total_samples,
        batch_size=args.batch_size,
        pipeline_errors=pipeline_errors,
        total_pipeline_runtime=total_pipeline_runtime,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()

