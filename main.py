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

Examples:
    python main.py
    python main.py --samples 100 --attack pgd
    python main.py --samples 25 --attack deepfool --device cuda
    python main.py --attack fgsm --epsilon 0.05 --samples 50
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

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
from app.vulnerability_analysis import VulnerabilityEngine
from app.vulnerability_analysis.assessment_result import AssessmentResult
from app.vulnerability_analysis.scoring_result import ScoringResult

load_dotenv()

MODEL_ID      = "bazyl/gtsrb-model"
DATASET_NAME  = "bazyl/GTSRB"
NUM_CLASSES   = 43
SEPARATOR     = "=" * 68
SUPPORTED_ATTACKS = ["fgsm", "pgd", "deepfool", "cw", "fab"]


def banner():
    print(SEPARATOR)
    print("  AdverScan - Transportation AI Adversarial Robustness System")
    print("  Full Pipeline Demo  |  GTSRB Traffic Sign Classification")
    print(SEPARATOR)


def parse_args():
    parser = argparse.ArgumentParser(
        description="AdverScan full pipeline demo for GTSRB model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--samples",    type=int,   default=50,       help="Number of GTSRB test samples")
    parser.add_argument("--attack",     type=str,   default="fgsm",   choices=SUPPORTED_ATTACKS, help="Adversarial attack")
    parser.add_argument("--epsilon",    type=float, default=8/255,    help="Perturbation budget epsilon")
    parser.add_argument("--batch-size", type=int,   default=32,       help="Batch size")
    parser.add_argument("--device",     type=str,   default=None,     help="'cpu' or 'cuda'")
    parser.add_argument("--output-dir", type=str,   default="results",help="Root results directory")
    return parser.parse_args()


def resolve_device(requested):
    if requested and requested.startswith("cuda"):
        if torch.cuda.is_available():
            return "cuda"
        print("  WARNING: CUDA requested but unavailable - falling back to CPU.")
        return "cpu"
    if requested == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


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
        model_name="GTSRB_ViT",
        device=device,
    )
    print(f"  Ingestion complete in {time.time() - t0:.1f}s")
    print(f"  Framework:   {metadata.framework}")
    if metadata.total_parameters:
        print(f"  Parameters:  {metadata.total_parameters:,} (~{metadata.total_parameters / 1e6:.1f}M)")
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
def step3_and_4_attack_and_evaluate(adapter, attack_name, epsilon, samples, batch_size, device, output_dir):
    section(f"STEP 3 - Adversarial Attack ({attack_name.upper()}, epsilon={epsilon:.5f})")
    split_str = f"test[:{samples}]" if samples > 0 else "test"
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

    section("STEP 4 - Adversarial Evaluation Metrics")
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

    return all_adv_results[-1], clean_m, adv_m, y_true, y_clean_pred, y_adv_pred, total_samples


# -- STEP 5: Vulnerability Analysis --------------------------------------------
def step5_vulnerability_analysis(baseline_result, attack_result, clean_m, adv_m,
                                  y_true, y_clean_pred, y_adv_pred, num_samples, attack_name, output_dir):
    section("STEP 5 - Vulnerability Analysis (M4)")
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
def step6_final_summary(baseline, assessment, scoring, attack_name, epsilon, samples):
    print(f"\n{SEPARATOR}")
    print("  AdverScan - FINAL PIPELINE SUMMARY")
    print(SEPARATOR)
    print(f"  Model:            {MODEL_ID}")
    print(f"  Dataset:          {DATASET_NAME}  ({samples} samples)")
    print(f"  Attack:           {attack_name.upper()}")
    print(f"  Epsilon (eps):    {epsilon:.5f}  (~{epsilon * 255:.1f}/255)")
    print(SEPARATOR)
    print(f"  {'Metric':<28} {'Clean':>12} {'Adversarial':>13}")
    print(f"  {'-'*55}")
    print(f"  {'Accuracy':<28} {assessment.clean_accuracy * 100:>11.2f}% {assessment.adversarial_accuracy * 100:>12.2f}%")
    print(f"  {'Macro F1':<28} {assessment.clean_f1 * 100:>11.2f}% {assessment.adversarial_f1 * 100:>12.2f}%")
    print(f"  {'Avg Confidence':<28} {assessment.clean_confidence * 100:>11.2f}% {assessment.adversarial_confidence * 100:>12.2f}%")
    print(f"  {'-'*55}")
    print(f"  {'Accuracy Drop':<28} {'':>12} {assessment.accuracy_drop * 100:>12.2f}%")
    print(f"  {'Model Degradation':<28} {'':>12} {assessment.model_degradation * 100:>12.2f}%")
    asr_str = f"{assessment.attack_success_rate * 100:.2f}%" if assessment.attack_success_rate is not None else "N/A"
    print(f"  {'Attack Success Rate':<28} {'':>12} {asr_str:>12}")
    print(f"  {'-'*55}")
    risk_icon = {"LOW": "[LOW]", "MEDIUM": "[MED]", "HIGH": "[HIGH]", "CRITICAL": "[CRIT]"}.get(scoring.risk_level, "")
    print(f"  {'Vulnerability Score':<28} {'':>12} {scoring.vulnerability_score:>9.2f}/100")
    print(f"  {'Risk Level':<28} {'':>12} {risk_icon} {scoring.risk_level:>8}")
    print(SEPARATOR)
    print("  Pipeline complete. JSON results saved to ./results/")
    print(SEPARATOR)


# -- MAIN ----------------------------------------------------------------------
def main():
    banner()
    args       = parse_args()
    device     = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Device: {device}  |  Samples: {args.samples}  |  Attack: {args.attack.upper()}  |  Epsilon: {args.epsilon:.5f}")

    adapter, metadata = step1_ingest_model(device)

    baseline_result = step2_baseline_evaluation(
        adapter=adapter, samples=args.samples,
        batch_size=args.batch_size, output_dir=output_dir,
    )

    attack_result, clean_m, adv_m, y_true, y_clean_pred, y_adv_pred, total_samples = \
        step3_and_4_attack_and_evaluate(
            adapter=adapter, attack_name=args.attack, epsilon=args.epsilon,
            samples=args.samples, batch_size=args.batch_size,
            device=device, output_dir=output_dir,
        )

    assessment, scoring = step5_vulnerability_analysis(
        baseline_result=baseline_result, attack_result=attack_result,
        clean_m=clean_m, adv_m=adv_m,
        y_true=y_true, y_clean_pred=y_clean_pred, y_adv_pred=y_adv_pred,
        num_samples=total_samples, attack_name=args.attack, output_dir=output_dir,
    )

    step6_final_summary(
        baseline=baseline_result, assessment=assessment, scoring=scoring,
        attack_name=args.attack, epsilon=args.epsilon, samples=total_samples,
    )


if __name__ == "__main__":
    main()
