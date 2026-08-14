"""
Generalized Real-Data Sample Integration Test for GTSRB Baseline & Adversarial Evaluation (Modules 1, 2, 3).
Uses bazyl/GTSRB and bazyl/gtsrb-model via Module 1 ingest_model().
Dynamically discovers and executes registered attack classes via Module 3 Attack Engine (execute_attack),
evaluates individual attack results, and computes combined (worst-case / ensemble) robustness metrics.
"""

import argparse
import json
import time
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForImageClassification, ViTConfig

from app.ingestion import ingest_model
from app.evaluation.dataset_loader import GTSRBDatasetLoader
from app.evaluation.evaluator import BaselineEvaluator
from app.evaluation.metrics import MetricsCalculator
from app.attack_engine import (
    execute_attack,
    AttackConfig,
    discover_attacks,
    select_attacks,
    list_attacks,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="AdverScan Generalized GTSRB Baseline & Adversarial Attack Integration Test"
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="all",
        help="Target attack name (e.g., 'fgsm', 'pgd', 'deepfool', or 'all' to run all registered attacks)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.03137,
        help="Adversarial perturbation budget epsilon (default: 8/255 ~ 0.03137)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of test samples to evaluate (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Evaluation batch size (default: 10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device ('cuda' or 'cpu')",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_name = "bazyl/gtsrb-model"
    dataset_name = "bazyl/GTSRB"
    num_classes = 43

    # Device selection
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("AdverScan — Generalized GTSRB Baseline & Multi-Attack Evaluation")
    print("=" * 70)
    print(f"Target Device:             {device}")
    print(f"Dataset Split:             {dataset_name} (test[:{args.samples}])")
    print(f"Target Attack(s):          {args.attack.upper()}")
    print(f"Epsilon Budget:            {args.epsilon:.5f}")
    print(f"Samples Count:             {args.samples}")
    print(f"Batch Size:                {args.batch_size}")

    # 1. Discover registered attack modules dynamically
    discover_attacks()
    available_attacks = list_attacks()
    print(f"Registered Attacks:        {available_attacks}")

    # Determine attack classes to evaluate
    if args.attack.lower() == "all":
        selected_attack_classes = select_attacks(available_attacks)
    else:
        selected_attack_classes = select_attacks(args.attack.lower())

    # 2. Patch config & load HF ViT model via Module 1 ingest_model()
    print(f"\n[1/4] Ingesting Model via Module 1 (M1) contract on {device} ...")
    start_time = time.time()

    config_path = hf_hub_download(model_name, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    if "id2label" in cfg_dict:
        cfg_dict["id2label"] = {
            str(k): (str(v) if v is not None else "Unused")
            for k, v in cfg_dict["id2label"].items()
        }
        cfg_dict["label2id"] = {v: int(k) for k, v in cfg_dict["id2label"].items()}
        cfg_dict["num_labels"] = len(cfg_dict["id2label"])

    model_config = ViTConfig.from_dict(cfg_dict)
    raw_model = AutoModelForImageClassification.from_pretrained(
        model_name, config=model_config, use_safetensors=True
    )

    # Ingest model through M1 contract
    sample_input = torch.randn(1, 3, 224, 224)
    adapter, metadata = ingest_model(
        model_path=raw_model,
        sample_input=sample_input,
        model_name="GTSRB_ViT_Generalized_Test",
        device=device,
    )

    print(f"M1 Ingestion Complete:")
    print(f"  Framework:     {metadata.framework}")
    print(f"  Model Name:    {metadata.model_name}")
    print(f"  Target Device: {adapter.device}")
    print(f"  Adapter Type:  {adapter.__class__.__name__}")

    # 3. Initialize GTSRB dataset loader via Module 2
    split_str = f"test[:{args.samples}]" if args.samples > 0 else "test"
    print(f"\n[2/4] Loading dataset split '{split_str}' from '{dataset_name}' ...")
    loader = GTSRBDatasetLoader(
        dataset_name=dataset_name,
        processor_name=model_name,
        split=split_str,
        batch_size=args.batch_size,
    )
    print(f"  Loaded dataset samples: {len(loader)}")

    # 4. Execute Module 2 Baseline Evaluator
    print("\n[3/4] Running Module 2 BaselineEvaluator engine ...")
    evaluator = BaselineEvaluator(
        adapter=adapter,
        dataset_loader=loader,
        num_classes=num_classes,
        model_name="GTSRB_ViT_Baseline",
    )
    baseline_result = evaluator.evaluate(output_dir=None)

    # 5. Execute Module 3 Attack Engine for each selected attack
    print(f"\n[4/4] Executing Module 3 Attack Engine across {len(selected_attack_classes)} attack(s) ...")

    attack_results = {}
    sample_adv_preds = {}  # attack_name -> list of adv_preds per sample
    y_true_global: list[int] = []
    clean_preds_global: list[int] = []

    for idx, attack_cls in enumerate(selected_attack_classes):
        attack_name = getattr(attack_cls, "attack_name", attack_cls.__name__.lower())
        print(f"\n--- Running Attack [{idx+1}/{len(selected_attack_classes)}]: {attack_name.upper()} ({attack_cls.__name__}) ---")

        attack_config = AttackConfig(epsilon=args.epsilon)

        all_targets: list[int] = []
        clean_preds: list[int] = []
        adv_preds: list[int] = []
        clean_probs_list: list[np.ndarray] = []
        adv_probs_list: list[np.ndarray] = []

        atk_start = time.time()

        for batch_pixels, batch_targets, _ in loader.iterate_batches():
            batch_pixels = batch_pixels.to(device)
            batch_targets = batch_targets.to(device)

            # Clean Prediction
            with torch.no_grad():
                clean_outputs = adapter.predict(batch_pixels)
                if hasattr(clean_outputs, "logits"):
                    clean_outputs = clean_outputs.logits
                clean_logits = (
                    clean_outputs[:, :num_classes]
                    if clean_outputs.shape[-1] > num_classes
                    else clean_outputs
                )
                clean_probs = torch.softmax(clean_logits, dim=-1)
                c_preds = torch.argmax(clean_probs, dim=-1)

            # Dynamic execute_attack call
            adv_result = execute_attack(
                model=adapter,
                attack_cls=attack_cls,
                inputs=batch_pixels,
                labels=batch_targets,
                config=attack_config,
            )
            adv_pixels = adv_result.adversarial_examples

            # Adversarial Prediction
            with torch.no_grad():
                adv_outputs = adapter.predict(adv_pixels)
                if hasattr(adv_outputs, "logits"):
                    adv_outputs = adv_outputs.logits
                adv_logits = (
                    adv_outputs[:, :num_classes]
                    if adv_outputs.shape[-1] > num_classes
                    else adv_outputs
                )
                adv_probs = torch.softmax(adv_logits, dim=-1)
                a_preds = torch.argmax(adv_probs, dim=-1)

            all_targets.extend(batch_targets.cpu().numpy().tolist())
            clean_preds.extend(c_preds.cpu().numpy().tolist())
            adv_preds.extend(a_preds.cpu().numpy().tolist())
            clean_probs_list.append(clean_probs.cpu().numpy())
            adv_probs_list.append(adv_probs.cpu().numpy())

        atk_elapsed = time.time() - atk_start

        # Store global targets & clean predictions from first iteration
        if not y_true_global:
            y_true_global = all_targets
            clean_preds_global = clean_preds

        sample_adv_preds[attack_name] = adv_preds

        # Calculate Individual Adversarial Metrics
        y_true = np.array(all_targets, dtype=np.int64)
        y_clean_pred = np.array(clean_preds, dtype=np.int64)
        y_adv_pred = np.array(adv_preds, dtype=np.int64)
        y_adv_probs = np.concatenate(adv_probs_list, axis=0)

        adv_metrics = MetricsCalculator.compute_metrics(
            y_true=y_true, y_pred=y_adv_pred, y_probs=y_adv_probs, num_classes=num_classes
        )

        clean_acc = baseline_result.accuracy
        adv_acc = adv_metrics["accuracy"]
        acc_drop = clean_acc - adv_acc

        clean_correct_mask = y_clean_pred == y_true
        num_clean_correct = int(np.sum(clean_correct_mask))
        if num_clean_correct > 0:
            successful_attacks = int(np.sum(clean_correct_mask & (y_adv_pred != y_true)))
            asr = float(successful_attacks / num_clean_correct)
        else:
            asr = 0.0

        attack_results[attack_name] = {
            "attack_class": attack_cls.__name__,
            "adv_accuracy": adv_acc,
            "attack_success_rate": asr,
            "accuracy_drop": acc_drop,
            "adv_confidence": adv_metrics["average_confidence"],
            "adv_entropy": adv_metrics["average_entropy"],
            "runtime_seconds": atk_elapsed,
        }

    elapsed_sec = time.time() - start_time

    # 6. Compute Combined (Multi-Attack Ensemble / Worst-Case) Results
    y_true_arr = np.array(y_true_global, dtype=np.int64)
    y_clean_arr = np.array(clean_preds_global, dtype=np.int64)
    clean_correct_mask = y_clean_arr == y_true_arr
    num_clean_correct = int(np.sum(clean_correct_mask))

    # A sample is considered correctly classified under COMBINED attacks only if it is correctly classified under EVERY attack
    combined_correct_mask = np.ones(len(y_true_arr), dtype=bool)
    # A sample correctly classified on clean data is broken by combined attacks if ANY attack breaks it
    combined_attack_failed_mask = np.zeros(len(y_true_arr), dtype=bool)

    for atk_name, preds in sample_adv_preds.items():
        preds_arr = np.array(preds, dtype=np.int64)
        combined_correct_mask &= (preds_arr == y_true_arr)
        combined_attack_failed_mask |= (preds_arr != y_true_arr)

    combined_adv_acc = float(np.mean(combined_correct_mask))
    combined_acc_drop = baseline_result.accuracy - combined_adv_acc

    if num_clean_correct > 0:
        combined_successful_attacks = int(np.sum(clean_correct_mask & combined_attack_failed_mask))
        combined_asr = float(combined_successful_attacks / num_clean_correct)
    else:
        combined_asr = 0.0

    mean_asr = float(np.mean([res["attack_success_rate"] for res in attack_results.values()]))
    mean_adv_acc = float(np.mean([res["adv_accuracy"] for res in attack_results.values()]))

    # 7. Report Integration Test & Combined Results
    print("\n" + "=" * 70)
    print("GENERALIZED GTSRB BASELINE & ADVERSARIAL TEST RESULTS")
    print("=" * 70)
    print(f"Dataset Identifier:        {baseline_result.dataset_name} ({split_str})")
    print(f"Model Identifier:          {baseline_result.model_name}")
    print(f"Evaluated Samples:         {baseline_result.num_samples}")
    print(f"Execution Device:          {baseline_result.device}")
    print(f"Execution Runtime:         {elapsed_sec:.2f} seconds")
    print("-" * 70)
    print("MODULE 2 BASELINE EVALUATION METRICS:")
    print(f"  Clean Accuracy:          {baseline_result.accuracy * 100:.2f}% ({baseline_result.accuracy:.4f})")
    print(f"  Macro Precision:         {baseline_result.precision_macro * 100:.2f}%")
    print(f"  Macro Recall:            {baseline_result.recall_macro * 100:.2f}%")
    print(f"  Macro F1-Score:          {baseline_result.f1_macro * 100:.2f}%")
    print(f"  Average Confidence:      {baseline_result.average_confidence * 100:.2f}% ({baseline_result.average_confidence:.6f})")
    print(f"  Average Entropy:         {baseline_result.average_entropy:.6f} bits")

    print("\n" + "=" * 70)
    print("INDIVIDUAL ATTACK EVALUATION METRICS:")
    print("=" * 70)
    for atk_name, res in attack_results.items():
        print(f"Attack: {atk_name.upper()} ({res['attack_class']})")
        print(f"  Epsilon Budget:          {args.epsilon:.5f}")
        print(f"  Adversarial Accuracy:    {res['adv_accuracy'] * 100:.2f}% ({res['adv_accuracy']:.4f})")
        print(f"  Attack Success Rate:     {res['attack_success_rate'] * 100:.2f}% ({res['attack_success_rate']:.4f})")
        print(f"  Accuracy Drop:           {res['accuracy_drop'] * 100:.2f}% ({res['accuracy_drop']:.4f})")
        print(f"  Adversarial Avg Conf:    {res['adv_confidence'] * 100:.2f}% ({res['adv_confidence']:.6f})")
        print(f"  Adversarial Avg Entropy: {res['adv_entropy']:.6f} bits")
        print(f"  Execution Time:          {res['runtime_seconds']:.2f} seconds")
        print("-" * 70)

    print("\n" + "=" * 70)
    print("COMBINED (ENSEMBLE / WORST-CASE) MULTI-ATTACK RESULTS:")
    print("=" * 70)
    print(f"  Evaluated Attacks Count: {len(attack_results)} ({', '.join(a.upper() for a in attack_results.keys())})")
    print(f"  Clean Accuracy:          {baseline_result.accuracy * 100:.2f}%")
    print(f"  Mean Adversarial Acc:    {mean_adv_acc * 100:.2f}%")
    print(f"  Mean Attack Success Rate:{mean_asr * 100:.2f}%")
    print(f"  Worst-Case Adv Accuracy: {combined_adv_acc * 100:.2f}% ({combined_adv_acc:.4f})")
    print(f"  Combined ASR (>=1 fail): {combined_asr * 100:.2f}% ({combined_asr:.4f})")
    print(f"  Combined Accuracy Drop:  {combined_acc_drop * 100:.2f}% ({combined_acc_drop:.4f})")

    print("\n" + "=" * 70)
    print(f"{'ATTACK NAME':<15} | {'ADV ACC (%)':<12} | {'ASR (%)':<10} | {'ACC DROP (%)':<12} | {'TIME (s)':<8}")
    print("-" * 70)
    for atk_name, res in attack_results.items():
        print(f"{atk_name.upper():<15} | {res['adv_accuracy']*100:<12.2f} | {res['attack_success_rate']*100:<10.2f} | {res['accuracy_drop']*100:<12.2f} | {res['runtime_seconds']:<8.2f}")
    print("-" * 70)
    print(f"{'COMBINED (WORST)':<15} | {combined_adv_acc*100:<12.2f} | {combined_asr*100:<10.2f} | {combined_acc_drop*100:<12.2f} | {elapsed_sec:<8.2f}")
    print("=" * 70)
    print("\n✅ Generalized Real-Data Baseline & Multi-Attack Test Completed Successfully.")


if __name__ == "__main__":
    main()
