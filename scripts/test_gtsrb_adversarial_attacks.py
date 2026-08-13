"""
Adversarial Attack Evaluation Script for GTSRB ViT Model (Module 2/3).
Evaluates clean vs. adversarial accuracy, Attack Success Rate (ASR), accuracy drop,
average confidence, and Shannon entropy under adversarial attacks (e.g. FGSM).
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from transformers import AutoModelForImageClassification, ViTConfig

from app.ingestion import ingest_model
from app.evaluation.dataset_loader import GTSRBDatasetLoader
from app.evaluation.metrics import MetricsCalculator
from app.attacks import FGSMAttack

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="AdverScan GTSRB Adversarial Attack Evaluation")
    parser.add_argument("--samples", type=int, default=50, help="Number of test samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--epsilon", type=float, default=0.03137, help="Adversarial attack epsilon (default: 8/255 ~ 0.03137)")
    parser.add_argument("--attack", type=str, default="fgsm", choices=["fgsm"], help="Adversarial attack type")
    parser.add_argument("--device", type=str, default=None, help="Target device ('cuda' or 'cpu')")
    return parser.parse_args()


def main():
    args = parse_args()

    model_name = "bazyl/gtsrb-model"
    dataset_name = "bazyl/GTSRB"
    num_classes = 43

    # 1. Device Setup
    if args.device:
        device = args.device
        gpu_available = torch.cuda.is_available() and device.startswith("cuda")
    else:
        gpu_available = torch.cuda.is_available()
        device = "cuda" if gpu_available else "cpu"

    gpu_model = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

    print("=" * 64)
    print("AdverScan — GTSRB Adversarial Attack Evaluation")
    print("=" * 64)
    print("\nExecution Environment")
    print(f"GPU Available:      {gpu_available}")
    print(f"GPU Model:          {gpu_model}")
    print(f"Execution Device:   {device}")
    print(f"\nDataset:            {dataset_name} (test[:{args.samples}])")
    print(f"Attack:             {args.attack.upper()}")
    print(f"Epsilon:            {args.epsilon:.5f}")
    print(f"Samples:            {args.samples}")
    print(f"Batch Size:         {args.batch_size}")

    if not gpu_available and device.startswith("cuda"):
        print("⚠️ WARNING: CUDA is unavailable. Falling back to CPU.")
        device = "cpu"

    # 2. Patch config & load cached HF ViT model with use_safetensors=True
    print(f"\n[1/3] Ingesting Model via Module 1 (M1) contract on {device} ...")
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
        model_name="GTSRB_ViT_Adversarial_Test",
        device=device,
    )

    print("M1 Ingestion Complete:")
    print(f"  Framework:     {metadata.framework}")
    print(f"  Model Name:    {metadata.model_name}")
    print(f"  Target Device: {adapter.device}")
    print(f"  Adapter Type:  {adapter.__class__.__name__}")

    # 3. Load dataset split
    split_str = f"test[:{args.samples}]" if args.samples > 0 else "test"
    print(f"\n[2/3] Loading dataset split '{split_str}' from '{dataset_name}' ...")
    loader = GTSRBDatasetLoader(
        dataset_name=dataset_name,
        processor_name=model_name,
        split=split_str,
        batch_size=args.batch_size,
    )
    total_samples = len(loader)
    print(f"  Loaded Samples: {total_samples}")

    if total_samples == 0:
        raise ValueError("Dataset loader returned 0 samples. Aborting.")

    # 4. Instantiate Attack Generator
    if args.attack.lower() == "fgsm":
        attack_generator = FGSMAttack(
            model_adapter=adapter,
            epsilon=args.epsilon,
            num_classes=num_classes,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported attack type: {args.attack}")

    # 5. Run Clean vs Adversarial Evaluation Loop
    print(f"\n[3/3] Executing {args.attack.upper()} Attack Generation & Model Evaluation ...")
    all_targets: list[int] = []
    clean_preds: list[int] = []
    adv_preds: list[int] = []
    clean_probs_list: list[np.ndarray] = []
    adv_probs_list: list[np.ndarray] = []

    processed_count = 0
    for batch_pixels, batch_targets, _ in loader.iterate_batches():
        batch_pixels = batch_pixels.to(device)
        batch_targets = batch_targets.to(device)
        batch_len = len(batch_targets)

        # 5a. Clean Forward Pass
        with torch.no_grad():
            clean_outputs = adapter.predict(batch_pixels)
            if hasattr(clean_outputs, "logits"):
                clean_outputs = clean_outputs.logits
            clean_logits = clean_outputs[:, :num_classes] if clean_outputs.shape[-1] > num_classes else clean_outputs
            clean_probs = torch.softmax(clean_logits, dim=-1)
            c_preds = torch.argmax(clean_probs, dim=-1)

        # 5b. Generate Adversarial Perturbation
        adv_pixels = attack_generator.generate(batch_pixels, batch_targets)

        # 5c. Adversarial Forward Pass
        with torch.no_grad():
            adv_outputs = adapter.predict(adv_pixels)
            if hasattr(adv_outputs, "logits"):
                adv_outputs = adv_outputs.logits
            adv_logits = adv_outputs[:, :num_classes] if adv_outputs.shape[-1] > num_classes else adv_outputs
            adv_probs = torch.softmax(adv_logits, dim=-1)
            a_preds = torch.argmax(adv_probs, dim=-1)

        all_targets.extend(batch_targets.cpu().numpy().tolist())
        clean_preds.extend(c_preds.cpu().numpy().tolist())
        adv_preds.extend(a_preds.cpu().numpy().tolist())

        clean_probs_list.append(clean_probs.cpu().numpy())
        adv_probs_list.append(adv_probs.cpu().numpy())

        processed_count += batch_len
        print(f"Processing {processed_count}/{total_samples} samples...", end="\r", flush=True)

    print()  # New line after evaluation

    elapsed_sec = time.time() - start_time

    # 6. Metrics Calculation
    y_true = np.array(all_targets, dtype=np.int64)
    y_clean_pred = np.array(clean_preds, dtype=np.int64)
    y_adv_pred = np.array(adv_preds, dtype=np.int64)

    y_clean_probs = np.concatenate(clean_probs_list, axis=0)
    y_adv_probs = np.concatenate(adv_probs_list, axis=0)

    clean_metrics = MetricsCalculator.compute_metrics(
        y_true=y_true, y_pred=y_clean_pred, y_probs=y_clean_probs, num_classes=num_classes
    )
    adv_metrics = MetricsCalculator.compute_metrics(
        y_true=y_true, y_pred=y_adv_pred, y_probs=y_adv_probs, num_classes=num_classes
    )

    clean_acc = clean_metrics["accuracy"]
    adv_acc = adv_metrics["accuracy"]
    acc_drop = clean_acc - adv_acc

    # Calculate Attack Success Rate (ASR)
    clean_correct_mask = (y_clean_pred == y_true)
    num_clean_correct = int(np.sum(clean_correct_mask))
    if num_clean_correct > 0:
        successful_attacks = int(np.sum(clean_correct_mask & (y_adv_pred != y_true)))
        asr = float(successful_attacks / num_clean_correct)
    else:
        asr = 0.0

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 7. Print Formatted Results
    print("\n" + "=" * 64)
    print("RESULTS")
    print("=" * 64)
    print(f"Clean Accuracy:                 {clean_acc * 100:.2f}% ({clean_acc:.6f})")
    print(f"Adversarial Accuracy:           {adv_acc * 100:.2f}% ({adv_acc:.6f})")
    print(f"Attack Success Rate:            {asr * 100:.2f}% ({asr:.6f})")
    print(f"Accuracy Drop:                  {acc_drop * 100:.2f}% ({acc_drop:.6f})")
    print(f"Clean Avg Confidence:           {clean_metrics['average_confidence'] * 100:.2f}% ({clean_metrics['average_confidence']:.6f})")
    print(f"Adversarial Avg Confidence:     {adv_metrics['average_confidence'] * 100:.2f}% ({adv_metrics['average_confidence']:.6f})")
    print(f"Clean Avg Entropy:              {clean_metrics['average_entropy']:.6f} bits")
    print(f"Adversarial Avg Entropy:        {adv_metrics['average_entropy']:.6f} bits")
    print("=" * 64)

    # 8. Save Adversarial Results JSON
    output_dir = Path("results/adversarial")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"adversarial_{args.attack.lower()}_eps{args.epsilon:.4f}_{total_samples}samples.json"

    json_data = {
        "dataset_name": dataset_name,
        "dataset_split": split_str,
        "model_name": "GTSRB_ViT_Adversarial_Evaluation",
        "attack_type": args.attack.lower(),
        "epsilon": args.epsilon,
        "num_samples": total_samples,
        "batch_size": args.batch_size,
        "device": device,
        "gpu_model": gpu_model,
        "execution_runtime_seconds": round(elapsed_sec, 2),
        "timestamp": timestamp,
        "clean_accuracy": clean_acc,
        "adversarial_accuracy": adv_acc,
        "attack_success_rate": asr,
        "accuracy_drop": acc_drop,
        "clean_average_confidence": clean_metrics["average_confidence"],
        "adversarial_average_confidence": adv_metrics["average_confidence"],
        "clean_average_entropy": clean_metrics["average_entropy"],
        "adversarial_average_entropy": adv_metrics["average_entropy"],
        "clean_metrics": clean_metrics,
        "adversarial_metrics": adv_metrics,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    print(f"\n✅ Adversarial Evaluation JSON saved to: {out_file}")


if __name__ == "__main__":
    main()
