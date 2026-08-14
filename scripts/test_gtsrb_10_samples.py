"""
Controlled Real-Data 10-Sample Integration Test for GTSRB Baseline & Adversarial Evaluation (Modules 1, 2, 3).
Uses bazyl/GTSRB (test split, first 10 samples) and bazyl/gtsrb-model via Module 1 ingest_model().
Executes Module 2 Baseline Evaluation and Module 3 Adversarial Attack Engine (execute_attack).
"""

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
from app.attack_engine import execute_attack, AttackConfig, get_attack


def main():
    print("=" * 65)
    print("AdverScan — 10-Sample GTSRB Baseline & Adversarial Attack Test")
    print("=" * 65)

    model_name = "bazyl/gtsrb-model"
    dataset_name = "bazyl/GTSRB"
    num_classes = 43
    num_samples_target = 10
    epsilon = 0.03137  # 8/255

    # 1. Patch config (fixes upstream null in id2label['43']) & load cached HF ViT model
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        model_name="GTSRB_ViT_10Sample_Test",
        device=device,
    )

    print(f"M1 Ingestion Complete:")
    print(f"  Framework:     {metadata.framework}")
    print(f"  Model Name:    {metadata.model_name}")
    print(f"  Target Device: {adapter.device}")
    print(f"  Adapter Type:  {adapter.__class__.__name__}")

    # 2. Initialize GTSRB dataset loader for exactly 10 samples from test split
    print(f"\n[2/4] Loading exactly {num_samples_target} samples from '{dataset_name}' (test split) ...")
    loader = GTSRBDatasetLoader(
        dataset_name=dataset_name,
        processor_name=model_name,
        split=f"test[:{num_samples_target}]",
        batch_size=10,
    )
    print(f"  Loaded dataset samples: {len(loader)}")

    # 3. Execute BaselineEvaluator over 10 samples
    print("\n[3/4] Running Module 2 BaselineEvaluator engine ...")
    evaluator = BaselineEvaluator(
        adapter=adapter,
        dataset_loader=loader,
        num_classes=num_classes,
        model_name="GTSRB_ViT_10Sample_Baseline",
    )

    result = evaluator.evaluate(output_dir=None)

    # 4. Execute Module 3 Attack Engine (execute_attack) with FGSM
    print(f"\n[4/4] Executing Module 3 Attack Engine (execute_attack with FGSM, eps={epsilon:.5f}) ...")
    attack_cls = get_attack("fgsm")
    attack_config = AttackConfig(epsilon=epsilon)

    all_targets: list[int] = []
    clean_preds: list[int] = []
    adv_preds: list[int] = []
    clean_probs_list: list[np.ndarray] = []
    adv_probs_list: list[np.ndarray] = []

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

        # Module 3 execute_attack call
        adv_pixels = execute_attack(
            model=adapter,
            attack_cls=attack_cls,
            inputs=batch_pixels,
            labels=batch_targets,
            config=attack_config,
        )

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

    elapsed_sec = time.time() - start_time

    # Calculate Adversarial Metrics
    y_true = np.array(all_targets, dtype=np.int64)
    y_clean_pred = np.array(clean_preds, dtype=np.int64)
    y_adv_pred = np.array(adv_preds, dtype=np.int64)
    y_clean_probs = np.concatenate(clean_probs_list, axis=0)
    y_adv_probs = np.concatenate(adv_probs_list, axis=0)

    adv_metrics = MetricsCalculator.compute_metrics(
        y_true=y_true, y_pred=y_adv_pred, y_probs=y_adv_probs, num_classes=num_classes
    )

    clean_acc = result.accuracy
    adv_acc = adv_metrics["accuracy"]
    acc_drop = clean_acc - adv_acc

    clean_correct_mask = y_clean_pred == y_true
    num_clean_correct = int(np.sum(clean_correct_mask))
    if num_clean_correct > 0:
        successful_attacks = int(np.sum(clean_correct_mask & (y_adv_pred != y_true)))
        asr = float(successful_attacks / num_clean_correct)
    else:
        asr = 0.0

    # 5. Report Results
    print("\n" + "=" * 65)
    print("10-SAMPLE GTSRB BASELINE & ADVERSARIAL INTEGRATION TEST RESULTS")
    print("=" * 65)
    print(f"Dataset Identifier:        {result.dataset_name} (test[:10])")
    print(f"Model Identifier:          {result.model_name}")
    print(f"Evaluated Samples:         {result.num_samples}")
    print(f"Execution Device:          {result.device}")
    print(f"Execution Runtime:         {elapsed_sec:.2f} seconds")
    print("-" * 65)
    print("MODULE 2 BASELINE EVALUATION METRICS:")
    print(f"  Clean Accuracy:          {clean_acc * 100:.2f}% ({clean_acc:.4f})")
    print(f"  Macro Precision:         {result.precision_macro * 100:.2f}%")
    print(f"  Macro Recall:            {result.recall_macro * 100:.2f}%")
    print(f"  Macro F1-Score:          {result.f1_macro * 100:.2f}%")
    print(f"  Average Confidence:      {result.average_confidence * 100:.2f}% ({result.average_confidence:.6f})")
    print(f"  Average Entropy:         {result.average_entropy:.6f} bits")
    print("-" * 65)
    print("MODULE 3 ADVERSARIAL ATTACK (execute_attack) METRICS:")
    print(f"  Attack Type:             FGSM (eps={epsilon:.5f})")
    print(f"  Adversarial Accuracy:    {adv_acc * 100:.2f}% ({adv_acc:.4f})")
    print(f"  Attack Success Rate:     {asr * 100:.2f}% ({asr:.4f})")
    print(f"  Accuracy Drop:           {acc_drop * 100:.2f}% ({acc_drop:.4f})")
    print(f"  Adversarial Avg Conf:    {adv_metrics['average_confidence'] * 100:.2f}% ({adv_metrics['average_confidence']:.6f})")
    print(f"  Adversarial Avg Entropy: {adv_metrics['average_entropy']:.6f} bits")
    print("=" * 65)
    print("\n✅ 10-Sample Real-Data Baseline & Attack Test Completed Successfully.")


if __name__ == "__main__":
    main()
