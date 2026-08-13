"""
Execution script to perform clean Baseline Evaluation (Module 2) on GTSRB test dataset (12,630 samples)
using Module 1 standardized PyTorch model adapter.
"""

import json
import time
import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from transformers import AutoModelForImageClassification, ViTConfig
from pathlib import Path

from app.ingestion import ingest_model
from app.evaluation import evaluate_baseline

# Load environment variables from .env if present
load_dotenv()


def main():
    print("=" * 65)
    print("AdverScan — Module 2 Baseline Evaluation Engine")
    print("=" * 65)

    model_name = "bazyl/gtsrb-model"
    dataset_name = "bazyl/GTSRB"
    batch_size = 32
    num_classes = 43

    print(f"\n[1/3] Standardizing & Ingesting Model via Module 1 (M1) ...")
    start_time = time.time()

    # Load and patch HF ViT config (fixes upstream null value in id2label['43'])
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
    raw_model = AutoModelForImageClassification.from_pretrained(model_name, config=model_config, use_safetensors=True)

    # Ingest model through Module 1 standardized interface
    sample_input = torch.randn(1, 3, 224, 224)
    adapter, metadata = ingest_model(
        model_path=raw_model,
        sample_input=sample_input,
        model_name="GTSRB_ViT_Classifier",
        device="cpu",
    )

    print(f"M1 Ingestion Complete.")
    print(f"  Framework: {metadata.framework}")
    print(f"  Model Name: {metadata.model_name}")
    print(f"  Target Device: {adapter.device}")
    print(f"  Adapter Type: {adapter.__class__.__name__}")

    # -------------------------------------------------------------
    # [2/3] Run Baseline Evaluation over full GTSRB test set
    # -------------------------------------------------------------
    print(f"\n[2/3] Running Clean Baseline Evaluation on '{dataset_name}' (test split) ...")
    print(f"  Batch size: {batch_size}")
    print(f"  Target classes: {num_classes}")

    result = evaluate_baseline(
        adapter=adapter,
        dataset_name=dataset_name,
        processor_name=model_name,
        split="test",
        batch_size=batch_size,
        num_classes=num_classes,
        model_name="GTSRB_ViT_Baseline",
        output_dir="results/baseline",
    )

    elapsed_sec = time.time() - start_time

    # -------------------------------------------------------------
    # [3/3] Report Baseline Evaluation Results
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("ADVERSCAN — M2 BASELINE EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Dataset Identifier:        {result.dataset_name} (test split)")
    print(f"Model Identifier:          {result.model_name}")
    print(f"Evaluated Samples:         {result.num_samples:,}")
    print(f"Number of Classes:         {result.num_classes}")
    print(f"Execution Device:          {result.device}")
    print(f"Evaluation Runtime:        {elapsed_sec:.2f} seconds")
    print("-" * 65)
    print(f"Overall Accuracy:          {result.accuracy * 100:.2f}% ({result.accuracy:.6f})")
    print(f"Macro Precision:           {result.precision_macro * 100:.2f}% ({result.precision_macro:.6f})")
    print(f"Macro Recall:              {result.recall_macro * 100:.2f}% ({result.recall_macro:.6f})")
    print(f"Macro F1-Score:            {result.f1_macro * 100:.2f}% ({result.f1_macro:.6f})")
    print(f"Weighted Precision:        {result.precision_weighted * 100:.2f}% ({result.precision_weighted:.6f})")
    print(f"Weighted Recall:           {result.recall_weighted * 100:.2f}% ({result.recall_weighted:.6f})")
    print(f"Weighted F1-Score:         {result.f1_weighted * 100:.2f}% ({result.f1_weighted:.6f})")
    print(f"Average Confidence:        {result.average_confidence * 100:.2f}% ({result.average_confidence:.6f})")
    print(f"Average Entropy:           {result.average_entropy:.6f} bits")
    print(f"Confusion Matrix Shape:    {len(result.confusion_matrix)} × {len(result.confusion_matrix[0])}")
    print("=" * 65)

    json_artifact = Path("results/baseline") / "baseline_gtsrb_vit_baseline.json"
    print(f"\n✅ Baseline Result persisted to: {json_artifact}")


if __name__ == "__main__":
    main()
