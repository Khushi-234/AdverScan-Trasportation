"""
Real-Data 5,000-Sample Baseline Evaluation Integration Test for GTSRB (Module 2).
Uses bazyl/GTSRB (test split, test[:5000]) and bazyl/gtsrb-model via Module 1 ingest_model().
Executes on GPU (NVIDIA GeForce RTX 4060 Ti) when CUDA is available.
"""

import json
import sys
import time
import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from transformers import AutoModelForImageClassification, ViTConfig

from app.ingestion import ingest_model
from app.evaluation.dataset_loader import GTSRBDatasetLoader
from app.evaluation.evaluator import BaselineEvaluator

load_dotenv()


def main():
    print("=" * 65)
    print("AdverScan — 5,000-Sample GTSRB Baseline Evaluation")
    print("=" * 65)

    model_name = "bazyl/gtsrb-model"
    dataset_name = "bazyl/GTSRB"
    num_classes = 43
    num_samples_target = 5000
    batch_size = 32

    # 1. GPU & Device Setup
    gpu_available = torch.cuda.is_available()
    device = "cuda" if gpu_available else "cpu"
    gpu_model = torch.cuda.get_device_name(0) if gpu_available else "N/A"

    print("\n--- Execution Environment ---")
    print(f"GPU Available:     {gpu_available}")
    print(f"GPU Model:         {gpu_model}")
    print(f"Execution Device:  {device}")

    if not gpu_available:
        print("⚠️ WARNING: CUDA is unavailable. Falling back to CPU.")

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
        model_name="GTSRB_ViT_5000Sample_Test",
        device=device,
    )

    print("M1 Ingestion Complete:")
    print(f"  Framework:     {metadata.framework}")
    print(f"  Model Name:    {metadata.model_name}")
    print(f"  Target Device: {adapter.device}")
    print(f"  Adapter Type:  {adapter.__class__.__name__}")

    # 3. Initialize GTSRB dataset loader for exactly 5,000 samples from test split
    print(f"\n[2/3] Loading dataset split 'test[:{num_samples_target}]' from '{dataset_name}' ...")
    loader = GTSRBDatasetLoader(
        dataset_name=dataset_name,
        processor_name=model_name,
        split=f"test[:{num_samples_target}]",
        batch_size=batch_size,
    )
    
    loaded_count = len(loader)
    print(f"  Requested Samples: {num_samples_target}")
    print(f"  Loaded Samples:    {loaded_count}")
    print(f"  Batch Size:        {batch_size}")

    if loaded_count < num_samples_target:
        raise ValueError(
            f"Dataset returned fewer samples ({loaded_count}) than requested ({num_samples_target}). Aborting."
        )

    # Wrap dataset loader's iterate_batches to display real-time evaluation progress
    original_iterate = loader.iterate_batches

    def progress_iterate_batches():
        total_samples = len(loader)
        processed_samples = 0
        for batch in original_iterate():
            batch_len = len(batch[1])
            processed_samples += batch_len
            print(f"Processing {processed_samples}/{total_samples} samples...", end="\r", flush=True)
            yield batch
        print()  # New line after progress complete

    loader.iterate_batches = progress_iterate_batches

    # 4. Execute BaselineEvaluator over 5,000 samples
    print("\n[3/3] Running BaselineEvaluator engine ...")
    evaluator = BaselineEvaluator(
        adapter=adapter,
        dataset_loader=loader,
        num_classes=num_classes,
        model_name="GTSRB_ViT_5000Sample_Baseline",
    )

    result = evaluator.evaluate(output_dir="results/baseline")
    elapsed_sec = time.time() - start_time

    # 5. Report Baseline Metrics
    print("\n" + "=" * 65)
    print("5,000-SAMPLE GTSRB BASELINE EVALUATION RESULTS")
    print("=" * 65)
    print(f"Dataset Identifier:        {result.dataset_name} (test[:{num_samples_target}])")
    print(f"Model Identifier:          {result.model_name}")
    print(f"Requested Samples:         {num_samples_target}")
    print(f"Evaluated Samples:         {result.num_samples}")
    print(f"Active Classes:            {result.num_classes} (44 logits sliced to 43)")
    print(f"Execution Device:          {result.device}")
    print(f"GPU Model:                 {gpu_model}")
    print(f"Execution Runtime:         {elapsed_sec:.2f} seconds")
    print("-" * 65)
    print(f"Accuracy:                  {result.accuracy * 100:.2f}% ({result.accuracy:.6f})")
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
    print("\n✅ 5,000-Sample GTSRB Baseline Evaluation Completed Successfully.")


if __name__ == "__main__":
    main()
