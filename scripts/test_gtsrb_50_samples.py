"""
Controlled Real-Data 50-Sample Integration Test for GTSRB Baseline Evaluation (Module 2).
Uses bazyl/GTSRB (test split, first 50 samples) and bazyl/gtsrb-model via Module 1 ingest_model().
Runs strictly on CPU.
"""

import json
import time
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForImageClassification, ViTConfig

from app.ingestion import ingest_model
from app.evaluation.dataset_loader import GTSRBDatasetLoader
from app.evaluation.evaluator import BaselineEvaluator


def main():
    print("=" * 65)
    print("AdverScan — 50-Sample GTSRB Baseline Evaluation Integration Test")
    print("=" * 65)

    model_name = "bazyl/gtsrb-model"
    dataset_name = "bazyl/GTSRB"
    num_classes = 43
    num_samples_target = 50

    # 1. Patch config (fixes upstream null in id2label['43']) & load cached HF ViT model
    print("\n[1/3] Ingesting Model via Module 1 (M1) contract on CPU ...")
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
    raw_model = AutoModelForImageClassification.from_pretrained(model_name, config=model_config, use_safetensors=True)

    # Ingest model through M1 contract
    sample_input = torch.randn(1, 3, 224, 224)
    adapter, metadata = ingest_model(
        model_path=raw_model,
        sample_input=sample_input,
        model_name="GTSRB_ViT_50Sample_Test",
        device="cpu",
    )

    print(f"M1 Ingestion Complete:")
    print(f"  Framework: {metadata.framework}")
    print(f"  Model Name: {metadata.model_name}")
    print(f"  Target Device: {adapter.device}")
    print(f"  Adapter Type: {adapter.__class__.__name__}")

    # 2. Initialize GTSRB dataset loader for exactly 50 samples from test split (batch_size=25)
    print(f"\n[2/3] Loading exactly {num_samples_target} samples from '{dataset_name}' (test split) ...")
    loader = GTSRBDatasetLoader(
        dataset_name=dataset_name,
        processor_name=model_name,
        split=f"test[:{num_samples_target}]",
        batch_size=25,
    )
    print(f"  Loaded dataset samples: {len(loader)}")

    # 3. Execute BaselineEvaluator over 50 samples
    print("\n[3/3] Running BaselineEvaluator engine ...")
    evaluator = BaselineEvaluator(
        adapter=adapter,
        dataset_loader=loader,
        num_classes=num_classes,
        model_name="GTSRB_ViT_50Sample_Baseline",
    )

    result = evaluator.evaluate(output_dir=None)
    elapsed_sec = time.time() - start_time

    # 4. Report Integration Test Metrics
    print("\n" + "=" * 65)
    print("50-SAMPLE GTSRB INTEGRATION TEST RESULTS")
    print("=" * 65)
    print(f"Dataset Identifier:        {result.dataset_name} (test[:50])")
    print(f"Model Identifier:          {result.model_name}")
    print(f"Evaluated Samples:         {result.num_samples}")
    print(f"Active Classes:            {result.num_classes} (44 logits sliced to 43)")
    print(f"Execution Device:          {result.device}")
    print(f"Execution Runtime:         {elapsed_sec:.2f} seconds")
    print("-" * 65)
    print(f"Accuracy:                  {result.accuracy * 100:.2f}% ({result.accuracy:.4f})")
    print(f"Macro Precision:           {result.precision_macro * 100:.2f}%")
    print(f"Macro Recall:              {result.recall_macro * 100:.2f}%")
    print(f"Macro F1-Score:            {result.f1_macro * 100:.2f}%")
    print(f"Weighted Precision:        {result.precision_weighted * 100:.2f}%")
    print(f"Weighted Recall:           {result.recall_weighted * 100:.2f}%")
    print(f"Weighted F1-Score:         {result.f1_weighted * 100:.2f}%")
    print(f"Average Confidence:        {result.average_confidence * 100:.2f}% ({result.average_confidence:.6f})")
    print(f"Average Entropy:           {result.average_entropy:.6f} bits")
    print(f"Confusion Matrix Shape:    {len(result.confusion_matrix)} × {len(result.confusion_matrix[0])}")
    print("=" * 65)
    print("\n✅ 50-Sample Real-Data Integration Test Completed Successfully.")


if __name__ == "__main__":
    main()
