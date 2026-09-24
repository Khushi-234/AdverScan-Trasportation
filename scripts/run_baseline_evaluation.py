"""
Execution script to perform clean Baseline Evaluation (Module 2) across multiple models (GTSRB, Intel Image Classification, custom HF models, and PyTorch checkpoints).
Supports interactive model selection from terminal and logs activity to results/ingestion_log.json.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import sys
import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from transformers import AutoModelForImageClassification, ViTConfig

# Ensure framework root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion import ingest_model
from app.ingestion.model_registry import ModelPresetRegistry, setup_model_environment
from app.evaluation import evaluate_baseline

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="AdverScan Baseline Evaluation Demo")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Target model preset ID, Hugging Face model ID, Kaggle dataset URL, or local checkpoint path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset identifier (e.g. 'bazyl/GTSRB', 'puneet6060/intel-image-classification')",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples to evaluate (default: 100)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Target execution device ('cpu' or 'cuda')",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable interactive terminal prompts and use default arguments",
    )
    return parser.parse_args()


def prompt_model_selection():
    print("=" * 65)
    print("           ADVERSCAN MULTI-MODEL SELECTION & INGESTION              ")
    print("=" * 65)

    presets = ModelPresetRegistry.list_presets()
    print("\nSelect target model for ingestion and security evaluation:")
    for idx, p in enumerate(presets, 1):
        d_domain = p.get("data_domain", "image")
        m_type = p.get("model_type", "Vision Transformer")
        print(f"  {idx}. {p['name']} ({p['model_id']})")
        print(f"     └─ Domain: [{d_domain.upper()}] {p.get('domain', '')}")
        print(f"     └─ Architecture: {m_type} | Recommended Dataset: {p.get('dataset_name', 'N/A')}")
    print(f"  {len(presets) + 1}. Custom Hugging Face Hub Model ID")
    print(f"  {len(presets) + 2}. Custom Local PyTorch Checkpoint (.pt / .pth file)")

    choice = input(f"\nEnter choice [1-{len(presets) + 2}] [1]: ").strip() or "1"

    if choice.isdigit() and 1 <= int(choice) <= len(presets):
        selected = presets[int(choice) - 1]
        model_id = selected["model_id"]
        dataset_name = selected["dataset_name"]
    elif choice == str(len(presets) + 1):
        model_id = input("Enter Hugging Face Model ID [bazyl/gtsrb-model]: ").strip() or "bazyl/gtsrb-model"
        dataset_name = input("Enter Hugging Face Dataset Name [bazyl/GTSRB]: ").strip() or "bazyl/GTSRB"
    elif choice == str(len(presets) + 2):
        model_id = input("Enter Local Checkpoint Path [gtsrb_model.pt]: ").strip() or "gtsrb_model.pt"
        dataset_name = input("Enter Evaluation Dataset Name [bazyl/GTSRB]: ").strip() or "bazyl/GTSRB"
    else:
        model_id = presets[0]["model_id"]
        dataset_name = presets[0]["dataset_name"]

    return model_id, dataset_name


def main():
    args = parse_args()

    if not args.non_interactive and args.model is None:
        model_id, dataset_name = prompt_model_selection()
    else:
        model_id = args.model or "bazyl/gtsrb-model"
        resolved_info = ModelPresetRegistry.resolve_model_info(model_id)
        dataset_name = args.dataset or resolved_info.get("dataset_name", "bazyl/GTSRB")

    device = args.device

    print(f"\n[1/3] Standardizing & Setting Environment for '{model_id}'...")
    env_info = setup_model_environment(
        model_id_or_path=model_id,
        device=device,
        custom_dataset_name=dataset_name,
        log_log_event=True,
    )
    print(f"  ✓ Target Data Domain: [{env_info['data_domain'].upper()}]")
    print(f"  ✓ Model Architecture: {env_info['model_type']}")
    print(f"  ✓ Dataset Recommended: {env_info['recommended_dataset']}")
    print(f"  ✓ Execution Device: {env_info['target_device']}")

    # Load model weights safely
    resolved_model_id = env_info["model_id"]
    try:
        config_path = hf_hub_download(resolved_model_id, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

        if "id2label" in cfg_dict and isinstance(cfg_dict["id2label"], dict):
            cfg_dict["id2label"] = {
                str(k): (str(v) if v is not None else "Unused")
                for k, v in cfg_dict["id2label"].items()
            }
            cfg_dict["label2id"] = {v: int(k) for k, v in cfg_dict["id2label"].items()}
            cfg_dict["num_labels"] = len(cfg_dict["id2label"])

        model_config = ViTConfig.from_dict(cfg_dict)
        raw_model = AutoModelForImageClassification.from_pretrained(resolved_model_id, config=model_config, use_safetensors=True)
    except Exception:
        raw_model = AutoModelForImageClassification.from_pretrained(resolved_model_id)

    # Ingest model into AdverScan framework
    sample_input = torch.randn(1, 3, 224, 224)
    adapter, metadata = ingest_model(
        model_path=raw_model,
        sample_input=sample_input,
        model_name=env_info["model_name"],
        device=device,
        domain=env_info["data_domain"],
        log_ingestion=True,
    )

    num_classes = metadata.num_classes or env_info.get("expected_num_classes") or 6

    print(f"\n[2/3] Running Clean Baseline Evaluation on '{dataset_name}' (test split, {args.samples} samples)...")
    split_str = f"test[:{args.samples}]" if args.samples > 0 else "test"

    result = evaluate_baseline(
        adapter=adapter,
        dataset_name=dataset_name,
        processor_name=resolved_model_id,
        split=split_str,
        batch_size=32,
        num_classes=num_classes,
        model_name=metadata.model_name,
        output_dir="results/baseline",
    )

    print("\n" + "=" * 65)
    print("              ADVERSCAN — BASELINE EVALUATION SUMMARY             ")
    print("=" * 65)
    print(f"Dataset Identifier:        {result.dataset_name} ({split_str})")
    print(f"Model Identifier:          {result.model_name}")
    print(f"Evaluated Samples:         {result.num_samples:,}")
    print(f"Number of Classes:         {result.num_classes}")
    print(f"Execution Device:          {result.device}")
    print("-" * 65)
    print(f"Clean Overall Accuracy:    {result.accuracy * 100:.2f}% ({result.accuracy:.4f})")
    print(f"Macro F1-Score:            {result.f1_macro * 100:.2f}% ({result.f1_macro:.4f})")
    print(f"Average Confidence:        {result.average_confidence * 100:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()

