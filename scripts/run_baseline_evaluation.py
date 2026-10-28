"""
Execution script to perform clean Baseline Evaluation (Module 2) across Image Classification models.
Supports interactive model selection from terminal, single image model evaluation, and evaluating registered image model presets sequentially.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from dotenv import load_dotenv

# Ensure framework root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion import ingest_model
from app.evaluation import evaluate_baseline
from app.evaluation.results import EvaluationResult
from transformers import AutoModelForImageClassification

load_dotenv()

IMAGE_MODEL_PRESETS: List[Dict[str, Any]] = [
    {
        "name": "GTSRB Traffic Sign Classifier (ViT)",
        "model_id": "bazyl/gtsrb-model",
        "dataset_name": "bazyl/GTSRB",
        "processor_name": "bazyl/gtsrb-model",
        "domain": "Transportation / Vision",
        "data_domain": "image",
        "model_type": "Vision Transformer (ViT)",
        "expected_num_classes": 43,
    },
    {
        "name": "Intel Image Classification (ViT)",
        "model_id": "puneet6060/intel-image-classification",
        "dataset_name": "puneet6060/intel-image-classification",
        "processor_name": "puneet6060/intel-image-classification",
        "domain": "Vision Classification",
        "data_domain": "image",
        "model_type": "Vision Transformer (ViT)",
        "expected_num_classes": 6,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="AdverScan Image Baseline Evaluation Demo")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Target Hugging Face model ID or local checkpoint path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Image dataset identifier (e.g. 'bazyl/GTSRB')",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of image samples to evaluate (default: 100)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Target execution device ('cpu' or 'cuda')",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all registered image model presets sequentially",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable interactive terminal prompts and use default arguments",
    )
    return parser.parse_args()


def prompt_model_selection() -> Tuple[str, str, str]:
    print("=" * 65)
    print("       ADVERSCAN IMAGE MODEL SELECTION & INGESTION          ")
    print("=" * 65)

    print("\nSelect target image model for ingestion and security evaluation:")
    for idx, p in enumerate(IMAGE_MODEL_PRESETS, 1):
        print(f"  {idx}. {p['name']} ({p['model_id']})")
        print(f"     └─ Domain: [IMAGE] {p['domain']}")
        print(f"     └─ Architecture: {p['model_type']} | Dataset: {p['dataset_name']}")
    print(f"  {len(IMAGE_MODEL_PRESETS) + 1}. Custom Hugging Face Image Model ID")
    print(f"  {len(IMAGE_MODEL_PRESETS) + 2}. Evaluate ALL Image Model Presets Sequentially")

    choice = input(f"\nEnter choice [1-{len(IMAGE_MODEL_PRESETS) + 2}] [1]: ").strip() or "1"

    if choice.isdigit() and 1 <= int(choice) <= len(IMAGE_MODEL_PRESETS):
        selected = IMAGE_MODEL_PRESETS[int(choice) - 1]
        model_id = selected["model_id"]
        dataset_name = selected["dataset_name"]
        processor_name = selected.get("processor_name", model_id)
    elif choice == str(len(IMAGE_MODEL_PRESETS) + 1):
        model_id = input("Enter Hugging Face Model ID [bazyl/gtsrb-model]: ").strip() or "bazyl/gtsrb-model"
        processor_name = input(f"Enter Image Processor ID [{model_id}]: ").strip() or model_id
        dataset_name = input("Enter Image Dataset Name [bazyl/GTSRB]: ").strip() or "bazyl/GTSRB"
    elif choice == str(len(IMAGE_MODEL_PRESETS) + 2) or choice.lower() == "all":
        return "__all__", "", ""
    else:
        model_id = IMAGE_MODEL_PRESETS[0]["model_id"]
        dataset_name = IMAGE_MODEL_PRESETS[0]["dataset_name"]
        processor_name = IMAGE_MODEL_PRESETS[0].get("processor_name", model_id)

    return model_id, dataset_name, processor_name


def run_single_evaluation(
    model_id: str,
    dataset_name: Optional[str] = None,
    processor_name: Optional[str] = None,
    samples: int = 100,
    device: str = "cpu",
    output_dir: str = "results/baseline",
) -> Optional[EvaluationResult]:
    dataset_name = dataset_name or "bazyl/GTSRB"
    processor_name = processor_name or model_id

    print(f"\n[1/3] Ingesting Image Model '{model_id}' on {device}...")
    raw_model = AutoModelForImageClassification.from_pretrained(model_id)
    sample_input = torch.randn(1, 3, 224, 224)

    adapter, metadata = ingest_model(
        model_path=raw_model,
        sample_input=sample_input,
        model_name=model_id.split("/")[-1],
        device=device,
    )

    num_classes = metadata.num_classes or getattr(raw_model.config, "num_labels", 43)

    print(f"  ✓ Target Data Domain: [IMAGE]")
    print(f"  ✓ Architecture:       {metadata.model_architecture}")
    print(f"  ✓ Total Parameters:   {metadata.total_parameters:,}" if metadata.total_parameters else "")
    print(f"  ✓ Execution Device:   {adapter.device}")

    print(f"\n[2/3] Running Clean Baseline Evaluation on '{dataset_name}' (test split, {samples} samples)...")
    split_str = f"test[:{samples}]" if samples > 0 else "test"

    result = evaluate_baseline(
        adapter=adapter,
        dataset_name=dataset_name,
        processor_name=processor_name,
        split=split_str,
        batch_size=32,
        num_classes=num_classes,
        model_name=metadata.model_name,
        output_dir=output_dir,
        log_mlflow=False,
    )

    print("\n" + "=" * 70)
    print("              ADVERSCAN — BASELINE EVALUATION SUMMARY             ")
    print("=" * 70)
    print("  MODEL SPECIFICATIONS")
    print(f"  • Model Identifier:        {result.model_name}")
    if result.model_architecture:
        print(f"  • Architecture:            {result.model_architecture}")
    print(f"  • Ingestion Framework:     huggingface")
    print(f"  • Execution Device:        {result.device}")
    print("-" * 70)
    print("  DATASET & TASK DETAILS")
    print(f"  • Dataset Identifier:      {result.dataset_name} ({split_str})")
    print(f"  • Data Domain:             IMAGE")
    print(f"  • Total Evaluated Samples: {result.num_samples:,}")
    print(f"  • Total Number of Classes: {result.num_classes}")
    print(f"  • Evaluation Batch Size:   {result.batch_size}")
    print("-" * 70)
    print("  CLEAN PERFORMANCE BASELINE")
    print(f"  • Clean Overall Accuracy:  {result.accuracy * 100:.2f}%")
    print(f"  • Macro F1-Score:          {result.f1_macro * 100:.2f}%")
    print(f"  • Weighted F1-Score:       {result.f1_weighted * 100:.2f}%")
    print(f"  • Average Confidence:      {result.average_confidence * 100:.2f}%")
    print(f"  • Average Entropy:         {result.average_entropy:.4f}")
    print("=" * 70)

    return result


def main():
    args = parse_args()

    model_id: str = "bazyl/gtsrb-model"
    dataset_name: Optional[str] = "bazyl/GTSRB"
    processor_name: Optional[str] = "bazyl/gtsrb-model"
    evaluate_all = False

    if args.all:
        evaluate_all = True
    elif not args.non_interactive and args.model is None:
        p_model, p_dataset, p_proc = prompt_model_selection()
        evaluate_all = (p_model == "__all__")
        if not evaluate_all:
            model_id = p_model
            dataset_name = p_dataset
            processor_name = p_proc
    else:
        model_id = args.model or "bazyl/gtsrb-model"
        dataset_name = args.dataset or "bazyl/GTSRB"
        processor_name = model_id

    device = args.device

    if evaluate_all:
        print("\n" + "=" * 80)
        print(f"   STARTING SEQUENTIAL BASELINE EVALUATION ACROSS ALL {len(IMAGE_MODEL_PRESETS)} IMAGE PRESETS")
        print("=" * 80)
        all_results = []
        for idx, preset in enumerate(IMAGE_MODEL_PRESETS, 1):
            p_id = preset["model_id"]
            p_name = preset["name"]
            d_name = preset["dataset_name"]
            proc_name = preset.get("processor_name", p_id)
            print(f"\n>>> [{idx}/{len(IMAGE_MODEL_PRESETS)}] Evaluating: {p_name} ({p_id}) [IMAGE]")
            try:
                res = run_single_evaluation(
                    model_id=p_id,
                    dataset_name=d_name,
                    processor_name=proc_name,
                    samples=args.samples,
                    device=device,
                )
                acc_val = f"{res.accuracy * 100:.2f}%" if res is not None and res.accuracy is not None else "N/A"
                f1_val = f"{res.f1_macro * 100:.2f}%" if res is not None and res.f1_macro is not None else "N/A"
                conf_val = f"{res.average_confidence * 100:.2f}%" if res is not None and res.average_confidence is not None else "N/A"
                all_results.append({
                    "model_id": p_id,
                    "name": p_name,
                    "domain": "IMAGE",
                    "dataset": d_name,
                    "status": "SUCCESS" if res is not None else "FAILED",
                    "accuracy": acc_val,
                    "f1_macro": f1_val,
                    "confidence": conf_val,
                })
            except Exception as e:
                print(f"❌ Evaluation FAILED for {p_id}: {e}")
                all_results.append({
                    "model_id": p_id,
                    "name": p_name,
                    "domain": "IMAGE",
                    "dataset": d_name,
                    "status": "FAILED",
                    "accuracy": "N/A",
                    "f1_macro": "N/A",
                    "confidence": "N/A",
                })

        print("\n" + "=" * 90)
        print("            ADVERSCAN — ALL IMAGE MODEL PRESETS EVALUATION REPORT                 ")
        print("=" * 90)
        header = f"{'Preset Model ID':<35} | {'Domain':<6} | {'Status':<7} | {'Accuracy':<9} | {'F1-Macro':<9} | {'Avg Conf':<8}"
        print(header)
        print("-" * len(header))
        for r in all_results:
            status_icon = "✓" if r["status"] == "SUCCESS" else "✗"
            m_id_str = str(r.get("model_id") or "")
            print(f"{m_id_str[:35]:<35} | IMAGE  | {status_icon} {r['status']:<5} | {r['accuracy']:<9} | {r['f1_macro']:<9} | {r['confidence']:<8}")
        print("=" * 90)
        return

    # Single model evaluation
    run_single_evaluation(
        model_id=model_id,
        dataset_name=dataset_name,
        processor_name=processor_name,
        samples=args.samples,
        device=device,
    )


if __name__ == "__main__":
    main()
