"""
Execution script to perform clean Baseline Evaluation (Module 2) across multiple models (GTSRB, Intel Image Classification, custom HF models, and PyTorch checkpoints).
Supports interactive model selection from terminal, single model evaluation, and evaluating all MODEL_PRESETS sequentially.
"""

import argparse
import sys
from pathlib import Path
import torch
from dotenv import load_dotenv

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
        "--all",
        action="store_true",
        help="Evaluate all registered MODEL_PRESETS sequentially and output a summary comparison table",
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
        print(f"     └─ Preprocessing: {p.get('processor_name', p['model_id'])}")
    print(f"  {len(presets) + 1}. Custom Hugging Face Hub Model ID")
    print(f"  {len(presets) + 2}. Custom Local PyTorch Checkpoint (.pt / .pth file)")
    print(f"  {len(presets) + 3}. Evaluate ALL Model Presets Sequentially")

    choice = input(f"\nEnter choice [1-{len(presets) + 3}] [1]: ").strip() or "1"

    if choice.isdigit() and 1 <= int(choice) <= len(presets):
        selected = presets[int(choice) - 1]
        model_id = selected["model_id"]
        dataset_name = selected["dataset_name"]
        processor_name = selected.get("processor_name", model_id)
        data_domain = selected.get("data_domain", "image")
    elif choice == str(len(presets) + 1):
        model_id = input("Enter Hugging Face Model ID [bazyl/gtsrb-model]: ").strip() or "bazyl/gtsrb-model"
        processor_name = input(f"Enter Processor / Tokenizer ID [{model_id}]: ").strip() or model_id

        # Determine domain or prompt user
        from app.ingestion.model_registry import detect_model_domain
        detected_domain = detect_model_domain(model_id, processor_name=processor_name)
        if detected_domain in ("image", "text", "time-series", "tabular"):
            domain_choice = input(f"Enter Data Domain ('image', 'text', 'time-series', or 'tabular') [{detected_domain}]: ").strip().lower() or detected_domain
        else:
            domain_choice = input("Enter Data Domain ('image', 'text', 'time-series', or 'tabular') [image]: ").strip().lower() or "image"
        data_domain = domain_choice if domain_choice in ("image", "text", "time-series", "tabular") else "image"

        from app.evaluation.dataset_loader.factory import resolve_dataset_file
        if data_domain == "tabular":
            default_dataset = resolve_dataset_file(None, [".csv", ".tsv"], model_identifier=model_id) or "data/test.csv"
        elif data_domain == "time-series":
            default_dataset = resolve_dataset_file(None, [".pt", ".pth", ".npz"], model_identifier=model_id) or "data/test.pt"
        elif data_domain == "text":
            default_dataset = "imdb"
        else:
            default_dataset = "bazyl/GTSRB"
        dataset_name = input(f"Enter Evaluation Dataset Name / Path [{default_dataset}]: ").strip() or default_dataset
    elif choice == str(len(presets) + 2):
        model_id = input("Enter Local Checkpoint Path [gtsrb_model.pt]: ").strip() or "gtsrb_model.pt"
        processor_name = None
        from app.ingestion.model_registry import detect_model_domain
        detected_domain = detect_model_domain(model_id)
        default_dom = detected_domain if detected_domain in ("image", "text", "time-series", "tabular") else "image"
        domain_choice = input(f"Enter Data Domain ('image', 'text', 'time-series', or 'tabular') [{default_dom}]: ").strip().lower() or default_dom
        data_domain = domain_choice if domain_choice in ("image", "text", "time-series", "tabular") else "image"

        from app.evaluation.dataset_loader.factory import resolve_dataset_file
        if data_domain == "tabular":
            default_dataset = resolve_dataset_file(None, [".csv", ".tsv"], model_identifier=model_id) or "data/test.csv"
        elif data_domain == "time-series":
            default_dataset = resolve_dataset_file(None, [".pt", ".pth", ".npz"], model_identifier=model_id) or "data/test.pt"
        elif data_domain == "text":
            default_dataset = "imdb"
        else:
            default_dataset = "bazyl/GTSRB"
        dataset_name = input(f"Enter Evaluation Dataset Name / Path [{default_dataset}]: ").strip() or default_dataset
    elif choice == str(len(presets) + 3) or choice.lower() == "all":
        return "__all__", None, None, None
    else:
        model_id = presets[0]["model_id"]
        dataset_name = presets[0]["dataset_name"]
        processor_name = presets[0].get("processor_name", model_id)
        data_domain = presets[0].get("data_domain", "image")

    return model_id, dataset_name, processor_name, data_domain


def run_single_evaluation(
    model_id: str,
    dataset_name: str,
    processor_name: str,
    data_domain: str,
    samples: int = 100,
    device: str = "cpu",
    output_dir: str = "results/baseline",
):
    print(f"\n[1/3] Standardizing & Setting Environment for '{model_id}'...")
    env_info = setup_model_environment(
        model_id_or_path=model_id,
        device=device,
        custom_dataset_name=dataset_name,
        custom_processor_name=processor_name,
        data_domain=data_domain,
        log_log_event=True,
    )
    print(f"  ✓ Target Data Domain: [{env_info['data_domain'].upper()}]")
    print(f"  ✓ Model Architecture: {env_info['model_type']}")
    print(f"  ✓ Dataset Recommended: {env_info['recommended_dataset']}")
    print(f"  ✓ Preprocessor Config: {env_info['processor_name']}")
    print(f"  ✓ Execution Device: {env_info['target_device']}")

    # Ingest model into AdverScan framework dynamically
    resolved_model_id = env_info["model_id"]
    domain = str(env_info.get("data_domain", "")).lower().replace("_", "-")
    if domain == "image":
        sample_input = torch.randn(1, 3, 224, 224)
    elif domain == "time-series":
        seq_len = env_info.get("sequence_length") or 14
        num_ch = env_info.get("num_channels") or 1
        if "patchtst" in resolved_model_id.lower() or "patchtsmixer" in resolved_model_id.lower() or "transformer" in str(env_info.get("model_type", "")).lower():
            sample_input = torch.randn(1, seq_len, num_ch)
        else:
            sample_input = torch.randn(1, num_ch, seq_len)
    elif domain == "tabular":
        num_feats = env_info.get("num_features") or 14
        sample_input = torch.randn(1, num_feats)
    else:
        sample_input = "AdverScan clean baseline evaluation sample sentence."

    from app.ingestion.exceptions import UnsafeModelError
    try:
        adapter, metadata = ingest_model(
            model_path=resolved_model_id,
            sample_input=sample_input,
            model_name=env_info["model_name"],
            device=device,
            data_domain=env_info.get("data_domain"),
            dataset_name=dataset_name,
            processor_name=env_info.get("processor_name"),
            log_ingestion=True,
        )
    except UnsafeModelError as ume:
        print(f"\n[SECURITY WARNING] {ume}")
        confirm = input("\nThis model requires deserializing pickle/joblib files. Do you want to proceed? [y/N]: ").strip().lower()
        if confirm in ("y", "yes"):
            adapter, metadata = ingest_model(
                model_path=resolved_model_id,
                sample_input=sample_input,
                model_name=env_info["model_name"],
                device=device,
                data_domain=env_info.get("data_domain"),
                dataset_name=dataset_name,
                processor_name=env_info.get("processor_name"),
                unsafe_allow_pickle=True,
                log_ingestion=True,
            )
        else:
            print("\nAborted model loading due to security policy.")
            return None
    except Exception:
        if domain == "time-series" and isinstance(sample_input, torch.Tensor) and sample_input.ndim == 3:
            sample_input = sample_input.transpose(1, 2)
            adapter, metadata = ingest_model(
                model_path=resolved_model_id,
                sample_input=sample_input,
                model_name=env_info["model_name"],
                device=device,
                data_domain=env_info.get("data_domain"),
                dataset_name=dataset_name,
                processor_name=env_info.get("processor_name"),
                log_ingestion=True,
            )
        else:
            raise

    if metadata.total_parameters is not None:
        train_p = metadata.trainable_parameters or metadata.total_parameters
        pct = (train_p / metadata.total_parameters * 100) if metadata.total_parameters > 0 else 100.0
        print(f"• Model Architecture:   {metadata.model_architecture}")
        print(f"• Total Parameters:     {metadata.total_parameters:,} (~{metadata.total_parameters / 1e6:.2f}M)")
        print(f"• Trainable Parameters: {train_p:,} ({pct:.1f}%)")

    num_classes = metadata.num_classes or env_info.get("expected_num_classes") or 6

    print(f"\n[2/3] Running Clean Baseline Evaluation on '{dataset_name}' (test split, {samples} samples)...")
    split_str = f"test[:{samples}]" if samples > 0 else "test"

    eval_kwargs = {}
    from pathlib import Path

    from app.evaluation.dataset_loader.factory import resolve_dataset_file

    if domain == "tabular":
        data_file = resolve_dataset_file(dataset_name, [".csv", ".tsv"], model_identifier=model_id)

        if data_file:
            print(f"  ✓ Dynamically resolved tabular dataset: '{data_file}'")
            eval_kwargs["data"] = data_file
            eval_kwargs["target_col"] = "target"
        else:
            print("  [INFO] No local CSV dataset path found; evaluating on synthetic sample inputs.")
            num_eval_samples = samples if samples > 0 else 100
            input_shape_eval = (num_eval_samples,) + tuple(sample_input.shape[1:])
            eval_inputs = torch.randn(input_shape_eval)
            eval_targets = torch.randint(0, num_classes, (num_eval_samples,))
            eval_kwargs["data"] = (eval_inputs, eval_targets)

    elif domain == "time-series":
        ts_file = resolve_dataset_file(dataset_name, [".pt", ".pth", ".npz", ".npy"], model_identifier=model_id)

        if ts_file:
            print(f"  ✓ Dynamically resolved time-series dataset: '{ts_file}'")
            loaded = torch.load(ts_file, map_location="cpu", weights_only=False)
            if isinstance(loaded, dict):
                x_ts = loaded.get("inputs", loaded.get("X"))
                y_ts = loaded.get("targets", loaded.get("y"))
            elif isinstance(loaded, tuple) and len(loaded) == 2:
                x_ts, y_ts = loaded
            else:
                x_ts, y_ts = loaded, None

            if samples > 0 and len(x_ts) > samples:
                x_ts = x_ts[:samples]
                if y_ts is not None:
                    y_ts = y_ts[:samples]
            eval_kwargs["inputs"] = x_ts
            eval_kwargs["targets"] = y_ts
        else:
            print("  [INFO] No local tensor dataset path found; evaluating on synthetic sequence samples.")
            num_eval_samples = samples if samples > 0 else 100
            input_shape_eval = (num_eval_samples,) + tuple(sample_input.shape[1:])
            eval_inputs = torch.randn(input_shape_eval)
            eval_targets = torch.randint(0, num_classes, (num_eval_samples,))
            eval_kwargs["inputs"] = eval_inputs
            eval_kwargs["targets"] = eval_targets

    result = evaluate_baseline(
        adapter=adapter,
        dataset_name=dataset_name,
        processor_name=env_info.get("processor_name") or model_id,
        data_domain=env_info.get("data_domain", "image"),
        split=split_str,
        batch_size=32,
        num_classes=num_classes,
        model_name=metadata.model_name,
        output_dir=output_dir,
        **eval_kwargs,
    )

    print("\n" + "=" * 70)
    print("              ADVERSCAN — BASELINE EVALUATION SUMMARY             ")
    print("=" * 70)
    print("  MODEL SPECIFICATIONS")
    print(f"  • Model Identifier:        {result.model_name}")
    if result.model_architecture:
        print(f"  • Architecture:            {result.model_architecture}")
    if result.total_parameters is not None:
        total_p = result.total_parameters
        train_p = result.trainable_parameters if result.trainable_parameters is not None else total_p
        train_pct = (train_p / total_p * 100) if total_p > 0 else 100.0
        print(f"  • Total Parameters:        {total_p:,} (~{total_p / 1e6:.2f}M)")
        print(f"  • Trainable Parameters:    {train_p:,} ({train_pct:.1f}%)")
        print(f"  • Estimated Model Size:    ~{total_p * 4 / (1024 * 1024):.1f} MB (FP32)")
    print(f"  • Execution Device:        {result.device}")
    print("-" * 70)
    print("  DATASET & TASK DETAILS")
    print(f"  • Dataset Identifier:      {result.dataset_name} ({split_str})")
    print(f"  • Data Domain:             {env_info.get('data_domain', 'image')}")
    print(f"  • Total Evaluated Samples: {result.num_samples:,}")
    print(f"  • Total Number of Classes: {result.num_classes}")
    if result.class_names:
        classes_str = ", ".join(result.class_names[:8])
        if len(result.class_names) > 8:
            classes_str += f" ... (+{len(result.class_names) - 8} more)"
        print(f"  • Class Labels:            {classes_str}")
    print(f"  • Evaluation Batch Size:   {result.batch_size}")
    print("-" * 70)
    print("  CLEAN PERFORMANCE BASELINE")
    print(f"  • Clean Overall Accuracy:  {result.accuracy * 100:.2f}% ({result.accuracy:.4f})")
    print(f"  • Macro F1-Score:          {result.f1_macro * 100:.2f}% ({result.f1_macro:.4f})")
    print(f"  • Weighted F1-Score:       {result.f1_weighted * 100:.2f}% ({result.f1_weighted:.4f})")
    print(f"  • Average Confidence:      {result.average_confidence * 100:.2f}%")
    print(f"  • Average Entropy:         {result.average_entropy:.4f}")
    print("=" * 70)

    return result


def main():
    args = parse_args()

    if args.all:
        evaluate_all = True
    elif not args.non_interactive and args.model is None:
        model_id, dataset_name, processor_name, data_domain = prompt_model_selection()
        evaluate_all = (model_id == "__all__")
    else:
        evaluate_all = False
        model_id = args.model or "bazyl/gtsrb-model"
        resolved_info = ModelPresetRegistry.resolve_model_info(model_id)
        dataset_name = args.dataset or resolved_info.get("dataset_name", "bazyl/GTSRB")
        processor_name = resolved_info.get("processor_name", model_id)
        data_domain = resolved_info.get("data_domain")

    device = args.device

    if evaluate_all:
        presets = ModelPresetRegistry.list_presets()
        print("\n" + "=" * 80)
        print(f"   STARTING SEQUENTIAL BASELINE EVALUATION ACROSS ALL {len(presets)} MODEL PRESETS")
        print("=" * 80)
        all_results = []
        for idx, preset in enumerate(presets, 1):
            p_id = preset["model_id"]
            p_name = preset.get("name", p_id)
            d_name = preset.get("dataset_name")
            proc_name = preset.get("processor_name", p_id)
            dom = preset.get("data_domain", "image")
            print(f"\n>>> [{idx}/{len(presets)}] Evaluating: {p_name} ({p_id}) [{dom.upper()}]")
            try:
                res = run_single_evaluation(
                    model_id=p_id,
                    dataset_name=d_name,
                    processor_name=proc_name,
                    data_domain=dom,
                    samples=args.samples,
                    device=device,
                )
                all_results.append({
                    "model_id": p_id,
                    "name": p_name,
                    "domain": dom,
                    "dataset": d_name,
                    "status": "SUCCESS",
                    "accuracy": f"{res.accuracy * 100:.2f}%",
                    "f1_macro": f"{res.f1_macro * 100:.2f}%",
                    "confidence": f"{res.average_confidence * 100:.2f}%",
                    "error": None,
                })
            except Exception as e:
                print(f"❌ Evaluation FAILED for {p_id}: {e}")
                all_results.append({
                    "model_id": p_id,
                    "name": p_name,
                    "domain": dom,
                    "dataset": d_name,
                    "status": "FAILED",
                    "accuracy": "N/A",
                    "f1_macro": "N/A",
                    "confidence": "N/A",
                    "error": str(e),
                })

        print("\n" + "=" * 90)
        print("                 ADVERSCAN — ALL MODEL PRESETS EVALUATION REPORT                  ")
        print("=" * 90)
        header = f"{'Preset Model ID':<35} | {'Domain':<6} | {'Status':<7} | {'Accuracy':<9} | {'F1-Macro':<9} | {'Avg Conf':<8}"
        print(header)
        print("-" * len(header))
        for r in all_results:
            status_icon = "✓" if r["status"] == "SUCCESS" else "✗"
            print(f"{r['model_id'][:35]:<35} | {r['domain'][:6].upper():<6} | {status_icon} {r['status']:<5} | {r['accuracy']:<9} | {r['f1_macro']:<9} | {r['confidence']:<8}")
        print("=" * 90)
        return

    # Single model evaluation
    run_single_evaluation(
        model_id=model_id,
        dataset_name=dataset_name,
        processor_name=processor_name,
        data_domain=data_domain,
        samples=args.samples,
        device=device,
    )


if __name__ == "__main__":
    main()
