"""
AdverScan — End-to-End Interactive Security Assessment Demo & CLI.

Provides both an interactive prompt menu and command-line execution for Module 8 (Orchestration).
Enables practical demonstration of M1 Model Ingestion -> M2 Baseline Evaluation -> M3 Attack Engine
-> M2 Adversarial Evaluation -> M5 Vulnerability Analysis -> M6 XAI -> M7 Hardening -> M8 OrchestrationResult.
"""

import argparse
import json
import sys
import time
from typing import List, Tuple
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForImageClassification, ViTConfig

from app.attack_engine import discover_attacks, list_attacks
from app.evaluation.dataset_loader import GTSRBDatasetLoader
from app.orchestration import AdverScanOrchestrator, PipelineConfig
from app.utils import resolve_device, get_execution_device_info, load_gtsrb_vit_model, patch_hf_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="AdverScan M8 End-to-End Interactive Security Assessment Demo & CLI"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Force interactive mode menu prompt",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        default=False,
        help="Disable interactive menu prompt and use CLI arguments",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of samples to evaluate (default: 3)",
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["fgsm"],
        help="Attacks to execute (e.g. fgsm pgd deepfool)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["baseline_only", "attack_assessment", "full"],
        help="Execution mode (baseline_only, attack_assessment, full)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Execution device ('cpu' or 'cuda')",
    )
    parser.add_argument(
        "--enable-xai",
        action="store_true",
        default=False,
        help="Enable M6 XAI explainability",
    )
    parser.add_argument(
        "--xai-techniques",
        nargs="+",
        default=["shap"],
        help="XAI techniques to apply (shap, lime)",
    )
    parser.add_argument(
        "--enable-hardening",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable M7 Hardening Engine",
    )
    parser.add_argument(
        "--defense",
        type=str,
        default="spatial_smoothing",
        help="Hardening defense type (spatial_smoothing, randomized_smoothing, adversarial_training, auto)",
    )
    return parser.parse_args()


def prompt_user_selection():
    print("=" * 70)
    print("                 ADVERSCAN SECURITY ASSESSMENT DEMO                 ")
    print("=" * 70)

    # 1. Model Selection
    print("\nAvailable Models:")
    print("  1. bazyl/gtsrb-model (Vision Transformer - GTSRB Traffic Sign Classifier)")
    model_choice = input("Select model [1]: ").strip() or "1"
    if model_choice == "1":
        model_name = "bazyl/gtsrb-model"
    else:
        model_name = model_choice  # Allow custom HF model path string if supplied

    # 2. Dataset Selection
    print("\nAvailable Datasets:")
    print("  1. bazyl/GTSRB (German Traffic Sign Recognition Benchmark)")
    dataset_choice = input("Select dataset [1]: ").strip() or "1"
    if dataset_choice == "1":
        dataset_name = "bazyl/GTSRB"
    else:
        dataset_name = dataset_choice

    # 3. Sample Count
    while True:
        samples_str = input("\nNumber of samples to evaluate [3]: ").strip() or "3"
        try:
            samples_count = int(samples_str)
            if samples_count > 0:
                break
            print("  ❌ Please enter a positive integer greater than 0.")
        except ValueError:
            print("  ❌ Invalid integer. Please try again.")

    # 4. Discover Attacks from M3
    discover_attacks()
    available_attacks = list_attacks()  # ['deepfool', 'fgsm', 'pgd']
    print("\nAvailable Adversarial Attacks (M3 Attack Engine):")
    for idx, atk in enumerate(available_attacks, 1):
        print(f"  {idx}. {atk.upper()}")
    print("  M. Multiple / All attacks")

    atk_input = input("Select attack(s) (e.g. '1', '1 2', 'all') [1]: ").strip().lower() or "1"
    selected_attacks: List[str] = []

    if atk_input in ["m", "all", "all attacks"]:
        selected_attacks = available_attacks
    else:
        for token in atk_input.split():
            if token.isdigit():
                num = int(token)
                if 1 <= num <= len(available_attacks):
                    selected_attacks.append(available_attacks[num - 1])
            elif token in available_attacks:
                selected_attacks.append(token)

    if not selected_attacks:
        print("  ⚠️ No valid attack recognized; defaulting to FGSM.")
        selected_attacks = ["fgsm"]

    # 5. XAI Selection
    enable_xai_str = input("\nEnable M6 XAI Explainability? [y/N]: ").strip().lower()
    enable_xai = enable_xai_str in ["y", "yes", "true", "1"]
    selected_xai: List[str] = []
    if enable_xai:
        print("\nAvailable XAI Techniques:")
        print("  1. SHAP (SHapley Additive exPlanations)")
        print("  2. LIME (Local Interpretable Model-agnostic Explanations)")
        print("  3. Both (SHAP & LIME)")
        xai_choice = input("Select XAI technique [1]: ").strip() or "1"
        if xai_choice == "2":
            selected_xai = ["lime"]
        elif xai_choice == "3":
            selected_xai = ["shap", "lime"]
        else:
            selected_xai = ["shap"]

    # 6. Hardening Selection
    enable_hard_str = input("\nEnable M7 Hardening Engine? [y/N]: ").strip().lower()
    enable_hardening = enable_hard_str in ["y", "yes", "true", "1"]
    selected_defense = "auto"
    if enable_hardening:
        print("\nAvailable Hardening Defenses:")
        print("  1. Spatial Smoothing (spatial_smoothing)")
        print("  2. Bit Depth Reduction (bit_depth_reduction)")
        print("  3. JPEG Compression (jpeg_compression)")
        print("  4. Randomized Smoothing (randomized_smoothing)")
        print("  5. Adversarial Training (adversarial_training)")
        print("  6. Auto Selection (auto)")
        def_choice = input("Select defense [6]: ").strip() or "6"
        def_map = {
            "1": "spatial_smoothing",
            "2": "bit_depth_reduction",
            "3": "jpeg_compression",
            "4": "randomized_smoothing",
            "5": "adversarial_training",
            "6": "auto",
        }
        selected_defense = def_map.get(def_choice, "auto")

    # 7. Pipeline Execution Mode Selection
    print("\nExecution Modes:")
    print("  1. Baseline Only (M1 → M2)")
    print("  2. Attack Assessment (M1 → M2 → M3 → M2_Adv → M5)")
    print("  3. Full Security Assessment (M1 → M2 → M3 → M2_Adv → M5 → M6 / M7)")
    mode_choice = input("Select mode [3]: ").strip() or "3"
    mode_map = {
        "1": "baseline_only",
        "2": "attack_assessment",
        "3": "full",
    }
    selected_mode = mode_map.get(mode_choice, "full")

    # 10. Device Selection
    dev_info = get_execution_device_info("auto")
    print("\nTarget Execution Device:")
    if dev_info["gpu_available"]:
        print(f"  1. Auto-detect [CUDA / GPU ({dev_info['gpu_model']})] (Default)")
        print("  2. CPU")
    else:
        print("  1. Auto-detect [CPU (CUDA unavailable)] (Default)")
        print("  2. CPU")

    dev_choice = input("Select device [1]: ").strip() or "1"
    selected_device = "cpu" if dev_choice == "2" else resolve_device("auto")

    return {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "samples": samples_count,
        "attacks": selected_attacks,
        "enable_xai": enable_xai,
        "xai_techniques": selected_xai,
        "enable_hardening": enable_hardening,
        "defense": selected_defense,
        "mode": selected_mode,
        "device": selected_device,
    }


def print_summary_box(opts: dict):
    print("\n" + "=" * 70)
    print("                 ADVERSCAN CONFIGURATION SUMMARY                    ")
    print("=" * 70)
    print(f"Model Identifier:       {opts['model_name']}")
    print(f"Dataset Identifier:     {opts['dataset_name']}")
    print(f"Evaluated Samples:      {opts['samples']}")
    print(f"Target Device:          {opts['device']}")
    print(f"Pipeline Mode:          {opts['mode']}")
    print(f"Selected Attacks:       {', '.join(opts['attacks']).upper()}")
    print(f"XAI Enabled:            {opts['enable_xai']} {opts['xai_techniques'] if opts['enable_xai'] else ''}")
    print(f"Hardening Enabled:      {opts['enable_hardening']} (Defense: {opts['defense'] if opts['enable_hardening'] else 'N/A'})")
    print("=" * 70)


def main():
    args = parse_args()

    # Determine whether interactive mode should run
    # Interactive runs if no args are passed or if --interactive flag is explicitly passed
    is_interactive = args.interactive or (len(sys.argv) == 1)
    if args.no_interactive:
        is_interactive = False

    if is_interactive:
        opts = prompt_user_selection()
        print_summary_box(opts)
        confirm = input("Start security assessment? [Y/n]: ").strip().lower()
        if confirm in ["n", "no"]:
            print("  Assessment cancelled by user.")
            sys.exit(0)
    else:
        opts = {
            "model_name": "bazyl/gtsrb-model",
            "dataset_name": "bazyl/GTSRB",
            "samples": args.samples,
            "attacks": [a.lower() for a in args.attacks],
            "enable_xai": args.enable_xai,
            "xai_techniques": [t.lower() for t in args.xai_techniques],
            "enable_hardening": args.enable_hardening,
            "defense": args.defense.lower(),
            "mode": args.mode.lower(),
            "device": resolve_device(args.device),
        }
        print_summary_box(opts)

    # 1. Load HuggingFace PyTorch Model
    model_name = opts["model_name"]
    dataset_name = opts["dataset_name"]
    device = opts["device"]

    print(f"\n[1/3] Ingesting PyTorch model '{model_name}' on device '{device}'...")
    try:
        if model_name == "bazyl/gtsrb-model":
            raw_model, model_config = load_gtsrb_vit_model(model_name)
        else:
            config_path = hf_hub_download(model_name, "config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                cfg_dict = json.load(f)
            cfg_dict = patch_hf_config(cfg_dict)
            model_config = ViTConfig.from_dict(cfg_dict)
            raw_model = AutoModelForImageClassification.from_pretrained(
                model_name, config=model_config, use_safetensors=True
            )
    except Exception as e:
        print(f"  ❌ Error loading model '{model_name}': {e}")
        sys.exit(1)

    # 2. Setup Dataset Loader for sample count
    split_str = f"test[:{opts['samples']}]"
    print(f"[2/3] Loading dataset split '{split_str}' from '{dataset_name}'...")
    try:
        dataset_loader = GTSRBDatasetLoader(
            dataset_name=dataset_name,
            processor_name=model_name,
            split=split_str,
            batch_size=opts["samples"],
        )
    except Exception as e:
        print(f"  ❌ Error loading dataset '{dataset_name}': {e}")
        sys.exit(1)

    # 3. Build PipelineConfig & Execute AdverScanOrchestrator
    pipeline_config = PipelineConfig(
        model_path=raw_model,
        sample_input=torch.randn(1, 3, 224, 224),
        device=device,
        model_name="GTSRB_ViT_Demo",
        num_classes=43,
        mode=opts["mode"],
        attacks=opts["attacks"],
        enable_xai=opts["enable_xai"],
        xai_techniques=opts["xai_techniques"],
        enable_hardening=opts["enable_hardening"],
        defense=opts["defense"],
        enable_retest=opts["enable_hardening"],
        enable_report=True,
        custom_dataset_loader=dataset_loader,
    )

    print(f"[3/3] Executing AdverScanOrchestrator across pipeline (Mode: '{opts['mode']}')...\n")
    orchestrator = AdverScanOrchestrator()
    result = orchestrator.run(pipeline_config)

    # 4. Display Results Summary
    print("=" * 70)
    print("                 ADVERSCAN SECURITY ASSESSMENT RESULTS              ")
    print("=" * 70)
    print(f"Pipeline Status:          {result.status}")
    print(f"Execution Mode:           {result.execution_mode}")
    print(f"Total Execution Time:     {result.execution_time_seconds:.4f} seconds")
    print(f"Timestamp:                {result.timestamp}")

    print("\n------------------------------------------------------------")
    print("M2 — BASELINE CLEAN EVALUATION")
    print("------------------------------------------------------------")
    if result.baseline_evaluation:
        base_acc = result.baseline_evaluation.get("accuracy", 0.0)
        base_conf = result.baseline_evaluation.get("average_confidence", 0.0)
        f1_macro = result.baseline_evaluation.get("f1_macro", 0.0)
        print(f"Clean Accuracy:           {base_acc * 100:.2f}%")
        print(f"Clean Average Confidence: {base_conf * 100:.2f}%")
        print(f"Macro F1-Score:           {f1_macro * 100:.2f}%")
    else:
        print("Baseline evaluation result unavailable.")

    if result.attack_results:
        print("\n------------------------------------------------------------")
        print("M3 — ADVERSARIAL ATTACK ENGINE RESULTS")
        print("------------------------------------------------------------")
        for atk_name, atk_data in result.attack_results.items():
            print(f"[{atk_name.upper()}] Class: {atk_data.get('attack_class')}")
            print(f"    Execution Time:       {atk_data.get('execution_time_seconds', 0.0):.2f}s")
            print(f"    Parameters:           {atk_data.get('parameters', {})}")

    if result.vulnerability_analysis:
        print("\n------------------------------------------------------------")
        print("M5 — VULNERABILITY ANALYSIS & SCORING")
        print("------------------------------------------------------------")
        for atk_name, vuln in result.vulnerability_analysis.items():
            sc = vuln.get("scoring", {})
            ass = vuln.get("assessment", {})
            asr = ass.get("attack_success_rate")
            asr_str = f"{asr * 100:.2f}%" if asr is not None else "N/A"
            print(f"[{atk_name.upper()}] Vulnerability Score: {sc.get('vulnerability_score', 0.0):.2f} / 100")
            print(f"    Risk Level:           {sc.get('risk_level', 'UNKNOWN')}")
            print(f"    Model Degradation:    {ass.get('model_degradation', 0.0) * 100:.2f}%")
            print(f"    Attack Success Rate:  {asr_str}")
            print(f"    Accuracy Drop:        {ass.get('accuracy_drop', 0.0) * 100:.2f}%")

    print("\n------------------------------------------------------------")
    print("M6 — XAI EXPLAINABILITY")
    print("------------------------------------------------------------")
    if result.xai_results:
        for xai_key, xai_res in result.xai_results.items():
            print(f"[{xai_key.upper()}] Technique: {xai_res.get('technique')}")
            print(f"    Prediction Changed:   {xai_res.get('prediction_changed')}")
            print(f"    Attack Caused Failure:{xai_res.get('attack_caused_failure')}")
            print(f"    Clean Prediction:     {xai_res.get('clean_prediction')}")
            print(f"    Adversarial Pred:     {xai_res.get('adversarial_prediction')}")
    else:
        print("XAI: Disabled or not executed.")

    print("\n------------------------------------------------------------")
    print("M7 — HARDENING ENGINE")
    print("------------------------------------------------------------")
    if result.hardening_results:
        hard_meta = result.hardening_results.get("metadata", {})
        print(f"Applied Defense:          {hard_meta.get('defense_name')} ({hard_meta.get('defense_type')})")
        print(f"Hardening Success:        {result.hardening_results.get('success')}")
        print(f"Hardened Model Class:     {result.hardening_results.get('hardened_model_class')}")
        if result.hardening_results.get("recommendations"):
            print(f"Recommendations:          {result.hardening_results.get('recommendations')}")
    else:
        print("Hardening: Disabled or not executed.")

    print("\n------------------------------------------------------------")
    print("M8 — RE-TEST & COMPARISON ENGINE")
    print("------------------------------------------------------------")
    if result.retest_results:
        print(f"Overall Improved:         {result.retest_results.get('overall_improved')}")
        comparisons = result.retest_results.get("comparisons", {})
        for atk_k, comp in comparisons.items():
            print(f"[{atk_k.upper()}] Delta Vulnerability Score: {comp.get('delta_vulnerability_score', 0.0):.2f}")
            print(f"    Delta Accuracy Drop:  {comp.get('delta_accuracy_drop', 0.0) * 100:.2f}%")
            print(f"    Risk Level Before:    {comp.get('before_risk_level')}")
            print(f"    Risk Level After:     {comp.get('after_risk_level')}")
            print(f"    Is Vector Improved:   {comp.get('is_improved')}")
    else:
        print("Re-Test: Disabled or not executed.")

    print("\n------------------------------------------------------------")
    print("M9 — SECURITY REPORT GENERATOR")
    print("------------------------------------------------------------")
    if result.report_result:
        print(f"Report ID:                {result.report_result.get('report_id')}")
        print(f"Status:                   {result.report_result.get('status')}")
        print(f"Risk Level:               {result.report_result.get('risk_level')}")
        print(f"Vulnerability Score:      {result.report_result.get('vulnerability_score')}")
        if result.report_result.get("recommendations"):
            print("Recommendations:")
            for rec in result.report_result.get("recommendations", []):
                print(f"  - {rec}")
    else:
        print("Report Generator: Disabled or not executed.")

    if result.errors:
        print("\n------------------------------------------------------------")
        print("ERRORS / WARNINGS LOGGED")
        print("------------------------------------------------------------")
        for err in result.errors:
            print(f"❌ [{err.get('module')}] {err.get('error_type')}: {err.get('message')}")

    print("=" * 70)
    print("✅ AdverScan Security Assessment Demo Completed.")


if __name__ == "__main__":
    main()
