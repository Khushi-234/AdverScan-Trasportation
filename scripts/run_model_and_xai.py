"""
AdverScan — Step-by-Step Model Run and XAI Pipeline Execution Script.
Demonstrates end-to-end model ingestion, inference, adversarial attack, and XAI profiling
with explicit step-by-step progress and measurement (explain_seconds and num_model_queries).
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.explainability.config import XAIConfig
from app.explainability.explainer import XAIExplainer
from app.explainability.explanation_result import ExplanationResult
from app.explainability.modality import Modality
from app.ingestion import ingest_model


class DemoVisionClassifier(nn.Module):
    """
    Lightweight, fast Convolutional Neural Network for image classification.
    Used for instant local execution and demonstration without heavy downloading.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_block(x)
        return self.classifier(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AdverScan — Step-by-Step Model Run & XAI Profiling Execution"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="demo",
        choices=["demo", "gtsrb"],
        help="Target model: 'demo' (fast local CNN) or 'gtsrb' (GTSRB ViT Transformer)",
    )
    parser.add_argument(
        "--technique",
        type=str,
        default="saliency",
        choices=["saliency", "gradcam", "integrated_gradients", "shap", "lime"],
        help="XAI attribution technique identifier",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="fast",
        choices=["fast", "standard", "thorough"],
        help="XAI evaluation mode preset (Step 2: XAIConfig): 'fast', 'standard', or 'thorough' (default: fast)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Prediction chunking batch size (Step 3: predict_batched) (default: 64)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Adversarial perturbation budget epsilon (default: 0.05)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/xai/model_run_and_xai_summary.json",
        help="Destination JSON path for explanation result",
    )
    return parser.parse_args()


def load_target_model(model_choice: str, device: str) -> Tuple[Any, Any, str]:
    """Load or initialize target model and return (adapter, raw_model, model_name)."""
    if model_choice == "gtsrb":
        print("  -> Loading Hugging Face GTSRB Vision Transformer model...")
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForImageClassification, ViTConfig

        model_name = "bazyl/gtsrb-model"
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
        sample_input = torch.randn(1, 3, 224, 224)
        adapter, _ = ingest_model(
            model_path=raw_model,
            sample_input=sample_input,
            model_name="GTSRB_ViT_Classifier",
            device=device,
        )
        return adapter, raw_model, "GTSRB ViT Classifier"
    else:
        # Fast demo CNN
        torch.manual_seed(42)
        raw_model = DemoVisionClassifier(in_channels=3, num_classes=10)
        raw_model.eval()
        sample_input = torch.randn(1, 3, 32, 32)
        adapter, _ = ingest_model(
            model_path=raw_model,
            sample_input=sample_input,
            model_name="Demo_Vision_CNN",
            device=device,
        )
        return adapter, raw_model, "Demo Fast Vision CNN"


def generate_adversarial_sample(
    model: nn.Module,
    clean_sample: torch.Tensor,
    epsilon: float,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    """Generate FGSM adversarial sample targeting the clean predicted class."""
    clean_sample = clean_sample.to(device).clone().detach().requires_grad_(True)
    outputs = model(clean_sample)
    if hasattr(outputs, "logits"):
        outputs = outputs.logits

    clean_pred_idx = int(torch.argmax(outputs[0]).item())
    target = torch.tensor([clean_pred_idx], device=device)

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs, target)
    model.zero_grad()
    loss.backward()

    # Fast Gradient Sign Method (FGSM)
    gradient = clean_sample.grad.data
    adversarial_sample = clean_sample + epsilon * gradient.sign()
    adversarial_sample = torch.clamp(adversarial_sample, -3.0, 3.0).detach()

    return adversarial_sample, clean_pred_idx


def main() -> None:
    args = parse_args()

    # Device configuration
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    print("\n" + "=" * 70)
    print("  ADVERSCAN — MODEL RUN & XAI EXPLANATION PIPELINE")
    print("=" * 70)
    print(f"  Execution Time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device Selected:   {device}")
    print(f"  Model Type:        {args.model.upper()}")
    print(f"  XAI Technique:     {args.technique.upper()}")
    print(f"  Epsilon Budget:    {args.epsilon}")
    print("=" * 70)

    # -------------------------------------------------------------
    # [STEP 1/5] Ingesting Model & Validating Architecture
    # -------------------------------------------------------------
    print("\n[STEP 1/5] Ingesting & Preparing Model...")
    t0_step1 = time.time()
    adapter, raw_model, model_title = load_target_model(args.model, device_str)
    raw_model = raw_model.to(device).eval()
    num_params = sum(p.numel() for p in raw_model.parameters())
    print(f"  ✅ Model Loaded:       {model_title}")
    print(f"  ✅ Model Parameters:   {num_params:,}")
    print(f"  ✅ Target Device:      {device}")
    print(f"  ⏱️ Step 1 Duration:    {time.time() - t0_step1:.4f}s")

    # -------------------------------------------------------------
    # [STEP 2/5] Preparing Baseline & Adversarial Inputs
    # -------------------------------------------------------------
    print("\n[STEP 2/5] Preparing Clean & Adversarial Input Samples...")
    t0_step2 = time.time()
    input_size = (1, 3, 224, 224) if args.model == "gtsrb" else (1, 3, 32, 32)
    torch.manual_seed(1337)
    clean_sample = torch.randn(*input_size, device=device)

    # Compute adversarial perturbation using FGSM
    adv_sample, clean_true_idx = generate_adversarial_sample(
        raw_model, clean_sample, epsilon=args.epsilon, device=device
    )

    l_inf = float(torch.max(torch.abs(clean_sample - adv_sample)).item())
    l_2 = float(torch.norm(clean_sample - adv_sample).item())
    print(f"  ✅ Input Shape:        {list(clean_sample.shape)}")
    print(f"  ✅ L_inf Perturbation: {l_inf:.6f}")
    print(f"  ✅ L_2 Perturbation:   {l_2:.6f}")
    print(f"  ⏱️ Step 2 Duration:    {time.time() - t0_step2:.4f}s")

    # -------------------------------------------------------------
    # [STEP 3/5] Running Model Inference (Clean vs. Adversarial)
    # -------------------------------------------------------------
    print("\n[STEP 3/5] Running Model Inference & Predictions...")
    t0_step3 = time.time()
    with torch.no_grad():
        clean_out = raw_model(clean_sample)
        if hasattr(clean_out, "logits"):
            clean_out = clean_out.logits
        clean_probs = torch.softmax(clean_out[0], dim=-1)
        clean_pred = int(torch.argmax(clean_probs).item())
        clean_conf = float(clean_probs[clean_pred].item())

        adv_out = raw_model(adv_sample)
        if hasattr(adv_out, "logits"):
            adv_out = adv_out.logits
        adv_probs = torch.softmax(adv_out[0], dim=-1)
        adv_pred = int(torch.argmax(adv_probs).item())
        adv_conf = float(adv_probs[adv_pred].item())

    pred_changed = clean_pred != adv_pred
    print(f"  📊 Clean Prediction:       Class {clean_pred} (Confidence: {clean_conf * 100:.2f}%)")
    print(f"  📊 Adversarial Prediction:   Class {adv_pred} (Confidence: {adv_conf * 100:.2f}%)")
    print(f"  ⚠️  Prediction Flipped:      {pred_changed} {'(ATTACK SUCCESSFUL!)' if pred_changed else '(Prediction Retained)'}")
    print(f"  ⏱️ Step 3 Duration:        {time.time() - t0_step3:.4f}s")

    # -------------------------------------------------------------
    # [STEP 4/5] Executing XAI Explanation with Profiling & Batching
    # -------------------------------------------------------------
    print("\n[STEP 4/5] Executing XAI Explanation (Step 1 Profiling + Step 2 Config + Step 3 Batching)...")
    xai_cfg = XAIConfig(mode=args.mode, batch_size=args.batch_size)
    print(f"  -> Technique:       {args.technique.upper()}")
    print(f"  -> Modality:        IMAGE")
    print(f"  -> Config Mode:     {xai_cfg.mode.upper()}")
    print(f"  -> IG Steps:        {xai_cfg.ig_steps}")
    print(f"  -> LIME Samples:    {xai_cfg.lime_num_samples}")
    print(f"  -> SHAP Max Evals:  {xai_cfg.shap_max_evals}")
    print(f"  -> Batch Size:      {xai_cfg.batch_size}")
    print("  -> Measuring runtime and model queries via explainability.profiling...")

    explainer = XAIExplainer(default_technique=args.technique, config=xai_cfg)
    t0_xai = time.time()

    # Call wrapped XAIExplainer.explain (internally profiled)
    result: ExplanationResult = explainer.explain(
        model=raw_model,
        clean_input=clean_sample,
        adversarial_input=adv_sample,
        true_label=clean_true_idx,
        technique=args.technique,
        attack_name=f"fgsm_eps_{args.epsilon}",
        modality=Modality.IMAGE,
        compute_metrics=False,
    )
    total_wall_time = time.time() - t0_xai

    # Extract profiling metrics from ExplanationResult.metadata
    explain_seconds = result.metadata.get("explain_seconds", 0.0)
    num_model_queries = result.metadata.get("num_model_queries", 0)
    clean_explain_sec = result.metadata.get("clean_explain_seconds", 0.0)
    clean_queries = result.metadata.get("clean_num_model_queries", 0)
    adv_explain_sec = result.metadata.get("adversarial_explain_seconds", 0.0)
    adv_queries = result.metadata.get("adversarial_num_model_queries", 0)

    print("\n" + "-" * 70)
    print("  PROFILING MEASUREMENT SUMMARY (STEP 1)")
    print("-" * 70)
    print(f"  ⏱️ Total Explain Seconds:   {explain_seconds:.6f} s (Wall clock: {total_wall_time:.4f} s)")
    print(f"  🔢 Total Model Queries:     {num_model_queries} query calls")
    print(f"     ├── Clean Explanation:   {clean_explain_sec:.6f} s  ({clean_queries} queries)")
    print(f"     └── Adv Explanation:     {adv_explain_sec:.6f} s  ({adv_queries} queries)")
    print("-" * 70)

    # -------------------------------------------------------------
    # [STEP 5/5] Explanation Analysis & Attribution Summary
    # -------------------------------------------------------------
    print("\n[STEP 5/5] Analyzing Explanation & Persisting Artifact...")
    comparison = result.comparison
    fail_analysis = result.failure_analysis

    print(f"  Technique Used:          {result.technique.upper()}")
    print(f"  Attribution Status:      {result.attribution_status.upper()}")
    print(f"  Attack Caused Failure:   {result.attack_caused_failure}")
    if "cosine_similarity" in comparison:
        print(f"  Attribution Cosine Sim:  {comparison['cosine_similarity']:.4f}")
    if "top_feature_overlap" in comparison:
        print(f"  Top Feature Overlap:     {comparison['top_feature_overlap'] * 100:.1f}%")

    # Persist JSON Artifact
    out_path = Path(args.output)
    result.save_json(out_path, store_npz=False)
    print(f"\n  💾 Explanation Result persisted to: {out_path.resolve()}")
    print("=" * 70)
    print("  ✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
