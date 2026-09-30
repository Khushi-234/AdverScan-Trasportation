"""
Validation script for Hugging Face GTSRB dataset (bazyl/GTSRB) and ViT model (bazyl/gtsrb-model).
Validates dataset loading, single-image preprocessing, ViT model inference, label mapping,
and gradient flow compatibility for future Module 2 & 3 research.
"""

import io
import json
import torch
from dotenv import load_dotenv
from PIL import Image
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, AutoModelForImageClassification, ViTConfig

# Load environment variables from .env if present
load_dotenv()


def main():
    print("=" * 60)
    print("AdverScan — ITS Model & Dataset Pipeline Validation")
    print("=" * 60)

    # -------------------------------------------------------------
    # STEP 2 — Load Dataset
    # -------------------------------------------------------------
    print("\n[STEP 2] Loading Hugging Face Dataset: bazyl/GTSRB ...")
    dataset_dict = load_dataset("bazyl/GTSRB")
    
    print("\n--- DatasetDict Overview ---")
    print(f"Available splits: {list(dataset_dict.keys())}")
    for split_name, split_data in dataset_dict.items():
        print(f"  Split '{split_name}': {len(split_data)} examples, columns = {split_data.column_names}")

    # -------------------------------------------------------------
    # STEP 3 — Inspect One Sample
    # -------------------------------------------------------------
    print("\n[STEP 3] Inspecting First Sample from 'train' split ...")
    target_split = "train" if "train" in dataset_dict else list(dataset_dict.keys())[0]
    sample = dataset_dict[target_split][0]

    # In bazyl/GTSRB, the image bytes are stored in sample['Path']['bytes']
    image_bytes = sample["Path"]["bytes"]
    sample_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    actual_class_id = sample["ClassId"]

    print(f"Sample index: 0 (from split '{target_split}')")
    print(f"Image type: {type(sample_img)}")
    print(f"Image size (W, H): {sample_img.size}")
    print(f"Image mode: {sample_img.mode}")
    print(f"Actual ClassId: {actual_class_id}")

    # -------------------------------------------------------------
    # STEP 4 — Load Model & Processor
    # -------------------------------------------------------------
    model_name = "bazyl/gtsrb-model"
    print(f"\n[STEP 4] Loading Model & Processor from: {model_name} ...")

    # Load and patch config to fix upstream null value in id2label['43']
    config_path = hf_hub_download(model_name, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    # Upstream config fix: key '43' is null in JSON; replace with 'Unused' string label
    if "id2label" in cfg_dict:
        cfg_dict["id2label"] = {
            str(k): (str(v) if v is not None else "Unused")
            for k, v in cfg_dict["id2label"].items()
        }
        cfg_dict["label2id"] = {v: int(k) for k, v in cfg_dict["id2label"].items()}
        cfg_dict["num_labels"] = len(cfg_dict["id2label"])

    model_config = ViTConfig.from_dict(cfg_dict)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name, config=model_config, use_safetensors=True)
    model.eval()

    print("\n--- Model Specifications ---")
    print(f"Model Class: {model.__class__.__name__}")
    print(f"Base Architecture: ViT ({model.config.model_type.upper()})")
    print(f"Number of Config Labels: {model.config.num_labels} (43 GTSRB classes + 1 unused class 43)")

    # -------------------------------------------------------------
    # STEP 5 — Label Configuration Verification
    # -------------------------------------------------------------
    print("\n[STEP 5] Verifying Label Mapping Configuration ...")
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = model.config.label2id

    actual_label = id2label.get(actual_class_id, f"LABEL_{actual_class_id}")
    print(f"GTSRB Active Classes: 43 (Indices 0 to 42)")
    print(f"Sample id2label mapping (first 5): {dict(list(id2label.items())[:5])}")
    print(f"Actual ClassId {actual_class_id} maps to label: '{actual_label}'")

    # -------------------------------------------------------------
    # STEP 6 — Preprocessing Verification
    # -------------------------------------------------------------
    print("\n[STEP 6] Inspecting AutoImageProcessor Preprocessing Configuration ...")
    print(f"Processor Type: {processor.__class__.__name__}")
    print(f"Target Image Size: {getattr(processor, 'size', 'N/A')}")
    print(f"Do Resize: {getattr(processor, 'do_resize', 'N/A')}")
    print(f"Do Rescale: {getattr(processor, 'do_rescale', 'N/A')} (factor: {getattr(processor, 'rescale_factor', 'N/A')})")
    print(f"Do Normalize: {getattr(processor, 'do_normalize', 'N/A')}")
    print(f"Image Mean: {getattr(processor, 'image_mean', 'N/A')}")
    print(f"Image Std: {getattr(processor, 'image_std', 'N/A')}")

    inputs = processor(images=sample_img, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    print(f"Preprocessed input tensor shape: {pixel_values.shape}")
    print(f"Preprocessed input dtype: {pixel_values.dtype}")
    print(f"Preprocessed value range: [{pixel_values.min().item():.4f}, {pixel_values.max().item():.4f}]")

    # -------------------------------------------------------------
    # STEP 7 — Single Image Inference
    # -------------------------------------------------------------
    print("\n[STEP 7] Running Single Image Inference ...")
    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs.logits  # Shape: (1, 44)

    # Slice logits to 43 GTSRB classes
    gtsrb_logits = logits[:, :43]
    probs = torch.softmax(gtsrb_logits, dim=-1)
    pred_class_id = torch.argmax(probs, dim=-1).item()
    confidence = probs[0, pred_class_id].item()
    pred_label = id2label.get(pred_class_id, f"LABEL_{pred_class_id}")

    print("\n========================================")
    print("AdverScan — ITS Single Image Test")
    print("========================================")
    print(f"Dataset: bazyl/GTSRB")
    print(f"Model: {model_name}")
    print(f"Architecture: ViT")
    print(f"Actual ClassId: {actual_class_id}")
    print(f"Predicted ClassId: {pred_class_id}")
    print(f"Actual Label: {actual_label}")
    print(f"Predicted Label: {pred_label}")
    print(f"Confidence: {confidence * 100:.2f}% ({confidence:.6f})")
    print(f"Logits Shape: {tuple(logits.shape)} (GTSRB Sliced: {tuple(gtsrb_logits.shape)})")
    print("========================================\n")

    # -------------------------------------------------------------
    # STEP 8 — Gradient Flow Verification
    # -------------------------------------------------------------
    print("[STEP 8] Verifying Gradient Computation Flow ...")
    grad_pixel_values = processor(images=sample_img, return_tensors="pt")["pixel_values"]
    grad_pixel_values.requires_grad_(True)

    grad_outputs = model(grad_pixel_values)
    grad_logits = grad_outputs.logits[:, :43]
    
    # Loss w.r.t target ground truth class
    target_score = grad_logits[0, actual_class_id]
    target_score.backward()

    has_grad = grad_pixel_values.grad is not None
    grad_shape = tuple(grad_pixel_values.grad.shape) if has_grad else None
    is_finite = torch.isfinite(grad_pixel_values.grad).all().item() if has_grad else False
    is_non_zero = (grad_pixel_values.grad.abs().sum() > 0).item() if has_grad else False

    print(f"pixel_values.grad is not None: {has_grad}")
    print(f"Gradient shape: {grad_shape}")
    print(f"Gradients are finite: {is_finite}")
    print(f"Gradients are non-zero: {is_non_zero}")
    print(f"Gradient norm (L2): {torch.norm(grad_pixel_values.grad).item():.6f}")

    if has_grad and is_finite and is_non_zero:
        print("\n✅ Gradient Flow Verification: PASSED")
    else:
        print("\n❌ Gradient Flow Verification: FAILED")

    print("\nValidation process completed successfully.")


if __name__ == "__main__":
    main()
