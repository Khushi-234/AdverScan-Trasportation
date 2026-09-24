"""
Model registry, environment manager, and ingestion logging for AdverScan framework.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "bazyl/gtsrb-model": {
        "model_id": "bazyl/gtsrb-model",
        "name": "GTSRB Vision Transformer",
        "processor_name": "bazyl/gtsrb-model",
        "dataset_name": "bazyl/GTSRB",
        "data_domain": "image",
        "model_type": "Vision Transformer (ViT)",
        "domain": "Computer Vision (Traffic Sign Recognition / ITS)",
        "description": "Vision Transformer (ViT) fine-tuned on German Traffic Sign Recognition Benchmark (43 classes)",
        "task_type": "classification",
        "num_classes": 43,
        "input_resolution": (3, 224, 224),
        "framework": "huggingface",
    },
    "ashutoshsharma58/scenery3214154": {
        "model_id": "ashutoshsharma58/scenery3214154",
        "name": "Intel Image Classification ViT",
        "processor_name": "google/vit-base-patch16-224-in21k",
        "dataset_name": "miladfa7/Intel-Image-Classification",
        "data_domain": "image",
        "model_type": "Vision Transformer (ViT)",
        "domain": "Computer Vision (Scene & Natural Landscape Classification)",
        "description": "Vision Transformer fine-tuned on Intel Image Classification dataset (6 scene classes: Buildings, Forest, Glacier, Mountain, Sea, Street)",
        "task_type": "classification",
        "num_classes": 6,
        "input_resolution": (3, 224, 224),
        "framework": "huggingface",
        "kaggle_url": "https://www.kaggle.com/datasets/puneet6060/intel-image-classification",
    },
    "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10": {
        "model_id": "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10",
        "name": "CIFAR-10 Vision Transformer",
        "processor_name": "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10",
        "dataset_name": "uoft-cs/cifar10",
        "data_domain": "image",
        "model_type": "Vision Transformer (ViT)",
        "domain": "Computer Vision (General Object Classification / CIFAR-10)",
        "description": "Vision Transformer (ViT) fine-tuned on CIFAR-10 dataset (10 object classes)",
        "task_type": "classification",
        "num_classes": 10,
        "input_resolution": (3, 224, 224),
        "framework": "huggingface",
    },
    "google/vit-base-patch16-224": {
        "model_id": "google/vit-base-patch16-224",
        "name": "ImageNet-1k Vision Transformer",
        "processor_name": "google/vit-base-patch16-224",
        "dataset_name": "imagenet-1k",
        "data_domain": "image",
        "model_type": "Vision Transformer (ViT)",
        "domain": "Computer Vision (ImageNet-1k Benchmark)",
        "description": "Base Vision Transformer pre-trained and fine-tuned on ImageNet-1k (1,000 classes)",
        "task_type": "classification",
        "num_classes": 1000,
        "input_resolution": (3, 224, 224),
        "framework": "huggingface",
    },
}

# Separate mapping from dataset identifiers / Kaggle URLs to their recommended evaluation models
DATASET_TO_RECOMMENDED_MODEL: Dict[str, str] = {
    "puneet6060/intel-image-classification": "ashutoshsharma58/scenery3214154",
    "intel-image-classification": "ashutoshsharma58/scenery3214154",
    "miladfa7/intel-image-classification": "ashutoshsharma58/scenery3214154",
    "bazyl/gtsrb": "bazyl/gtsrb-model",
    "gtsrb": "bazyl/gtsrb-model",
    "cifar10": "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10",
    "uoft-cs/cifar10": "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10",
}


class ModelPresetRegistry:
    """
    Registry for supported model presets and custom model resolution.
    Distinguishes model_id (what to load), dataset_name (what to evaluate),
    and processor_name (how to preprocess inputs).
    """

    @classmethod
    def list_presets(cls) -> List[Dict[str, Any]]:
        """Return list of unique pre-configured model definitions."""
        return list(MODEL_PRESETS.values())

    @classmethod
    def get_preset(cls, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model preset metadata by identifier if available."""
        return MODEL_PRESETS.get(model_id)

    @classmethod
    def resolve_model_info(
        cls,
        model_id_or_path: str,
        data_domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Resolve metadata for a model ID (preset or custom HF model), Kaggle dataset URL, or local file path.
        Distinguishes model_id, dataset_name, and processor_name, and infers domain without blindly assuming image.
        """
        clean_input = str(model_id_or_path).strip()
        lower_input = clean_input.lower()
        
        matched_dataset = None
        if "kaggle.com" in lower_input:
            for ds_key, target_model in DATASET_TO_RECOMMENDED_MODEL.items():
                if ds_key in lower_input:
                    clean_input = target_model
                    matched_dataset = ds_key
                    break
        elif lower_input in DATASET_TO_RECOMMENDED_MODEL:
            matched_dataset = lower_input
            clean_input = DATASET_TO_RECOMMENDED_MODEL[lower_input]

        if clean_input in MODEL_PRESETS:
            preset = cls.get_preset(clean_input).copy()
            if matched_dataset:
                preset["dataset_name"] = matched_dataset
            return preset

        p = Path(clean_input)
        if p.is_file():
            inferred_domain = data_domain or "image"
            return {
                "model_id": str(p.resolve()),
                "name": p.stem,
                "processor_name": None,
                "dataset_name": "custom_dataset",
                "data_domain": inferred_domain,
                "model_type": f"PyTorch Checkpoint ({p.suffix})",
                "domain": f"{inferred_domain.capitalize()} / Local Checkpoint",
                "description": f"Local PyTorch model checkpoint ({p.name})",
                "task_type": "classification",
                "num_classes": None,
                "input_resolution": (3, 224, 224) if inferred_domain == "image" else None,
                "framework": "pytorch_local",
            }

        # Custom Hugging Face Hub Model ID — inspect architecture dynamically
        inferred_domain = data_domain or "image"
        model_type_str = "Hugging Face Model"
        num_classes = None

        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(clean_input)
            m_type = getattr(cfg, "model_type", "").lower()
            archs = [str(a).lower() for a in getattr(cfg, "architectures", [])]

            if any(k in m_type or any(k in a for a in archs) for k in (
                "bert", "roberta", "gpt", "llama", "t5", "bart", "electra", "text", "sequenceclassification", "causallm"
            )):
                inferred_domain = data_domain or "text"
                model_type_str = f"Transformer NLP ({getattr(cfg, 'model_type', 'text')})"
            elif any(k in m_type or any(k in a for a in archs) for k in (
                "vit", "resnet", "convnext", "swin", "deit", "image", "vision"
            )):
                inferred_domain = data_domain or "image"
                model_type_str = f"Vision Model ({getattr(cfg, 'model_type', 'vision')})"
            else:
                model_type_str = f"HF Model ({getattr(cfg, 'model_type', 'generic')})"

            num_classes = getattr(cfg, "num_labels", None)
        except Exception:
            pass

        return {
            "model_id": clean_input,
            "name": clean_input.split("/")[-1],
            "processor_name": clean_input,
            "dataset_name": "custom_dataset",
            "data_domain": inferred_domain,
            "model_type": model_type_str,
            "domain": f"{inferred_domain.capitalize()} (Custom Hugging Face)",
            "description": f"Custom Hugging Face model ({clean_input})",
            "task_type": "classification",
            "num_classes": num_classes,
            "input_resolution": (3, 224, 224) if inferred_domain == "image" else None,
            "framework": "huggingface",
        }



def log_ingestion_event(
    event_data: Dict[str, Any],
    log_filepath: Union[str, Path] = "results/ingestion_log.json"
) -> str:
    """
    Record an ingestion or environment setup event to a structured JSON log.

    Args:
        event_data: Metadata dictionary describing the model and setup.
        log_filepath: Target output file path.

    Returns:
        Path string to the written log file.
    """
    log_path = Path(log_filepath)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    if log_path.exists() and log_path.stat().st_size > 0:
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                history = json.load(f)
                if not isinstance(history, list):
                    history = [history]
        except Exception:
            history = []

    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "event_type": "MODEL_INGESTION_AND_ENVIRONMENT_SETUP",
        **event_data,
    }
    history.append(entry)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return str(log_path)


def setup_model_environment(
    model_id_or_path: str,
    device: Optional[Union[str, torch.device]] = None,
    custom_dataset_name: Optional[str] = None,
    custom_processor_name: Optional[str] = None,
    data_domain: Optional[str] = None,
    log_log_event: bool = True,
    log_filepath: Union[str, Path] = "results/ingestion_log.json",
) -> Dict[str, Any]:
    """
    Configure and verify execution environment parameters for a selected model.
    Distinguishes model_id (what to load), dataset_name (what to evaluate),
    and processor_name (how to preprocess inputs).

    Args:
        model_id_or_path: Hugging Face hub ID, model preset ID, or local file path.
        device: Target execution device ('cuda', 'cpu', etc.).
        custom_dataset_name: Optional override dataset identifier.
        custom_processor_name: Optional override processor / tokenizer identifier.
        data_domain: Optional override data domain ('image', 'text', etc.).
        log_log_event: Whether to log this setup event to disk.
        log_filepath: Location of ingestion log file.

    Returns:
        Environment metadata dictionary.
    """
    from app.ingestion.runtime.device_manager import DeviceManager

    target_device = DeviceManager.get_device(device)
    model_info = ModelPresetRegistry.resolve_model_info(model_id_or_path, data_domain=data_domain)

    dataset_name = custom_dataset_name or model_info.get("dataset_name", "bazyl/GTSRB")
    processor_name = custom_processor_name or model_info.get("processor_name", model_info["model_id"])

    resolution = model_info.get("input_resolution")
    expected_resolution = list(resolution) if resolution is not None else None

    env_config = {
        "model_id": model_info["model_id"],
        "model_name": model_info["name"],
        "processor_name": processor_name,
        "data_domain": model_info.get("data_domain", "image"),
        "model_type": model_info.get("model_type", "Vision Transformer (ViT)"),
        "domain": model_info.get("domain", "Computer Vision"),
        "framework": model_info["framework"],
        "target_device": str(target_device),
        "cuda_available": torch.cuda.is_available(),
        "recommended_dataset": dataset_name,
        "task_type": model_info.get("task_type", "classification"),
        "expected_num_classes": model_info.get("num_classes"),
        "expected_input_resolution": expected_resolution,
        "status": "ENVIRONMENT_CONFIGURED",
    }

    if log_log_event:
        log_ingestion_event(env_config, log_filepath=log_filepath)

    return env_config

