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
        "data_domain": "image",
        "model_type": "Vision Transformer (ViT)",
        "domain": "Computer Vision (Traffic Sign Recognition / ITS)",
        "description": "Vision Transformer (ViT) fine-tuned on German Traffic Sign Recognition Benchmark (43 classes)",
        "dataset_name": "bazyl/GTSRB",
        "task_type": "classification",
        "num_classes": 43,
        "input_resolution": (3, 224, 224),
        "framework": "huggingface",
    },
    "ashutoshsharma58/scenery3214154": {
        "model_id": "ashutoshsharma58/scenery3214154",
        "name": "Intel Image Classification ViT",
        "data_domain": "image",
        "model_type": "Vision Transformer (ViT)",
        "domain": "Computer Vision (Scene & Natural Landscape Classification)",
        "description": "Vision Transformer fine-tuned on Intel Image Classification dataset (6 scene classes: Buildings, Forest, Glacier, Mountain, Sea, Street)",
        "dataset_name": "miladfa7/Intel-Image-Classification",
        "task_type": "classification",
        "num_classes": 6,
        "input_resolution": (3, 224, 224),
        "framework": "huggingface",
        "kaggle_url": "https://www.kaggle.com/datasets/puneet6060/intel-image-classification",
    },
    "puneet6060/intel-image-classification": {
        "model_id": "ashutoshsharma58/scenery3214154",
        "name": "Intel Image Classification ViT (Kaggle Dataset)",
        "data_domain": "image",
        "model_type": "Vision Transformer (ViT)",
        "domain": "Computer Vision (Scene & Natural Landscape Classification)",
        "description": "Vision Transformer fine-tuned on Kaggle Intel Image Classification dataset (6 scene classes)",
        "dataset_name": "miladfa7/Intel-Image-Classification",
        "task_type": "classification",
        "num_classes": 6,
        "input_resolution": (3, 224, 224),
        "framework": "huggingface",
        "kaggle_url": "https://www.kaggle.com/datasets/puneet6060/intel-image-classification",
    },
    "intel-image-classification": {
        "model_id": "ashutoshsharma58/scenery3214154",
        "name": "Intel Image Classification ViT",
        "data_domain": "image",
        "model_type": "Vision Transformer (ViT)",
        "domain": "Computer Vision (Scene & Natural Landscape Classification)",
        "description": "Vision Transformer model for Kaggle Intel Image Classification dataset",
        "dataset_name": "miladfa7/Intel-Image-Classification",
        "task_type": "classification",
        "num_classes": 6,
        "input_resolution": (3, 224, 224),
        "framework": "huggingface",
        "kaggle_url": "https://www.kaggle.com/datasets/puneet6060/intel-image-classification",
    },
}


class ModelPresetRegistry:
    """
    Registry for supported model presets and custom model resolution.
    """

    @classmethod
    def list_presets(cls) -> List[Dict[str, Any]]:
        """Return list of unique pre-configured model definitions."""
        seen = set()
        unique_presets = []
        for preset in MODEL_PRESETS.values():
            m_id = preset["model_id"]
            if m_id not in seen:
                seen.add(m_id)
                unique_presets.append(preset)
        return unique_presets

    @classmethod
    def get_preset(cls, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model preset metadata by identifier if available."""
        return MODEL_PRESETS.get(model_id)

    @classmethod
    def resolve_model_info(cls, model_id_or_path: str) -> Dict[str, Any]:
        """
        Resolve metadata for a model ID (preset or custom HF model), Kaggle dataset URL, or local file path.
        """
        clean_input = str(model_id_or_path).strip()
        
        # Handle Kaggle URL resolution
        if "kaggle.com" in clean_input and "puneet6060/intel-image-classification" in clean_input:
            clean_input = "puneet6060/intel-image-classification"

        if clean_input in MODEL_PRESETS:
            return cls.get_preset(clean_input).copy()

        p = Path(clean_input)
        if p.is_file():
            return {
                "model_id": str(p.resolve()),
                "name": p.stem,
                "data_domain": "image",
                "model_type": "PyTorch Checkpoint (.pt/.pth)",
                "domain": "Computer Vision / PyTorch Checkpoint",
                "description": f"Local PyTorch model checkpoint ({p.name})",
                "dataset_name": "custom_dataset",
                "task_type": "classification",
                "num_classes": None,
                "input_resolution": (3, 224, 224),
                "framework": "pytorch_local",
            }

        # Custom Hugging Face Hub Model ID
        return {
            "model_id": clean_input,
            "name": clean_input.split("/")[-1],
            "data_domain": "image",
            "model_type": "Hugging Face Model",
            "domain": "Computer Vision (Custom Hugging Face)",
            "description": f"Custom Hugging Face model ({clean_input})",
            "dataset_name": "custom_dataset",
            "task_type": "classification",
            "num_classes": None,
            "input_resolution": (3, 224, 224),
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
    log_log_event: bool = True,
    log_filepath: Union[str, Path] = "results/ingestion_log.json",
) -> Dict[str, Any]:
    """
    Configure and verify execution environment parameters for a selected model.

    Args:
        model_id_or_path: Hugging Face hub ID, model preset ID, or local file path.
        device: Target execution device ('cuda', 'cpu', etc.).
        custom_dataset_name: Optional override dataset identifier.
        log_log_event: Whether to log this setup event to disk.
        log_filepath: Location of ingestion log file.

    Returns:
        Environment metadata dictionary.
    """
    from app.ingestion.runtime.device_manager import DeviceManager

    target_device = DeviceManager.get_device(device)
    model_info = ModelPresetRegistry.resolve_model_info(model_id_or_path)

    dataset_name = custom_dataset_name or model_info.get("dataset_name", "bazyl/GTSRB")

    env_config = {
        "model_id": model_info["model_id"],
        "model_name": model_info["name"],
        "data_domain": model_info.get("data_domain", "image"),
        "model_type": model_info.get("model_type", "Vision Transformer (ViT)"),
        "domain": model_info.get("domain", "Computer Vision"),
        "framework": model_info["framework"],
        "target_device": str(target_device),
        "cuda_available": torch.cuda.is_available(),
        "recommended_dataset": dataset_name,
        "task_type": model_info.get("task_type", "classification"),
        "expected_num_classes": model_info.get("num_classes"),
        "expected_input_resolution": list(model_info.get("input_resolution", (3, 224, 224))),
        "status": "ENVIRONMENT_CONFIGURED",
    }

    if log_log_event:
        log_ingestion_event(env_config, log_filepath=log_filepath)

    return env_config

