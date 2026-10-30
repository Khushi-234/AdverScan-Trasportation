"""
Model utility and detection functions for AdverScan ingestion module.
Includes domain detection, framework detection, and environment setup utilities.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from app.ingestion.presets import DATASET_TO_RECOMMENDED_MODEL, MODEL_PRESETS
from app.ingestion.utils.logger import log_ingestion_event


def is_forecasting_or_regression(
    model_or_path: Any,
    config: Optional[Any] = None,
    task_type: Optional[str] = None,
) -> bool:
    """
    Detect whether a time-series model represents a forecasting/regression task
    rather than classification, based on pipeline_tag, architecture, config, or task info.
    """

# Input
#  │
#  ├── 1. Already have a Config object? ──> Inspect pipeline_tag & architectures list
#  │
#  ├── 2. Input is a string/model ID?
#  │        ├── 2a. Check model name for keywords
#  │        ├── 2b. Try local AutoConfig (local_files_only=True)
#  │        └── 2c. Fallback to HF Hub config.json download & parse
#  │
#  └── 3. Input is an in-memory PyTorch instance? ──> Inspect class name

    clean_task = str(task_type).lower().strip() if task_type else ""
    if clean_task in ("forecasting", "regression", "time-series-forecasting", "ts-forecasting"):
        return True

    forecasting_keywords = (
        "forprediction",
        "forforecasting",
        "forregression",
        "timeseriesprediction",
        "timeseriesforecasting",
        "timeseriesregression",
        "chronos",
        "timesfm",
    )

    # Check config object if provided
    if config is not None:
        p_tag = getattr(config, "pipeline_tag", None) or getattr(config, "task", None)
        if p_tag and any(k in str(p_tag).lower() for k in ("forecasting", "regression")):
            return True
        archs = getattr(config, "architectures", []) or []
        for arch in archs:
            if any(k in str(arch).lower() for k in forecasting_keywords):
                return True

    # Check string model_or_path
    if isinstance(model_or_path, str):
        clean_name = model_or_path.lower()
        if any(k in clean_name for k in ("forprediction", "forforecasting", "forregression", "forecaster")):
            return True

        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_or_path, local_files_only=True)
            if cfg is not None:
                return is_forecasting_or_regression(None, config=cfg)
        except Exception:
            pass

        try:
            from huggingface_hub import hf_hub_download
            cfg_path = hf_hub_download(model_or_path, "config.json")
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw_cfg = json.load(f)
            p_tag = raw_cfg.get("pipeline_tag")
            if p_tag and any(k in str(p_tag).lower() for k in ("forecasting", "regression")):
                return True
            archs = raw_cfg.get("architectures", [])
            for arch in archs:
                if any(k in str(arch).lower() for k in forecasting_keywords):
                    return True
        except Exception:
            pass

    # Check in-memory model or class
    if model_or_path is not None:
        cls_name = model_or_path.__class__.__name__.lower()
        if any(k in cls_name for k in ("forprediction", "forforecasting", "forregression", "forecaster", "forecasting")):
            return True

    return False


def is_sklearn_model(model_or_path: Any) -> bool:
    """
    Check if a model object or file path represents a scikit-learn estimator.
    """
    if isinstance(model_or_path, (str, Path)):
        p_str = str(model_or_path).lower()
        return p_str.endswith((".joblib", ".pkl", ".pickle"))

    # Duck typing for sklearn estimators
    if hasattr(model_or_path, "predict") and (
        hasattr(model_or_path, "fit") or hasattr(model_or_path, "classes_") or hasattr(model_or_path, "_estimator_type")
    ):
        # Exclude PyTorch modules that might define predict()
        if isinstance(model_or_path, torch.nn.Module):
            return False
        return True

    return False


def detect_model_source(model_or_path: Any) -> str:
    """
    Determine model source ('local', 'in_memory', 'huggingface', 'url').
    Distinguishes source (where it comes from) from framework (how it executes).
    """
    if not isinstance(model_or_path, (str, Path)):
        return "in_memory"

    path_str = str(model_or_path).strip()
    if path_str.startswith(("http://", "https://")):
        return "url"

    p = Path(path_str)
    if p.exists():
        return "local"

    if "/" in path_str and not path_str.startswith((".", "/")):
        return "huggingface"

    return "local"


def detect_model_framework(
    model_or_path: Any,
    framework_override: Optional[str] = None,
    model_class: Optional[Any] = None,
) -> str:
    """
    Determine model framework ('sklearn', 'pytorch', 'tensorflow').
    Never conflates source ('huggingface') with execution framework.
    """
    if framework_override:
        clean = framework_override.lower().strip()
        if clean in ("pytorch", "pytorch_local", "sklearn", "tensorflow"):
            return clean

    # 1. In-memory inspection
    if isinstance(model_or_path, (torch.nn.Module, torch.jit.ScriptModule)):
        return "pytorch"

    if model_class is not None and isinstance(model_class, type) and issubclass(model_class, torch.nn.Module):
        return "pytorch"

    if is_sklearn_model(model_or_path):
        return "sklearn"

    # 2. File extension inspection
    if isinstance(model_or_path, (str, Path)):
        p_str = str(model_or_path).lower()
        if p_str.endswith((".pt", ".pth", ".safetensors", ".ckpt", ".bin")):
            return "pytorch"
        if p_str.endswith((".joblib",)):
            return "sklearn"
        if p_str.endswith((".pkl", ".pickle")):
            return "sklearn"
        # Check Hugging Face repos for scikit-learn estimators
        if "/" in p_str and not p_str.startswith((".", "/")):
            if p_str.startswith("scikit-learn/") or "sklearn" in p_str:
                return "sklearn"
            try:
                from huggingface_hub import hf_hub_download
                cfg_p = hf_hub_download(str(model_or_path), "config.json", local_files_only=True)
                with open(cfg_p, "r", encoding="utf-8") as f:
                    cfg_d = json.load(f)
                if "sklearn" in cfg_d:
                    return "sklearn"
            except Exception:
                pass

    # 3. Default fallback
    if framework_override:
        return framework_override.lower().strip()

    return "pytorch"


def inspect_model_evidence(
    model_or_path: Any,
    framework: Optional[str] = None,
    domain: Optional[str] = None,
    sample_input: Optional[Any] = None,
) -> List[str]:
    """
    Collect structured evidence explaining framework, domain, and capability detection.
    Useful for audit logs and actionable error reporting.
    """
    evidence: List[str] = []
    source = detect_model_source(model_or_path)
    fw = framework or detect_model_framework(model_or_path)

    evidence.append(f"Model source resolved as '{source}'")
    evidence.append(f"Execution framework resolved as '{fw}'")

    if fw == "sklearn":
        cls_name = model_or_path.__class__.__name__ if not isinstance(model_or_path, (str, Path)) else "Serialized"
        evidence.append(f"Detected scikit-learn estimator ({cls_name})")
        if hasattr(model_or_path, "predict"):
            evidence.append("Model exposes predict()")
        if hasattr(model_or_path, "predict_proba"):
            evidence.append("Model exposes predict_proba()")
        if hasattr(model_or_path, "decision_function"):
            evidence.append("Model exposes decision_function()")
        if hasattr(model_or_path, "n_features_in_"):
            evidence.append(f"Model fitted on n_features_in_ = {model_or_path.n_features_in_}")
        if hasattr(model_or_path, "classes_"):
            evidence.append(f"Detected classes count = {len(model_or_path.classes_)}")
    elif fw == "pytorch":
        evidence.append("Detected PyTorch neural network architecture")

    if domain:
        evidence.append(f"Data domain identified as '{domain}'")

    if sample_input is not None:
        if hasattr(sample_input, "shape"):
            evidence.append(f"Sample input provided with shape {tuple(sample_input.shape)}")
        if hasattr(sample_input, "columns"):
            evidence.append(f"Sample tabular DataFrame columns: {list(sample_input.columns)}")

    return evidence


def detect_model_domain(
    model_or_path: Any,
    sample_input: Optional[Any] = None,
    processor_name: Optional[str] = None,
    model_class: Optional[Any] = None,
) -> str:
    """
    Automatically detect the data domain ('image', 'text', 'tabular', or 'time-series') from a model identifier,
    in-memory model instance, processor/tokenizer name, or sample input.
    Returns 'unknown' if domain cannot be determined with certainty, never falling back to 'image'.
    """
    # 0. Check if model is a scikit-learn estimator
    if is_sklearn_model(model_or_path):
        return "tabular"

    # 1. Preset registry check
    if isinstance(model_or_path, str):
        clean_id = model_or_path.strip()
        if clean_id in MODEL_PRESETS:
            return MODEL_PRESETS[clean_id].get("data_domain", "unknown")
        if clean_id in DATASET_TO_RECOMMENDED_MODEL:
            target = DATASET_TO_RECOMMENDED_MODEL[clean_id]
            if target in MODEL_PRESETS:
                return MODEL_PRESETS[target].get("data_domain", "unknown")

    # 2. Check processor / tokenizer name
    if processor_name and isinstance(processor_name, str):
        proc_lower = processor_name.lower().replace("_", "-")
        if any(k in proc_lower for k in ("tabular", "tab-processor", "tabular-processor")):
            return "tabular"
        if any(k in proc_lower for k in ("timeseries", "time-series", "ts-processor")):
            return "time-series"
        if any(k in proc_lower for k in ("tokenizer", "bert", "roberta", "gpt", "t5", "bart", "text")):
            return "text"
        if any(k in proc_lower for k in ("image", "vision", "vit", "resnet", "feature-extractor")):
            return "image"

    # 3. Check sample input
    if sample_input is not None:
        if "pandas" in getattr(sample_input.__class__, "__module__", ""):
            return "time-series"
        if isinstance(sample_input, (str, list)) and (
            isinstance(sample_input, str) or (len(sample_input) > 0 and isinstance(sample_input[0], str))
        ):
            return "text"
        if isinstance(sample_input, dict):
            if any(k in sample_input for k in ("input_ids", "attention_mask", "token_type_ids")):
                return "text"
            if any(k in sample_input for k in ("pixel_values", "images")):
                return "image"
            if any(k in sample_input for k in ("past_values", "series", "sequence")):
                return "time-series"
        if "PIL" in getattr(sample_input.__class__, "__module__", ""):
            return "image"
        if isinstance(sample_input, torch.Tensor) or (hasattr(sample_input, "shape") and hasattr(sample_input, "dtype")):
            if len(sample_input.shape) == 4:
                return "image"
            if len(sample_input.shape) == 3 and sample_input.shape[0] in (1, 3, 4) and sample_input.shape[1] > 10 and sample_input.shape[2] > 10:
                return "image"
            dtype_str = str(getattr(sample_input, "dtype", ""))
            if len(sample_input.shape) in (1, 2):
                if any(t in dtype_str for t in ("int", "long")):
                    return "text"
            # 3D float tensor that is not a 2D image (e.g. single channel sequence [B, 1, L] or [B, L, 1])
            if len(sample_input.shape) == 3 and any(t in dtype_str for t in ("float",)):
                if sample_input.shape[1] == 1 or sample_input.shape[2] == 1:
                    return "time-series"

    # 4. Check in-memory model or class
    target_model = model_or_path if isinstance(model_or_path, torch.nn.Module) else None
    if target_model is None and callable(model_class) and issubclass(model_class, torch.nn.Module):
        try:
            target_model = model_class()
        except Exception:
            pass

    if target_model is not None:
        cls_name = target_model.__class__.__name__.lower()
        if any(k in cls_name for k in ("tabular", "tabnet", "fttransformer", "tabtransformer", "tabularmlp", "tabularclassifier")):
            return "tabular"
        ts_model_keywords = (
            "patchtst", "patchtsmixer", "timesnet", "informer", "autoformer",
            "1dcnn", "cnn1d", "conv1d", "lstm", "gru", "timeseries", "time_series", "time-series"
        )
        if any(k in cls_name for k in ts_model_keywords):
            return "time-series"
        if any(k in cls_name for k in ("vit", "resnet", "conv", "vision", "image", "swin", "deit", "cnn", "vgg", "alexnet")):
            return "image"
        if any(k in cls_name for k in ("bert", "roberta", "gpt", "t5", "text", "nlp", "tokenizer", "bart")):
            return "text"

        has_conv = False
        has_conv1d = False
        has_rnn = False
        has_embed = False
        for module in target_model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.ConvTranspose2d, torch.nn.AdaptiveAvgPool2d)):
                has_conv = True
                break
            if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                has_conv1d = True
            if isinstance(module, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN, torch.nn.RNNBase)):
                has_rnn = True
            if isinstance(module, torch.nn.Embedding):
                has_embed = True

        if has_conv:
            return "image"
        if (has_conv1d or has_rnn) and not has_conv and not has_embed:
            return "time-series"
        if has_embed and not has_conv:
            return "text"

    # 5. Check string identifier or Hugging Face config
    if isinstance(model_or_path, str):
        name_lower = model_or_path.lower().replace("_", "-")

        text_keywords = (
            "bert", "roberta", "gpt", "llama", "t5", "bart", "electra", "distilbert",
            "albert", "deberta", "sentiment", "imdb", "nlp", "sequenceclassification",
            "causallm", "gliner", "decide", "token_classification", "tokenclassification",
            "question_answering", "summarization", "translation", "sentence_transformers",
            "embedding", "bge", "e5", "minilm", "extractor"
        )
        vision_keywords = (
            "vit", "resnet", "convnext", "swin", "deit", "vision", "cifar", "gtsrb",
            "mnist", "imagenet", "mobilenet", "efficientnet", "vgg", "clip_vision", "beit",
            "yolo", "densenet", "segmentation", "object_detection"
        )
        ts_keywords = (
            "patchtst", "patchtsmixer", "timesnet", "informer", "autoformer",
            "timeseries", "time-series", "chronos", "timesfm", "1dcnn", "cnn1d",
            "lstm", "gru"
        )
        tabular_keywords = (
            "tabular", "tabnet", "tabtransformer", "fttransformer", "tabularmlp"
        )

        # Fast local name keyword matching first
        if any(k in name_lower for k in tabular_keywords):
            return "tabular"
        if any(k in name_lower for k in ts_keywords):
            return "time-series"
        if any(k in name_lower for k in text_keywords) or re.search(r"(^|[^a-z])(ner|qa|text)([^a-z]|$)", name_lower):
            return "text"
        if any(k in name_lower for k in vision_keywords):
            return "image"

        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_or_path, local_files_only=True)
            if cfg is not None:
                p_tag = getattr(cfg, "pipeline_tag", "") or getattr(cfg, "task", "")
                if p_tag and any(k in str(p_tag).lower() for k in ("tabular", "tabular-classification")):
                    return "tabular"
                if p_tag and any(k in str(p_tag).lower() for k in ("time-series", "forecasting")):
                    return "time-series"
                m_type = getattr(cfg, "model_type", "").lower()
                archs = [str(a).lower() for a in getattr(cfg, "architectures", [])]
                combined = m_type + " " + " ".join(archs)
                if any(k in combined for k in tabular_keywords):
                    return "tabular"
                if any(k in combined for k in ts_keywords):
                    return "time-series"
                if any(k in combined for k in text_keywords) or re.search(r"(^|[^a-z])(ner|qa|text)([^a-z]|$)", combined):
                    return "text"
                if any(k in combined for k in vision_keywords):
                    return "image"
        except Exception:
            pass

        # Check raw config.json or tokenizer / preprocessor existence on HF Hub
        try:
            from huggingface_hub import hf_hub_download
            try:
                cfg_path = hf_hub_download(model_or_path, "config.json")
                with open(cfg_path, "r", encoding="utf-8") as f:
                    raw_cfg = json.load(f)
                p_tag = raw_cfg.get("pipeline_tag")
                if p_tag and any(k in str(p_tag).lower() for k in ("tabular", "tabular-classification")):
                    return "tabular"
                if "sklearn" in raw_cfg or any(k in str(raw_cfg).lower() for k in ("tabular", "tabular-classification")):
                    return "tabular"
                if p_tag and any(k in str(p_tag).lower() for k in ("time-series", "forecasting")):
                    return "time-series"
                combined_cfg = " ".join(str(v).lower() for v in raw_cfg.values() if isinstance(v, (str, list)))
                if any(k in combined_cfg for k in ts_keywords):
                    return "time-series"
                if any(k in combined_cfg for k in text_keywords) or re.search(r"(^|[^a-z])(ner|qa|text)([^a-z]|$)", combined_cfg):
                    return "text"
                if any(k in combined_cfg for k in vision_keywords):
                    return "image"
            except Exception:
                pass

            # If tokenizer exists, it's text
            try:
                hf_hub_download(model_or_path, "tokenizer_config.json")
                return "text"
            except Exception:
                pass

            # If preprocessor exists, it's image
            try:
                hf_hub_download(model_or_path, "preprocessor_config.json")
                return "image"
            except Exception:
                pass
        except Exception:
            pass

    return "unknown"


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
    from app.ingestion.model_registry import ModelPresetRegistry
    from app.utils.device_manager import DeviceManager

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
        "data_domain": model_info.get("data_domain", "unknown"),
        "model_type": model_info.get("model_type", "Model"),
        "domain": model_info.get("domain", "Unknown Domain"),
        "framework": model_info["framework"],
        "target_device": str(target_device),
        "cuda_available": torch.cuda.is_available(),
        "recommended_dataset": dataset_name,
        "task_type": model_info.get("task_type", "classification"),
        "expected_num_classes": model_info.get("num_classes"),
        "expected_input_resolution": expected_resolution,
        "sequence_length": model_info.get("sequence_length"),
        "num_channels": model_info.get("num_channels"),
        "num_features": model_info.get("num_features"),
        "status": "ENVIRONMENT_CONFIGURED",
    }

    if log_log_event:
        log_ingestion_event(env_config, log_filepath=log_filepath)

    return env_config


__all__ = [
    "is_forecasting_or_regression",
    "is_sklearn_model",
    "detect_model_source",
    "detect_model_framework",
    "inspect_model_evidence",
    "detect_model_domain",
    "setup_model_environment",
]
