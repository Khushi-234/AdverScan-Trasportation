"""
Model registry, environment manager, and ingestion logging for AdverScan framework.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

from app.ingestion.metadata.model_metadata import ModelMetadata
from app.ingestion.presets import DATASET_TO_RECOMMENDED_MODEL, MODEL_PRESETS



def is_forecasting_or_regression(
    model_or_path: Any,
    config: Optional[Any] = None,
    task_type: Optional[str] = None,
) -> bool:
    """
    Detect whether a time-series model represents a forecasting/regression task
    rather than classification, based on pipeline_tag, architecture, config, or task info.
    """
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


def detect_model_domain(
    model_or_path: Any,
    sample_input: Optional[Any] = None,
    processor_name: Optional[str] = None,
    model_class: Optional[Any] = None,
) -> str:
    """
    Automatically detect the data domain ('image', 'text', or 'time-series') from a model identifier,
    in-memory model instance, processor/tokenizer name, or sample input.
    Returns 'unknown' if domain cannot be determined with certainty, never falling back to 'image'.
    """
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
        import re

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

        # Fast local name keyword matching first
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
                if p_tag and any(k in str(p_tag).lower() for k in ("time-series", "forecasting")):
                    return "time-series"
                m_type = getattr(cfg, "model_type", "").lower()
                archs = [str(a).lower() for a in getattr(cfg, "architectures", [])]
                combined = m_type + " " + " ".join(archs)
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
    def has_preset(cls, model_id: str) -> bool:
        """Check whether model_id exists in presets."""
        return str(model_id).strip() in MODEL_PRESETS

    @classmethod
    def detect_domain(
        cls,
        model_id: Any,
        sample_input: Optional[Any] = None,
        processor_name: Optional[str] = None,
        model_class: Optional[Any] = None,
    ) -> str:
        """Helper to automatically detect domain from model or return 'unknown'."""
        return detect_model_domain(
            model_or_path=model_id,
            sample_input=sample_input,
            processor_name=processor_name,
            model_class=model_class,
        )

    @classmethod
    def build_runtime_config(
        cls,
        model_id: str,
        dataset_name: Optional[str] = None,
        data_domain: Optional[str] = None,
        task_type: Optional[str] = None,
        framework: Optional[str] = None,
        processor_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Build a normalized runtime configuration dictionary for either a preset model
        or a user-provided custom model.
        """
        clean_id = str(model_id).strip()
        if cls.has_preset(clean_id):
            config = cls.get_preset(clean_id).copy()
            if dataset_name is not None:
                config["dataset_name"] = dataset_name
            if data_domain is not None:
                config["data_domain"] = data_domain
                config["domain"] = data_domain
            if task_type is not None:
                config["task_type"] = task_type
            if framework is not None:
                config["framework"] = framework
            if processor_name is not None:
                config["processor_name"] = processor_name
            config.update(kwargs)
            return config

        # When model_id is not in presets, build runtime configuration from user-provided values
        p = Path(clean_id)
        is_local_file = p.is_file()

        resolved_framework = framework or ("pytorch_local" if is_local_file else "huggingface")
        resolved_domain = data_domain
        if not resolved_domain:
            resolved_domain = cls.detect_domain(model_id=clean_id, processor_name=processor_name)
        resolved_domain = resolved_domain or "unknown"
        if str(resolved_domain).lower().strip().replace("_", "-") in ("timeseries", "time-series"):
            resolved_domain = "time-series"

        resolved_task = task_type
        if not resolved_task:
            if is_forecasting_or_regression(clean_id):
                resolved_task = "forecasting"
            else:
                resolved_task = "classification"

        resolved_dataset = dataset_name or "custom_dataset"
        resolved_processor = processor_name or (None if is_local_file else clean_id)

        config = {
            "model_id": clean_id,
            "name": clean_id.split("/")[-1],
            "processor_name": resolved_processor,
            "dataset_name": resolved_dataset,
            "data_domain": resolved_domain,
            "domain": f"{resolved_domain.capitalize()} (Custom Model)" if resolved_domain != "unknown" else "Unknown (Custom Model)",
            "model_type": f"Custom {resolved_framework.capitalize()} Model",
            "description": f"User-provided custom model ({clean_id})",
            "task_type": resolved_task,
            "framework": resolved_framework,
            "input_resolution": (3, 224, 224) if resolved_domain == "image" else None,
            **kwargs,
        }
        return config

    @classmethod
    def create_model_metadata(
        cls,
        model_id: str,
        dataset_name: Optional[str] = None,
        data_domain: Optional[str] = None,
        task_type: Optional[str] = None,
        framework: Optional[str] = None,
        processor_name: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelMetadata:
        """
        Build and normalize runtime ModelMetadata from either registry preset or custom user input.
        """
        config = cls.build_runtime_config(
            model_id=model_id,
            dataset_name=dataset_name,
            data_domain=data_domain,
            task_type=task_type,
            framework=framework,
            processor_name=processor_name,
            **kwargs,
        )
        return ModelMetadata.from_dict(config)

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
            inferred_domain = data_domain or cls.detect_domain(clean_input) or "unknown"
            return {
                "model_id": str(p.resolve()),
                "name": p.stem,
                "processor_name": None,
                "dataset_name": "custom_dataset",
                "data_domain": inferred_domain,
                "model_type": f"PyTorch Checkpoint ({p.suffix})",
                "domain": f"{inferred_domain.capitalize()} / Local Checkpoint" if inferred_domain != "unknown" else "Unknown / Local Checkpoint",
                "description": f"Local PyTorch model checkpoint ({p.name})",
                "task_type": "classification",
                "num_classes": None,
                "input_resolution": (3, 224, 224) if inferred_domain == "image" else None,
                "framework": "pytorch_local",
            }

        # Custom Hugging Face Hub Model ID — inspect architecture dynamically
        inferred_domain = data_domain or cls.detect_domain(clean_input) or "unknown"
        model_type_str = f"Hugging Face Model ({inferred_domain})"
        num_classes = None

        try:
            from transformers import AutoConfig
            try:
                cfg = AutoConfig.from_pretrained(clean_input, local_files_only=True)
            except Exception:
                if "/" in clean_input:
                    try:
                        cfg = AutoConfig.from_pretrained(clean_input)
                    except Exception:
                        cfg = None
                else:
                    cfg = None
            
            m_type = getattr(cfg, "model_type", "").lower()
            if inferred_domain == "text":
                model_type_str = f"Transformer NLP ({m_type or 'text'})"
            elif inferred_domain == "image":
                model_type_str = f"Vision Model ({m_type or 'vision'})"
            else:
                model_type_str = f"HF Model ({m_type or 'generic'})"

            seq_len = getattr(cfg, "context_length", getattr(cfg, "sequence_length", None))
            num_channels = getattr(cfg, "num_input_channels", getattr(cfg, "num_channels", None))
            num_classes = getattr(cfg, "num_targets", getattr(cfg, "num_labels", getattr(cfg, "num_classes", None)))
        except Exception:
            seq_len = None
            num_channels = None

        return {
            "model_id": clean_input,
            "name": clean_input.split("/")[-1],
            "processor_name": clean_input,
            "dataset_name": "custom_dataset",
            "data_domain": inferred_domain,
            "model_type": model_type_str,
            "domain": f"{inferred_domain.capitalize()} (Custom Hugging Face)" if inferred_domain != "unknown" else "Unknown (Custom Hugging Face)",
            "description": f"Custom Hugging Face model ({clean_input})",
            "task_type": "classification",
            "num_classes": num_classes,
            "input_resolution": (3, 224, 224) if inferred_domain == "image" else None,
            "sequence_length": seq_len,
            "num_channels": num_channels,
            "framework": "huggingface",
        }




from app.ingestion.utils.logger import log_ingestion_event


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
        "status": "ENVIRONMENT_CONFIGURED",
    }

    if log_log_event:
        log_ingestion_event(env_config, log_filepath=log_filepath)

    return env_config

