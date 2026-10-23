"""
Model registry and preset definitions for AdverScan framework.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from app.ingestion.metadata.model_metadata import ModelMetadata
from app.ingestion.presets import DATASET_TO_RECOMMENDED_MODEL, MODEL_PRESETS
from app.ingestion.utils.logger import log_ingestion_event
from app.ingestion.utils.model_utils import (
    detect_model_domain,
    detect_model_framework,
    detect_model_source,
    inspect_model_evidence,
    is_forecasting_or_regression,
    is_sklearn_model,
    setup_model_environment,
)



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
        clean = model_id.strip()
        if clean in MODEL_PRESETS:
            return MODEL_PRESETS[clean]
        for p in MODEL_PRESETS.values():
            if p.get("model_id") == clean:
                return p
        return None

    @classmethod
    def has_preset(cls, model_id: str) -> bool:
        """Check whether model_id exists in presets."""
        return cls.get_preset(model_id) is not None


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
        clean_id = model_id.strip()
        preset = cls.get_preset(clean_id)
        if preset is not None:
            config = preset.copy()
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

        resolved_domain = data_domain
        if not resolved_domain:
            resolved_domain = detect_model_domain(model_or_path=clean_id, processor_name=processor_name)
        resolved_domain = resolved_domain or "unknown"
        if resolved_domain.lower().strip().replace("_", "-") in ("timeseries", "time-series"):
            resolved_domain = "time-series"

        detected_source = detect_model_source(clean_id)
        detected_fw = detect_model_framework(clean_id)
        if framework:
            resolved_framework = framework
        elif is_local_file:
            resolved_framework = "sklearn" if p.suffix.lower() in (".joblib", ".pkl", ".pickle") else "pytorch_local"
        elif detected_source == "huggingface":
            resolved_framework = "huggingface"
        elif detected_fw == "sklearn":
            resolved_framework = "sklearn"
        else:
            resolved_framework = detected_fw

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
        clean_input = model_id_or_path.strip()
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

        preset_found = cls.get_preset(clean_input)
        if preset_found is not None:
            preset = preset_found.copy()
            if matched_dataset:
                preset["dataset_name"] = matched_dataset
            return preset

        p = Path(clean_input)
        if p.is_file():
            inferred_domain = data_domain or detect_model_domain(clean_input) or "unknown"
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
        inferred_domain = data_domain or detect_model_domain(clean_input) or "unknown"
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
            num_features = getattr(cfg, "num_features", getattr(cfg, "n_features", None))
            if num_features is None and "/" in clean_input:
                try:
                    from huggingface_hub import hf_hub_download
                    import json
                    cfg_path = hf_hub_download(clean_input, "config.json", local_files_only=True)
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        raw_c = json.load(f)
                    cols = raw_c.get("sklearn", {}).get("columns") or raw_c.get("columns")
                    if cols and isinstance(cols, list):
                        num_features = len(cols)
                except Exception:
                    pass
        except Exception:
            seq_len = None
            num_channels = None
            num_features = None

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
            "num_features": num_features,
            "framework": "huggingface",
        }


__all__ = [
    "ModelPresetRegistry",
    "detect_model_domain",
    "detect_model_framework",
    "detect_model_source",
    "inspect_model_evidence",
    "is_forecasting_or_regression",
    "is_sklearn_model",
    "setup_model_environment",
    "log_ingestion_event",
]

