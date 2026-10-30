"""
Hugging Face model loader implementation for vision, text, time-series, and tabular models.
"""

import json
from typing import Any, Optional, Union
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import (
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoConfig,
    ViTConfig,
)

from app.ingestion.exceptions import ModelLoadError, UnsupportedTaskError
from app.ingestion.loader.base_loader import BaseModelLoader


def patch_hf_config(cfg_dict: dict) -> dict:
    """Normalize and patch id2label/label2id dictionaries in Hugging Face configs."""
    if "id2label" in cfg_dict and isinstance(cfg_dict["id2label"], dict):
        cfg_dict["id2label"] = {
            str(k): (str(v) if v is not None else "Unused")
            for k, v in cfg_dict["id2label"].items()
        }
        cfg_dict["label2id"] = {v: int(k) for k, v in cfg_dict["id2label"].items()}
        cfg_dict["num_labels"] = len(cfg_dict["id2label"])
    return cfg_dict



class HuggingFaceLoader(BaseModelLoader):
    """
    Loads Hugging Face models across supported domains (image, text, and time-series)
    from Hugging Face Model Hub or local model directories.
    """

    SUPPORTED_DOMAINS = {"image", "text", "time-series" , "tabular"}

    def load(
        self,
        model_path: Union[str, Any],
        device: Optional[Union[str, torch.device]] = None,
        domain: Optional[str] = None,
        task_type: Optional[str] = "classification",
        **kwargs: Any
    ) -> nn.Module:
        """
        Load a Hugging Face model.

        Args:
            model_path: Hugging Face hub repository ID (e.g. 'bazyl/gtsrb-model')
                        or directory path containing Hugging Face weights and config.json,
                        or an existing nn.Module instance.
            device: Target device to place model onto.
            domain: Domain identifier ('image' or 'text'). If omitted, auto-detected from model.
            **kwargs: Extra parameters passed to AutoModel loader.

        Returns:
            nn.Module: Loaded PyTorch model set in evaluation mode.
        """
        map_location = str(device) if device is not None else "cpu"

        # If already an nn.Module instance, move to device and return
        if isinstance(model_path, nn.Module):
            model_path.to(map_location)
            model_path.eval()
            return model_path

        model_id = str(model_path).strip()
        domain_str = str(domain).lower().strip().replace("_", "-") if domain is not None else ""
        if domain_str == "timeseries":
            domain_str = "time-series"

        if domain_str not in self.SUPPORTED_DOMAINS:
            # Attempt to auto-detect domain
            from app.ingestion.model_registry import detect_model_domain
            detected = detect_model_domain(model_id)
            if detected in self.SUPPORTED_DOMAINS:
                domain_str = detected
            else:
                domain_str = "unknown"

        if domain_str not in self.SUPPORTED_DOMAINS:
            raise ModelLoadError(
                f"Could not automatically determine model domain for Hugging Face model '{model_id}' (domain: '{domain_str}'). "
                f"Please explicitly specify domain='image', domain='text', or domain='time-series'."
            )

        try:
            # 1. Download/read config.json to perform config patching and task verification
            model_config = None
            cfg_dict = {}
            try:
                config_path = hf_hub_download(model_id, "config.json")
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg_dict = json.load(f)
                
                cfg_dict = patch_hf_config(cfg_dict)
                model_type = cfg_dict.get("model_type", "")
                if model_type == "vit":
                    model_config = ViTConfig.from_dict(cfg_dict)
                else:
                    model_config = AutoConfig.from_pretrained(model_id)
            except Exception:
                try:
                    model_config = AutoConfig.from_pretrained(model_id)
                except Exception:
                    model_config = None

            # Detect and reject forecasting/regression tasks for time-series models
            clean_task = str(task_type).lower().strip() if task_type else ""
            if clean_task in ("forecasting", "regression", "time-series-forecasting", "ts-forecasting"):
                raise UnsupportedTaskError(
                    "AdverScan currently supports time-series classification models only; forecasting models are not supported."
                )

            pipeline_tag = (
                cfg_dict.get("pipeline_tag")
                or getattr(model_config, "pipeline_tag", None)
                or getattr(model_config, "task", None)
            )
            if pipeline_tag:
                p_tag = str(pipeline_tag).lower().replace("_", "-")
                if any(k in p_tag for k in ("forecasting", "time-series-forecasting", "time-series-regression")):
                    raise UnsupportedTaskError(
                        "AdverScan currently supports time-series classification models only; forecasting models are not supported."
                    )

            # Check architecture names in config for forecasting / regression
            archs_list = []
            if model_config and getattr(model_config, "architectures", None):
                archs_list.extend([str(a) for a in model_config.architectures])
            if "architectures" in cfg_dict and isinstance(cfg_dict["architectures"], list):
                archs_list.extend([str(a) for a in cfg_dict["architectures"]])

            forecasting_arch_patterns = (
                "forprediction",
                "forforecasting",
                "forregression",
                "timeseriesprediction",
                "timeseriesforecasting",
                "timeseriesregression",
                "chronos",
                "timesfm",
            )
            for arch in archs_list:
                arch_clean = arch.lower()
                if any(pat in arch_clean for pat in forecasting_arch_patterns):
                    raise UnsupportedTaskError(
                        "AdverScan currently supports time-series classification models only; forecasting models are not supported."
                    )

            # 2. Domain-based loader selection
            model = None
            last_exc = None
            if domain_str == "text":
                try:
                    if model_config is not None:
                        model = AutoModelForSequenceClassification.from_pretrained(
                            model_id, config=model_config, **kwargs
                        )
                    else:
                        model = AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs)
                except Exception as e:
                    last_exc = e
                    try:
                        model = AutoModel.from_pretrained(model_id, trust_remote_code=True, **kwargs)
                    except Exception as e2:
                        last_exc = e2
                        # Fallback for gliner2 AutoExtractor
                        try:
                            from gliner2 import AutoExtractor
                            extractor = AutoExtractor.from_pretrained(model_id, **kwargs)
                            model = getattr(extractor, "encoder", getattr(extractor, "model", extractor))
                        except Exception as e3:
                            last_exc = e3
                            model = None

            elif domain_str == "image":
                try:
                    if model_config is not None:
                        model = AutoModelForImageClassification.from_pretrained(
                            model_id, config=model_config, **kwargs
                        )
                    else:
                        model = AutoModelForImageClassification.from_pretrained(model_id, **kwargs)
                except Exception as e:
                    last_exc = e
                    try:
                        model = AutoModel.from_pretrained(model_id, trust_remote_code=True, **kwargs)
                    except Exception as e2:
                        last_exc = e2
                        model = None

            elif domain_str == "time-series":
                import transformers

                # 1. Try architecture specified in model config
                archs = getattr(model_config, "architectures", []) or []
                for arch in archs:
                    if hasattr(transformers, arch):
                        cls = getattr(transformers, arch)
                        try:
                            if model_config is not None:
                                model = cls.from_pretrained(model_id, config=model_config, **kwargs)
                            else:
                                model = cls.from_pretrained(model_id, **kwargs)
                            if model is not None:
                                break
                        except Exception as e:
                            last_exc = e

                # 2. Try common time-series classification model classes in transformers
                if model is None:
                    ts_cls_names = [
                        "PatchTSTForClassification",
                        "PatchTSMixerForTimeSeriesClassification",
                        "AutoModelForSequenceClassification",
                    ]
                    for name in ts_cls_names:
                        if hasattr(transformers, name):
                            cls = getattr(transformers, name)
                            try:
                                if model_config is not None:
                                    model = cls.from_pretrained(model_id, config=model_config, **kwargs)
                                else:
                                    model = cls.from_pretrained(model_id, **kwargs)
                                if model is not None:
                                    break
                            except Exception as e:
                                last_exc = e

                # 3. Fallback to AutoModel
                if model is None:
                    try:
                        model = AutoModel.from_pretrained(model_id, trust_remote_code=True, **kwargs)
                    except Exception as e:
                        last_exc = e
                        model = None

            if model is None:
                err_detail = f": {last_exc}" if last_exc else "."
                raise ModelLoadError(f"Could not load Hugging Face model '{model_id}' for domain '{domain_str}'{err_detail}")

            model.to(map_location)
            model.eval()
            return model

        except (ModelLoadError, UnsupportedTaskError):
            raise
        except Exception as e:
            raise ModelLoadError(f"Failed to load Hugging Face model '{model_id}' for domain '{domain}': {str(e)}") from e