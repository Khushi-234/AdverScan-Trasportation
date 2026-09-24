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

from app.ingestion.exceptions import ModelLoadError
from app.ingestion.loader.base_loader import BaseModelLoader
from app.utils.model_utils import patch_hf_config


class HuggingFaceLoader(BaseModelLoader):
    """
    Loads Hugging Face models across multiple domains (Vision, NLP/Text, Time-Series, Tabular)
    from Hugging Face Model Hub or local model directories.
    """

    def load(
        self,
        model_path: Union[str, Any],
        device: Optional[Union[str, torch.device]] = None,
        domain: str = "image",
        **kwargs: Any
    ) -> nn.Module:
        """
        Load a Hugging Face model.

        Args:
            model_path: Hugging Face hub repository ID (e.g. 'bazyl/gtsrb-model')
                        or directory path containing Hugging Face weights and config.json.
            device: Target device to place model onto.
            domain: Domain identifier ('image', 'text', 'time_series', 'tabular').
            **kwargs: Extra parameters passed to AutoModel loader.

        Returns:
            nn.Module: Loaded PyTorch model set in evaluation mode.
        """
        map_location = str(device) if device is not None else "cpu"
        model_id = str(model_path).strip()
        domain_str = str(domain).lower().strip()

        try:
            # 1. Download/read config.json to perform config patching if needed
            cfg_dict = {}
            model_config = None
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
                model_config = None

            # 2. Domain-based loader selection with fallbacks
            model = None
            if "text" in domain_str or "nlp" in domain_str:
                try:
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_id, config=model_config, **kwargs
                    )
                except Exception:
                    pass

            if model is None:
                # Default / Vision domain loader
                try:
                    if model_config is not None:
                        model = AutoModelForImageClassification.from_pretrained(
                            model_id, config=model_config, use_safetensors=True, **kwargs
                        )
                    else:
                        model = AutoModelForImageClassification.from_pretrained(model_id, **kwargs)
                except Exception:
                    # Generic AutoModel fallback for text, time-series, tabular, or custom architectures
                    try:
                        model = AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs)
                    except Exception:
                        model = AutoModel.from_pretrained(model_id, **kwargs)

            model.to(map_location)
            model.eval()
            return model

        except Exception as e:
            raise ModelLoadError(f"Failed to load Hugging Face model '{model_id}' for domain '{domain}': {str(e)}") from e

