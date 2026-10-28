"""
Model loading utilities for GTSRB vision transformer model.
"""

from typing import Any, Tuple
from transformers import AutoModelForImageClassification


def load_gtsrb_vit_model(model_name: str = "bazyl/gtsrb-model") -> Tuple[Any, Any]:
    """
    Load GTSRB ViT model and configuration from Hugging Face Hub.
    """
    model = AutoModelForImageClassification.from_pretrained(model_name)
    return model, model.config
