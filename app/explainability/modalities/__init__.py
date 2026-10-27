"""
Modality handlers package for AdverScan XAI.
"""

from typing import Union
from app.explainability.modalities.base import BaseModalityHandler
from app.explainability.modalities.image import ImageModalityHandler
from app.explainability.modalities.text import TextModalityHandler
from app.explainability.modalities.tabular import TabularModalityHandler
from app.explainability.modalities.timeseries import TimeSeriesModalityHandler
from app.explainability.modality import Modality


def get_modality_handler(modality: Union[str, Modality], **kwargs) -> BaseModalityHandler:
    """
    Factory function to retrieve the appropriate modality handler.

    Args:
        modality: Modality enum or string.
        **kwargs: Modality-specific initialization arguments.

    Returns:
        Instance of BaseModalityHandler subclass.
    """
    m = Modality.from_str(modality)
    if m == Modality.IMAGE:
        return ImageModalityHandler()
    elif m == Modality.TEXT:
        return TextModalityHandler()
    elif m == Modality.TABULAR:
        return TabularModalityHandler(**kwargs)
    elif m == Modality.TIMESERIES:
        return TimeSeriesModalityHandler(**kwargs)
    else:
        raise ValueError(f"Unsupported modality handler: {m}")


__all__ = [
    "BaseModalityHandler",
    "ImageModalityHandler",
    "TextModalityHandler",
    "TabularModalityHandler",
    "TimeSeriesModalityHandler",
    "get_modality_handler",
]
