"""
Input processors for AdverScan framework across image and text domains.
"""

from app.ingestion.processors.base_processor import BaseProcessor
from app.ingestion.processors.image_processor import ImageProcessor
from app.ingestion.processors.text_processor import TextProcessor

__all__ = ["BaseProcessor", "ImageProcessor", "TextProcessor"]
