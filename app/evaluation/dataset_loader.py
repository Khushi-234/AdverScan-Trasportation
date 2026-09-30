"""
Dataset loading and batch processing for GTSRB baseline evaluation in AdverScan.
"""

from abc import ABC, abstractmethod
import io
from typing import Any, Callable, Generator, List, Optional, Tuple
import torch
from PIL import Image
from datasets import load_dataset
from transformers import AutoImageProcessor


class BaseDatasetLoader(ABC):
    """
    Abstract base dataset loader for AdverScan evaluation modules.
    Allows generic domain support (ITS, Financial, Medical, etc.).
    """

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Get dataset identifier."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get total number of samples in dataset split."""
        pass

    @abstractmethod
    def iterate_batches(
        self,
    ) -> Generator[Tuple[Any, Any, List[int]], None, None]:
        """Yield (batch_inputs, batch_targets, target_labels_list)."""
        pass


class GTSRBDatasetLoader(BaseDatasetLoader):
    """
    Dataset loader for Hugging Face GTSRB dataset (bazyl/GTSRB).
    Decodes image payloads, applies image processor, and yields mini-batches.
    """

    def __init__(
        self,
        dataset_name: str = "bazyl/GTSRB",
        processor_name: str = "bazyl/gtsrb-model",
        split: str = "test",
        batch_size: int = 32,
    ):
        """
        Initialize GTSRB dataset loader.

        Args:
            dataset_name: Hugging Face dataset identifier.
            processor_name: Hugging Face image processor model identifier.
            split: Dataset split to evaluate ('test' or 'train').
            batch_size: Evaluation batch size.
        """
        self._dataset_name = dataset_name
        self.processor_name = processor_name
        self.split = split
        self.batch_size = batch_size

        # Load Hugging Face dataset split
        self._dataset = load_dataset(dataset_name, split=split)
        self._processor = AutoImageProcessor.from_pretrained(processor_name)

    @property
    def dataset_name(self) -> str:
        """Get dataset identifier."""
        return self._dataset_name

    @property
    def processor(self) -> Any:
        """Get image processor instance."""
        return self._processor

    def __len__(self) -> int:
        """Total number of samples in split."""
        return len(self._dataset)

    def _decode_image(self, sample: dict) -> Image.Image:
        """Decode PIL RGB image from raw bytes payload."""
        if "Path" in sample and isinstance(sample["Path"], dict) and "bytes" in sample["Path"]:
            image_bytes = sample["Path"]["bytes"]
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        elif "image" in sample:
            img = sample["image"]
            if isinstance(img, Image.Image):
                return img.convert("RGB")
            return Image.open(img).convert("RGB")
        else:
            raise KeyError("Dataset sample does not contain a valid image payload or path bytes.")

    def iterate_batches(
        self,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, List[int]], None, None]:
        """
        Yield mini-batches of preprocessed tensors and ground-truth targets.

        Yields:
            Tuple of (pixel_values_tensor, target_labels_tensor, class_ids_list)
        """
        total_samples = len(self._dataset)

        for i in range(0, total_samples, self.batch_size):
            batch_samples = self._dataset[i : i + self.batch_size]
            
            # Reconstruct list of dicts if Hugging Face dataset returns dict of lists
            if isinstance(batch_samples, dict):
                num_items = len(batch_samples["ClassId"])
                items = [
                    {key: batch_samples[key][j] for key in batch_samples}
                    for j in range(num_items)
                ]
            else:
                items = batch_samples

            images: List[Image.Image] = []
            targets: List[int] = []

            for item in items:
                img = self._decode_image(item)
                class_id = item["ClassId"]
                images.append(img)
                targets.append(class_id)

            processed = self._processor(images=images, return_tensors="pt")
            pixel_values = processed["pixel_values"]
            targets_tensor = torch.tensor(targets, dtype=torch.long)

            yield pixel_values, targets_tensor, targets
