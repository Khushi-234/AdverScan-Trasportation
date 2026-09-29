"""
Text dataset loader for Hugging Face NLP and text classification datasets.
"""

from typing import Any, Generator, List, Optional, Tuple
import torch
from datasets import load_dataset

from app.evaluation.dataset_loader.base_dataset_loader import BaseDatasetLoader


class HFTextDatasetLoader(BaseDatasetLoader):
    """
    Generalized dataset loader for Hugging Face text / NLP classification datasets (e.g., IMDb, SST-2, AG News).
    Discovers text and label columns, tokenizes inputs using Hugging Face AutoTokenizer, and yields mini-batches.
    """

    TEXT_CANDIDATES = ["text", "sentence", "content", "premise", "input", "document", "review"]
    LABEL_CANDIDATES = ["label", "labels", "target", "class", "category"]

    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: Optional[str] = None,
        split: str = "test",
        batch_size: int = 32,
        max_length: int = 512,
    ):
        self._dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.split = split
        self.batch_size = batch_size
        self.max_length = max_length

        self._dataset = load_dataset(dataset_name, split=split)

        self._tokenizer = None
        if tokenizer_name:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load AutoTokenizer for '{tokenizer_name}': {e}"
                ) from e

        self.text_col = self._find_column(self.TEXT_CANDIDATES, default="text")
        self.label_col = self._find_column(self.LABEL_CANDIDATES, default=None)
        if self.label_col is None:
            raise KeyError(
                f"Dataset '{self.dataset_name}' does not contain any recognized label column. "
                f"Candidates checked: {self.LABEL_CANDIDATES}. "
                f"Available columns: {getattr(self._dataset, 'column_names', [])}"
            )

    def _find_column(self, candidates: List[str], default: Optional[str] = None) -> Optional[str]:
        if hasattr(self._dataset, "column_names") and self._dataset.column_names:
            cols = self._dataset.column_names
            for c in candidates:
                if c in cols:
                    return c
        if len(self._dataset) > 0:
            sample = self._dataset[0]
            if isinstance(sample, dict):
                for c in candidates:
                    if c in sample:
                        return c
        return default

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    def __len__(self) -> int:
        return len(self._dataset)

    def iterate_batches(
        self,
    ) -> Generator[Tuple[Any, torch.Tensor, List[int]], None, None]:
        total_samples = len(self._dataset)

        for i in range(0, total_samples, self.batch_size):
            batch_samples = self._dataset[i : i + self.batch_size]

            if isinstance(batch_samples, dict):
                first_key = next(iter(batch_samples))
                num_items = len(batch_samples[first_key])
                items = [
                    {key: batch_samples[key][j] for key in batch_samples}
                    for j in range(num_items)
                ]
            else:
                items = batch_samples

            texts: List[str] = []
            targets: List[int] = []
            for item in items:
                if not getattr(self, "label_col", None) or self.label_col not in item:
                    raise KeyError(
                        f"Dataset '{self.dataset_name}' sample is missing required label column '{getattr(self, 'label_col', None)}'. "
                        f"Available columns: {list(item.keys())}"
                    )
                texts.append(str(item.get(self.text_col, "")))
                targets.append(int(item[self.label_col]))

            if self._tokenizer is not None:
                encoded = self._tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                batch_inputs = encoded
            else:
                batch_inputs = texts

            targets_tensor = torch.tensor(targets, dtype=torch.long)
            yield batch_inputs, targets_tensor, targets
