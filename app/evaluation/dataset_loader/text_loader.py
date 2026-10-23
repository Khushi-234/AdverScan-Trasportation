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

    TEXT_CANDIDATES = ["text", "sentence", "content", "premise", "input", "document", "review", "utterance", "query", "passage"]
    LABEL_CANDIDATES = ["label", "labels", "target", "class", "category", "sentiment", "intent", "topic", "tag"]

    DATASET_ALIASES = {
        "sst2": "stanfordnlp/sst2",
        "glue/sst2": "stanfordnlp/sst2",
        "imdb": "stanfordnlp/imdb",
        "ag_news": "fancyzhx/ag_news",
        "cardiffnlp/tweet_eval": ("cardiffnlp/tweet_eval", "sentiment"),
        "tweet_eval": ("cardiffnlp/tweet_eval", "sentiment"),
    }

    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: Optional[str] = None,
        split: str = "test",
        batch_size: int = 32,
        max_length: int = 512,
    ):
        resolved_name = self.DATASET_ALIASES.get(dataset_name, dataset_name)
        self._dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.split = split
        self.batch_size = batch_size
        self.max_length = max_length

        def _load_hf_ds(target, split="test"):
            if isinstance(target, (tuple, list)):
                return load_dataset(target[0], target[1], split=split)
            return load_dataset(target, split=split)

        from pathlib import Path
        local_p = Path(dataset_name)
        if local_p.is_file():
            suffix = local_p.suffix.lower()
            import pandas as pd
            from datasets import Dataset

            if suffix == ".txt":
                with open(local_p, "r", encoding="utf-8", errors="ignore") as f:
                    lines = [line.strip() for line in f if line.strip()]
                if not lines:
                    raise ValueError(f"Text dataset file '{local_p}' is empty.")
                self._dataset = Dataset.from_dict({"text": lines, "label": [0] * len(lines)})
            elif suffix in (".csv", ".tsv"):
                df = pd.read_csv(local_p, sep="\t" if suffix == ".tsv" else ",")
                if df.empty:
                    raise ValueError(f"Text dataset file '{local_p}' is empty.")
                self._dataset = Dataset.from_pandas(df)
            elif suffix in (".json", ".jsonl"):
                lines_mode = suffix == ".jsonl"
                try:
                    df = pd.read_json(local_p, lines=lines_mode)
                except Exception:
                    df = pd.read_json(local_p, lines=not lines_mode)
                if df.empty:
                    raise ValueError(f"Text dataset file '{local_p}' is empty.")
                self._dataset = Dataset.from_pandas(df)
            else:
                raise ValueError(
                    f"Unsupported text file format '{suffix}'. "
                    f"Supported text formats: .txt, .csv, .tsv, .json, .jsonl"
                )
        else:
            try:
                self._dataset = _load_hf_ds(resolved_name, split=split)
            except Exception as e:
                if "Config name is missing" in str(e) and not isinstance(resolved_name, (tuple, list)):
                    resolved_name = (resolved_name, "sentiment")
                    self._dataset = _load_hf_ds(resolved_name, split=split)
                else:
                    slice_suffix = ""
                    if "[" in split and "]" in split:
                        slice_suffix = split[split.index("[") : split.index("]") + 1]
                    try:
                        fallback_split = f"validation{slice_suffix}"
                        self._dataset = _load_hf_ds(resolved_name, split=fallback_split)
                        self.split = fallback_split
                    except Exception:
                        self._dataset = _load_hf_ds(resolved_name, split="train")
                        self.split = "train"

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

        # If test split has dummy labels (-1, e.g. GLUE test sets), fallback to validation split
        if self.label_col and len(self._dataset) > 0 and self._dataset[0].get(self.label_col) == -1:
            slice_suffix = ""
            if "[" in self.split and "]" in self.split:
                slice_suffix = self.split[self.split.index("[") : self.split.index("]") + 1]
            try:
                val_split = f"validation{slice_suffix}"
                val_ds = _load_hf_ds(resolved_name, split=val_split)
                if len(val_ds) > 0 and val_ds[0].get(self.label_col) != -1:
                    self._dataset = val_ds
                    self.split = val_split
            except Exception:
                pass

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
