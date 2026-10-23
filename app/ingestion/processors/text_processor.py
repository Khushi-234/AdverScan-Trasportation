"""
Text processor implementation for NLP and sequence classification models.
"""

from typing import Any, Dict, List, Optional, Union
import torch

from app.ingestion.processors.base_processor import BaseProcessor


class TextProcessor(BaseProcessor):
    """
    Standard text input processor.
    Prepares NLP model inputs (tokenized dictionaries with input_ids and attention_mask)
    using AutoTokenizer or fallback tokenization.
    """

    def __init__(
        self,
        processor_name_or_path: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
        max_length: int = 128,
    ):
        self.processor_name = processor_name_or_path
        self._device = torch.device(device) if device is not None else None
        self.max_length = max_length
        self._tokenizer = tokenizer

        if self._tokenizer is None and processor_name_or_path:
            p_str = str(processor_name_or_path).strip()
            if not p_str.endswith((".pt", ".pth", ".bin", ".ckpt", ".tar", ".gz")):
                try:
                    from transformers import AutoTokenizer
                    try:
                        self._tokenizer = AutoTokenizer.from_pretrained(
                            processor_name_or_path, local_files_only=True
                        )
                    except Exception:
                        if "/" in p_str:
                            self._tokenizer = AutoTokenizer.from_pretrained(
                                processor_name_or_path
                            )
                except Exception:
                    self._tokenizer = None


    @property
    def data_domain(self) -> str:
        return "text"

    def get_processor(self) -> Any:
        return self._tokenizer

    def process(self, inputs: Any, **kwargs: Any) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Process raw text inputs (strings, lists of strings, or token dicts)
        into tokenized tensors suitable for transformer text models.
        """
        from collections.abc import Mapping
        # If inputs is already a dict or mapping of tensors
        if isinstance(inputs, Mapping) or hasattr(inputs, "items"):
            output_dict = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    output_dict[k] = v.to(self._device) if self._device is not None else v
                else:
                    output_dict[k] = v
            return output_dict

        # If inputs is already a raw tensor (e.g. pre-tokenized input_ids)
        if isinstance(inputs, torch.Tensor):
            t = inputs.to(self._device) if self._device is not None else inputs
            return {"input_ids": t, "attention_mask": torch.ones_like(t)}

        # If inputs is str or list of str
        texts: List[str] = [inputs] if isinstance(inputs, str) else list(inputs)

        if self._tokenizer is not None:
            tokenized = self._tokenizer(
                texts,
                padding=kwargs.get("padding", True),
                truncation=kwargs.get("truncation", True),
                max_length=kwargs.get("max_length", self.max_length),
                return_tensors="pt",
            )
            if self._device is not None:
                tokenized = {k: v.to(self._device) for k, v in tokenized.items()}
            return tokenized

        # Fallback dummy tokenization for offline/testing scenarios
        # Encode character ASCII values up to max_length
        batch_ids = []
        batch_masks = []
        for text in texts:
            ascii_vals = [min(ord(c), 1000) for c in str(text)[: self.max_length]]
            if not ascii_vals:
                ascii_vals = [0]
            # Pad to max_length or longest
            mask = [1] * len(ascii_vals)
            pad_len = max(0, self.max_length - len(ascii_vals))
            ascii_vals.extend([0] * pad_len)
            mask.extend([0] * pad_len)
            batch_ids.append(ascii_vals)
            batch_masks.append(mask)

        res = {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_masks, dtype=torch.long),
        }
        if self._device is not None:
            res = {k: v.to(self._device) for k, v in res.items()}
        return res
