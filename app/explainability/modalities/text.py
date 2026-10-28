"""
Text modality handler for natural language processing models in AdverScan XAI.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.explainability.input_utils import to_numpy
from app.explainability.modalities.base import BaseModalityHandler
from app.explainability.modality import Modality


class TextModalityHandler(BaseModalityHandler):
    """
    Handles text inputs and identifies important tokens or words contributing to predictions.
    """

    def __init__(self):
        super().__init__(modality=Modality.TEXT)

    def prepare_input(self, inputs: Any, tokenizer: Optional[Any] = None) -> Any:
        """
        Validate and format text inputs into strings or list of strings.
        Decodes token ID tensors or dictionaries using tokenizer if provided.
        """
        if isinstance(inputs, str):
            return inputs
        if isinstance(inputs, (list, tuple)):
            if all(isinstance(x, str) for x in inputs):
                return list(inputs)
            # If list of tokens, join into text
            return " ".join(str(x) for x in inputs)
        
        # Check dict / Mapping with text or input_ids
        if isinstance(inputs, dict) or hasattr(inputs, "items"):
            if "text" in inputs:
                return self.prepare_input(inputs["text"], tokenizer=tokenizer)
            if "input_ids" in inputs:
                ids = inputs["input_ids"]
                if tokenizer is not None and hasattr(tokenizer, "decode"):
                    try:
                        import torch
                        if isinstance(ids, torch.Tensor):
                            ids_cpu = ids.detach().cpu()
                            if ids_cpu.ndim == 2:
                                return [tokenizer.decode(row, skip_special_tokens=True) for row in ids_cpu]
                            return tokenizer.decode(ids_cpu, skip_special_tokens=True)
                        elif isinstance(ids, (list, tuple)):
                            return tokenizer.decode(ids, skip_special_tokens=True)
                    except Exception:
                        pass

        # Check raw tensor of token IDs
        try:
            import torch
            if isinstance(inputs, torch.Tensor) and tokenizer is not None and hasattr(tokenizer, "decode"):
                ids_cpu = inputs.detach().cpu()
                if ids_cpu.ndim == 2:
                    return [tokenizer.decode(row, skip_special_tokens=True) for row in ids_cpu]
                return tokenizer.decode(ids_cpu, skip_special_tokens=True)
        except Exception:
            pass

        return str(inputs)

    def prepare_batch(self, clean: Any, adversarial: Any, **kwargs: Any) -> Tuple[Any, Any]:
        """
        Prepare and validate clean and adversarial text inputs.
        """
        tok = kwargs.get("tokenizer")
        clean_text = self.prepare_input(clean, tokenizer=tok)
        adv_text = self.prepare_input(adversarial, tokenizer=tok)
        return clean_text, adv_text

    def _tokenize(self, text: Any) -> List[str]:
        """Split text into whitespace/punctuation tokens."""
        if isinstance(text, str):
            return text.split()
        if isinstance(text, (list, tuple)):
            if len(text) > 0 and isinstance(text[0], str):
                if len(text) == 1:
                    return text[0].split()
                return list(text)
        return [str(text)]

    def postprocess_attribution(
        self,
        attribution: Any,
        inputs: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Map raw attribution weights to words/tokens and generate token-level rankings.
        Preserves full-precision values, detects non-finite and near-zero attributions,
        validates token-to-attribution alignment, and avoids misleading top-feature rankings.
        """
        if attribution is None:
            return {
                "attribution": None,
                "top_features": [],
                "summary": {},
                "attribution_validation": {
                    "status": "unavailable",
                    "is_finite": False,
                    "is_near_zero": False,
                    "token_alignment_status": "unavailable",
                    "saliency_detected": False,
                },
            }

        arr = to_numpy(attribution, dtype=np.float64)
        arr = arr.ravel()

        tokens = self._tokenize(inputs)
        n_tokens = len(tokens)
        raw_len = len(arr)

        is_empty = bool(arr.size == 0)
        is_finite = bool(np.all(np.isfinite(arr)))
        if not is_finite:
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        is_near_zero = bool(is_empty or np.all(np.abs(arr) < 1e-12) or np.max(np.abs(arr)) < 1e-12)

        # Validate token alignment status
        if raw_len == n_tokens:
            token_alignment_status = "exact"
        elif raw_len > n_tokens:
            token_alignment_status = "truncated"
            arr = arr[:n_tokens]
        else:
            token_alignment_status = "padded"
            arr = np.pad(arr, (0, n_tokens - raw_len), mode="constant")

        val_status = "valid"
        if is_empty:
            val_status = "empty"
        elif not is_finite:
            val_status = "non_finite"
        elif is_near_zero:
            val_status = "near_zero"

        # Avoid misleading top feature rankings when attributions are near-zero or non-finite
        if is_near_zero or not is_finite or is_empty:
            top_features: List[Dict[str, Any]] = []
        else:
            top_features = self.get_top_features(arr, tokens, top_k=kwargs.get("top_k", 5))

        token_importance = [
            {"token": tok, "importance": float(score), "index": i}
            for i, (tok, score) in enumerate(zip(tokens, arr))
        ]

        validation_info = {
            "status": val_status,
            "is_finite": is_finite,
            "is_near_zero": is_near_zero,
            "is_empty": is_empty,
            "token_alignment_status": token_alignment_status,
            "raw_attribution_norm": float(np.linalg.norm(arr)) if is_finite else 0.0,
            "saliency_detected": bool(not is_near_zero and is_finite and not is_empty),
        }

        return {
            "attribution": arr.tolist(),
            "tokens": tokens,
            "token_importance": token_importance,
            "top_features": top_features,
            "attribution_validation": validation_info,
            "summary": {
                "total_tokens": n_tokens,
                "most_influential_token": top_features[0]["feature"] if top_features else None,
                "mean_token_importance": float(np.mean(np.abs(arr))) if len(arr) > 0 else 0.0,
                "saliency_detected": validation_info["saliency_detected"],
            },
        }

    def get_top_features(
        self,
        attribution: Any,
        inputs: Any,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Rank tokens by absolute attribution score.
        Avoids misleading rankings when attributions are near-zero or non-finite.
        """
        arr = to_numpy(attribution, dtype=np.float64).ravel()
        if not np.all(np.isfinite(arr)):
            return []
        if arr.size == 0 or np.all(np.abs(arr) < 1e-12) or np.max(np.abs(arr)) < 1e-12:
            return []

        tokens = self._tokenize(inputs) if not isinstance(inputs, list) or (inputs and not isinstance(inputs[0], str)) else inputs

        min_len = min(len(arr), len(tokens))
        arr = arr[:min_len]
        tokens = tokens[:min_len]

        sorted_indices = np.argsort(np.abs(arr))[::-1][:top_k]
        top_features = []

        for rank, idx in enumerate(sorted_indices):
            # Do not include features with exactly zero attribution if others exist
            if abs(float(arr[idx])) < 1e-12:
                continue
            top_features.append({
                "feature": tokens[idx],
                "importance": float(arr[idx]),
                "rank": rank + 1,
                "index": int(idx),
            })

        return top_features
