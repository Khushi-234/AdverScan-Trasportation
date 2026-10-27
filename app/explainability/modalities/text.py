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

    def prepare_input(self, inputs: Any) -> Any:
        """
        Validate and format text inputs into strings or list of strings.
        """
        if isinstance(inputs, str):
            return inputs
        if isinstance(inputs, (list, tuple)):
            if all(isinstance(x, str) for x in inputs):
                return list(inputs)
            # If list of tokens, join into text
            return " ".join(str(x) for x in inputs)
        return str(inputs)

    def prepare_batch(self, clean: Any, adversarial: Any) -> Tuple[Any, Any]:
        """
        Prepare and validate clean and adversarial text inputs.
        """
        clean_text = self.prepare_input(clean)
        adv_text = self.prepare_input(adversarial)
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
        """
        if attribution is None:
            return {"attribution": None, "top_features": [], "summary": {}}

        arr = to_numpy(attribution, dtype=np.float32)
        arr = arr.ravel()

        tokens = self._tokenize(inputs)
        n_tokens = len(tokens)

        # Align length of attribution with tokens
        if len(arr) != n_tokens:
            if len(arr) > n_tokens:
                arr = arr[:n_tokens]
            else:
                arr = np.pad(arr, (0, n_tokens - len(arr)), mode="constant")

        top_features = self.get_top_features(arr, tokens, top_k=kwargs.get("top_k", 5))

        token_importance = [
            {"token": tok, "importance": float(score), "index": i}
            for i, (tok, score) in enumerate(zip(tokens, arr))
        ]

        return {
            "attribution": arr.tolist(),
            "tokens": tokens,
            "token_importance": token_importance,
            "top_features": top_features,
            "summary": {
                "total_tokens": n_tokens,
                "most_influential_token": top_features[0]["feature"] if top_features else None,
                "mean_token_importance": float(np.mean(np.abs(arr))) if len(arr) > 0 else 0.0,
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
        """
        arr = to_numpy(attribution, dtype=np.float32).ravel()
        tokens = self._tokenize(inputs) if not isinstance(inputs, list) or (inputs and not isinstance(inputs[0], str)) else inputs

        min_len = min(len(arr), len(tokens))
        arr = arr[:min_len]
        tokens = tokens[:min_len]

        sorted_indices = np.argsort(np.abs(arr))[::-1][:top_k]
        top_features = []

        for rank, idx in enumerate(sorted_indices):
            top_features.append({
                "feature": tokens[idx],
                "importance": float(arr[idx]),
                "rank": rank + 1,
                "index": int(idx),
            })

        return top_features
