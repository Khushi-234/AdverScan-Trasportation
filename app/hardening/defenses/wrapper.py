"""
HardenedModelWrapper and StochasticInferenceWrapper for Module 7 (Hardening).

Provides common model wrappers:
- HardenedModelWrapper: wraps a base model to apply input transformations with protection
  against duplicate execution when already-hardened inputs are provided.
- StochasticInferenceWrapper: extends HardenedModelWrapper for stochastic defenses
  (e.g., randomized smoothing) using batched Monte Carlo sampling and probability aggregation.
"""

from typing import Any, Callable, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HardenedModelWrapper(nn.Module):
    """
    Generic wrapper that applies an input transformation prior to
    the base model's forward pass or inference.

    Supports both PyTorch nn.Module architectures and Scikit-learn / duck-typed
    estimators (exposing forward, predict, and predict_proba interfaces).

    Guards against duplicate application:
    If the input tensor was already hardened by this defense (e.g. passed as
    HardeningResult.hardened_inputs), it forwards directly to the base model without
    re-applying the transformation.
    """

    def __init__(
        self,
        model: Any,
        transform_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        defense_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.base_model = model  # convenient alias
        self.transform_fn = transform_fn
        self.defense_name = defense_name

    def _forward_model(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Execute forward pass on wrapped model, handling both PyTorch and Scikit-learn models."""
        if isinstance(self.model, nn.Module):
            return self.model(x, **kwargs)

        # Scikit-learn estimator or adapter
        if hasattr(self.model, "predict_proba"):
            x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
            probs = self.model.predict_proba(x_np)
            logits = np.log(np.clip(probs, 1e-12, 1.0))
            return torch.from_numpy(logits).float()
        elif hasattr(self.model, "decision_function"):
            x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
            dec = self.model.decision_function(x_np)
            if dec.ndim == 1:
                dec = np.column_stack([-dec, dec])
            return torch.from_numpy(dec).float()
        elif hasattr(self.model, "predict"):
            x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
            preds = self.model.predict(x_np)
            return torch.from_numpy(preds).float()
        elif callable(self.model):
            out = self.model(x, **kwargs)
            return out if isinstance(out, torch.Tensor) else torch.as_tensor(out).float()
        else:
            raise TypeError(f"Wrapped model of type {type(self.model)} is neither callable nor a valid estimator.")

    def forward(
        self,
        x: torch.Tensor,
        already_preprocessed: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass with double-application protection.

        Args:
            x: Input tensor.
            already_preprocessed: If True, bypasses transform_fn.
            **kwargs: Additional keyword arguments forwarded to the base model.

        Returns:
            torch.Tensor: Model output logits or predictions.
        """
        if already_preprocessed:
            return self._forward_model(x, **kwargs)

        # Check if tensor was already processed by this defense
        hardened_by = getattr(x, "_hardened_by", None)
        defense_keys = {self.defense_name, id(self.transform_fn)} - {None}

        if isinstance(hardened_by, set) and bool(hardened_by.intersection(defense_keys)):
            return self._forward_model(x, **kwargs)

        if self.transform_fn is not None:
            transformed_x = self.transform_fn(x)
        else:
            transformed_x = x
        return self._forward_model(transformed_x, **kwargs)

    def predict(self, x: Any, **kwargs: Any) -> Any:
        """
        Scikit-learn compatible prediction interface.
        Applies input transform before generating class predictions.
        """
        import numpy as np
        if self.transform_fn is not None:
            if isinstance(x, np.ndarray):
                x_t = torch.from_numpy(x).float()
                tx_t = self.transform_fn(x_t)
                x_trans = tx_t.detach().cpu().numpy()
            elif isinstance(x, torch.Tensor):
                tx_t = self.transform_fn(x)
                x_trans = tx_t.detach().cpu().numpy()
            else:
                x_trans = x
        else:
            x_trans = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

        if hasattr(self.model, "predict"):
            return self.model.predict(x_trans, **kwargs)
        out = self.forward(torch.as_tensor(x).float())
        return out.argmax(dim=-1).detach().cpu().numpy()

    def predict_proba(self, x: Any, **kwargs: Any) -> Any:
        """
        Scikit-learn compatible probability interface.
        Applies input transform before generating class probabilities.
        """
        import numpy as np
        if self.transform_fn is not None:
            if isinstance(x, np.ndarray):
                x_t = torch.from_numpy(x).float()
                tx_t = self.transform_fn(x_t)
                x_trans = tx_t.detach().cpu().numpy()
            elif isinstance(x, torch.Tensor):
                tx_t = self.transform_fn(x)
                x_trans = tx_t.detach().cpu().numpy()
            else:
                x_trans = x
        else:
            x_trans = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x_trans, **kwargs)
        out = self.forward(torch.as_tensor(x).float())
        return F.softmax(out, dim=-1).detach().cpu().numpy()

    @property
    def classes_(self) -> Optional[Any]:
        """Return classes_ if exposed by wrapped estimator."""
        return getattr(self.model, "classes_", None)


class StochasticInferenceWrapper(HardenedModelWrapper):
    """
    Common wrapper for stochastic inference defenses.

    Extends HardenedModelWrapper to support stochastic evaluation:
    - Generates multiple stochastic versions of the input.
    - Processes them in mini-batches (`batch_size`).
    - Collects model predictions and converts them to class probabilities.
    - Aggregates probabilities across stochastic samples via Monte Carlo estimation.
    - Guards against duplicate application if input was already preprocessed.

    Subclasses must implement `_perturb(inputs)`.
    """

    def __init__(
        self,
        base_model: Any,
        num_samples: int = 10,
        batch_size: int = 32,
        output_type: str = "prob",
        defense_name: Optional[str] = None,
    ) -> None:
        super().__init__(
            model=base_model,
            transform_fn=None,
            defense_name=defense_name,
        )

        if num_samples < 1:
            raise ValueError(f"num_samples must be at least 1, got {num_samples}")

        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")

        output_type_norm = str(output_type).lower()
        if output_type_norm not in {"prob", "log_prob", "logits"}:
            raise ValueError("output_type must be either 'prob', 'log_prob', or 'logits'")

        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.output_type = output_type_norm

    def _perturb(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Apply stochastic perturbation to inputs.

        Subclasses must implement this method.
        """
        raise NotImplementedError

    @staticmethod
    def _extract_logits(output: Any) -> torch.Tensor:
        """Extract logits from common model output formats (Tensor, tuple, or numpy array)."""
        import numpy as np
        if isinstance(output, tuple):
            output = output[0]

        if isinstance(output, np.ndarray):
            output = torch.from_numpy(output).float()

        if not isinstance(output, torch.Tensor):
            raise TypeError(
                "Base model output must be a torch.Tensor, numpy.ndarray, or a tuple whose first element is a Tensor/ndarray."
            )

        return output

    @staticmethod
    def _logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
        """Convert model logits to class probabilities for multiclass or binary."""
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)

        if logits.size(1) > 1:
            return F.softmax(logits, dim=-1)

        prob_pos = torch.sigmoid(logits)
        return torch.cat([1.0 - prob_pos, prob_pos], dim=-1)

    def _format_output(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Format aggregated probabilities according to configured output_type."""
        if self.output_type == "prob":
            return probabilities
        return torch.log(probabilities.clamp_min(1e-10))

    def forward(
        self,
        inputs: torch.Tensor,
        already_preprocessed: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass with double-application protection and stochastic aggregation.
        """
        # Double application check
        if already_preprocessed:
            return self._extract_logits(self.model(inputs, **kwargs))

        hardened_by = getattr(inputs, "_hardened_by", None)
        defense_keys = {self.defense_name} - {None}
        if isinstance(hardened_by, set) and bool(hardened_by.intersection(defense_keys)):
            return self._extract_logits(self.model(inputs, **kwargs))

        # Training mode: single-sample stochastic data augmentation with autograd enabled
        if self.training:
            perturbed_inputs = self._perturb(inputs)
            logits = self._extract_logits(self.model(perturbed_inputs, **kwargs))
            return logits

        # Evaluation / inference mode: batched Monte Carlo under torch.inference_mode()
        with torch.inference_mode():
            batch_size = inputs.size(0)
            feature_shape = inputs.shape[1:]

            expanded_inputs = (
                inputs
                .unsqueeze(1)
                .expand(batch_size, self.num_samples, *feature_shape)
                .reshape(batch_size * self.num_samples, *feature_shape)
            )

            total_samples = batch_size * self.num_samples
            probability_chunks: List[torch.Tensor] = []

            for start in range(0, total_samples, self.batch_size):
                end = min(start + self.batch_size, total_samples)
                chunk = expanded_inputs[start:end]

                perturbed_chunk = self._perturb(chunk)
                logits = self._extract_logits(self.model(perturbed_chunk, **kwargs))
                probs = self._logits_to_probabilities(logits)
                probability_chunks.append(probs)

            all_probabilities = torch.cat(probability_chunks, dim=0)
            averaged_probabilities = (
                all_probabilities
                .view(batch_size, self.num_samples, -1)
                .mean(dim=1)
            )

            return self._format_output(averaged_probabilities)
