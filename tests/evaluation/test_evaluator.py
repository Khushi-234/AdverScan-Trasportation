"""
Unit and integration tests for BaselineEvaluator workflow using mock adapters and loaders.
"""

from typing import Any, Generator, List, Tuple
import numpy as np
import pytest
import torch
import torch.nn as nn

from app.ingestion.adapters.base_adapter import BaseModelAdapter
from app.evaluation.evaluator import BaselineEvaluator
from app.evaluation.results import EvaluationResult


class MockModel(nn.Module):
    """Simple neural network returning 44 logits (43 GTSRB + 1 extra)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 44)

    def forward(self, x):
        return self.linear(x)


class MockAdapter(BaseModelAdapter):
    """Mock adapter wrapping MockModel for testing BaselineEvaluator."""

    def __init__(self):
        self._model = MockModel()

    def predict(self, inputs: Any, return_numpy: bool = False) -> Any:
        self._model.eval()
        with torch.no_grad():
            if isinstance(inputs, torch.Tensor):
                out = self._model(inputs)
            else:
                out = self._model(torch.randn(len(inputs), 10))
        if return_numpy:
            return out.cpu().numpy()
        return out

    def get_model(self) -> Any:
        return self._model

    def to(self, device: Any) -> "MockAdapter":
        return self

    def eval(self) -> "MockAdapter":
        self._model.eval()
        return self

    def train(self, mode: bool = True) -> "MockAdapter":
        self._model.train(mode)
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)


class MockDatasetLoader:
    """Mock dataset loader yielding 2 mini-batches of synthetic data."""

    def __init__(self, batch_size: int = 4, num_samples: int = 8):
        self.dataset_name = "mock/GTSRB"
        self.batch_size = batch_size
        self.num_samples = num_samples

    def iterate_batches(
        self,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, List[int]], None, None]:
        for _ in range(self.num_samples // self.batch_size):
            pixel_values = torch.randn(self.batch_size, 10)
            targets = torch.tensor([0, 1, 2, 3][: self.batch_size], dtype=torch.long)
            yield pixel_values, targets, targets.tolist()


def test_baseline_evaluator_workflow():
    """Test full BaselineEvaluator workflow with mock adapter and mock loader."""
    adapter = MockAdapter()
    loader = MockDatasetLoader(batch_size=4, num_samples=8)

    evaluator = BaselineEvaluator(
        adapter=adapter,
        dataset_loader=loader,  # type: ignore[arg-type]
        num_classes=43,
        model_name="MockViTModel",
    )

    result = evaluator.evaluate(output_dir=None)

    assert isinstance(result, EvaluationResult)
    assert result.dataset_name == "mock/GTSRB"
    assert result.model_name == "MockViTModel"
    assert result.num_samples == 8
    assert result.num_classes == 43
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.average_confidence <= 1.0
    assert result.average_entropy >= 0.0
    assert len(result.confusion_matrix) == 43
    assert len(result.confusion_matrix[0]) == 43


def test_logit_slicing_and_prediction_extraction():
    """Test slicing 44 logits down to 43 active classes and extracting predictions."""
    adapter = MockAdapter()
    loader = MockDatasetLoader(batch_size=4, num_samples=4)
    evaluator = BaselineEvaluator(adapter=adapter, dataset_loader=loader, num_classes=43)

    # 44-logit tensor (e.g. ViT HF output)
    raw_logits = torch.randn(4, 44)
    sliced_logits = raw_logits[:, :43]
    probs = torch.softmax(sliced_logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)

    assert sliced_logits.shape == (4, 43)
    assert probs.shape == (4, 43)
    assert preds.shape == (4,)
    assert torch.all(preds < 43)
    assert torch.all(preds >= 0)


def test_baseline_evaluator_invalid_adapter():
    """Test error when BaselineEvaluator is initialized with invalid adapter."""
    loader = MockDatasetLoader()
    with pytest.raises(TypeError, match="Expected adapter instance of BaseModelAdapter"):
        BaselineEvaluator(
            adapter="not_an_adapter",  # type: ignore[arg-type]
            dataset_loader=loader,  # type: ignore[arg-type]
        )

