"""
Unit tests for InMemoryDatasetLoader helper in Module 8 (Orchestration).
"""

import torch
import pytest

from app.orchestration.dataset_adapter import InMemoryDatasetLoader
from app.evaluation.dataset_loader import BaseDatasetLoader


def test_in_memory_dataset_loader_interface():
    inputs = torch.randn(10, 3, 32, 32)
    targets = torch.randint(0, 5, (10,))
    loader = InMemoryDatasetLoader(inputs=inputs, targets=targets, dataset_name="test_data", batch_size=4)

    assert isinstance(loader, BaseDatasetLoader)
    assert loader.dataset_name == "test_data"
    assert len(loader) == 10

    batches = list(loader.iterate_batches())
    assert len(batches) == 3  # 4 + 4 + 2

    b1_pixels, b1_targets, b1_list = batches[0]
    assert b1_pixels.shape == (4, 3, 32, 32)
    assert b1_targets.shape == (4,)
    assert len(b1_list) == 4
    assert b1_list == targets[:4].tolist()
