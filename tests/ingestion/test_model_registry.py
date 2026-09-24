"""
Unit tests for model registry, environment setup, and ingestion activity logging.
"""

import json
import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from app.ingestion.model_registry import (
    ModelPresetRegistry,
    setup_model_environment,
    log_ingestion_event,
)
from app.ingestion.pipeline import ingest_model


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


def test_preset_registry_list_and_get():
    presets = ModelPresetRegistry.list_presets()
    assert len(presets) >= 2
    preset_ids = [p["model_id"] for p in presets]
    assert "bazyl/gtsrb-model" in preset_ids
    assert "ashutoshsharma58/scenery3214154" in preset_ids

    intel_preset = ModelPresetRegistry.get_preset("ashutoshsharma58/scenery3214154")
    assert intel_preset is not None
    assert intel_preset["num_classes"] == 6
    assert intel_preset["data_domain"] == "image"
    assert "Vision Transformer" in intel_preset["model_type"]
    assert "Scene & Natural Landscape" in intel_preset["domain"]


def test_resolve_model_info_custom():
    custom_info = ModelPresetRegistry.resolve_model_info("facebook/convnext-base-224")
    assert custom_info["model_id"] == "facebook/convnext-base-224"
    assert custom_info["framework"] == "huggingface"
    assert "domain" in custom_info


def test_resolve_model_info_kaggle_url():
    url = "https://www.kaggle.com/datasets/puneet6060/intel-image-classification"
    info = ModelPresetRegistry.resolve_model_info(url)
    assert info["model_id"] == "ashutoshsharma58/scenery3214154"
    assert info["num_classes"] == 6
    assert info["dataset_name"] == "puneet6060/intel-image-classification"




def test_log_ingestion_event():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "ingestion_log.json"
        
        entry_1 = {"model_name": "TestModel1", "status": "INGESTION_SUCCESS"}
        log_path = log_ingestion_event(entry_1, log_filepath=log_file)

        assert Path(log_path).exists()
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]["model_name"] == "TestModel1"

        entry_2 = {"model_name": "TestModel2", "status": "INGESTION_SUCCESS"}
        log_ingestion_event(entry_2, log_filepath=log_file)
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert len(data) == 2
            assert data[1]["model_name"] == "TestModel2"


def test_setup_model_environment():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test_env_log.json"
        env = setup_model_environment(
            "bazyl/gtsrb-model",
            device="cpu",
            log_filepath=log_file,
        )

        assert env["status"] == "ENVIRONMENT_CONFIGURED"
        assert env["recommended_dataset"] == "bazyl/GTSRB"
        assert env["target_device"] == "cpu"
        assert log_file.exists()


def test_ingest_model_with_logging():
    model = DummyModel()
    sample_input = torch.randn(2, 10)

    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "ingestion_test.json"
        adapter, metadata = ingest_model(
            model_path=model,
            sample_input=sample_input,
            model_name="DummyTest",
            device="cpu",
            log_ingestion=True,
            log_filepath=log_file,
        )

        assert metadata.model_name == "DummyTest"
        assert log_file.exists()
        with open(log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
            assert len(logs) == 1
            assert logs[0]["model_name"] == "DummyTest"
