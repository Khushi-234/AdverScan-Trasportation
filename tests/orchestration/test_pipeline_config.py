"""
Unit tests for PipelineConfig dataclass, validation logic, serialization, and hashing.
"""

import tempfile
from pathlib import Path
import pytest
from app.orchestration.pipeline_config import PipelineConfig


def test_pipeline_config_defaults():
    config = PipelineConfig(model_path="dummy_path.pt")
    assert config.model_path == "dummy_path.pt"
    assert config.mode == "full"
    assert config.attacks == ["fgsm"]
    assert config.enable_xai is False
    assert config.enable_hardening is False
    assert config.seed == 42
    assert config.deterministic is False
    assert config.sample_count is None
    assert config.experiment_name == "Experiment"
    assert config.experiment_id is None
    assert config.description is None
    assert "accuracy" in config.evaluation_metrics
    config.validate()


def test_pipeline_config_experiment_identity():
    config = PipelineConfig(
        model_path="dummy.pt",
        experiment_id="EXP-101",
        experiment_name="Robustness Baseline",
        description="Testing FGSM robustness",
    )
    assert config.experiment_id == "EXP-101"
    assert config.experiment_name == "Robustness Baseline"
    assert config.description == "Testing FGSM robustness"
    config.validate()


def test_pipeline_config_invalid_experiment_id():
    config = PipelineConfig(model_path="dummy.pt", experiment_id="   ")
    with pytest.raises(ValueError, match="experiment_id must be a non-empty string"):
        config.validate()


def test_pipeline_config_invalid_mode():
    config = PipelineConfig(model_path="dummy.pt", mode="invalid_mode")
    with pytest.raises(ValueError, match="Invalid mode"):
        config.validate()


def test_pipeline_config_empty_attacks():
    config = PipelineConfig(model_path="dummy.pt", mode="full", attacks=[])
    with pytest.raises(ValueError, match="At least one attack must be specified"):
        config.validate()


def test_pipeline_config_invalid_xai():
    config = PipelineConfig(model_path="dummy.pt", mode="full", xai_techniques=["unsupported_xai"])
    with pytest.raises(ValueError, match="Unsupported XAI technique"):
        config.validate()


def test_pipeline_config_seed_validation():
    valid_config = PipelineConfig(model_path="dummy.pt", seed=12345)
    valid_config.validate()

    invalid_config = PipelineConfig(model_path="dummy.pt", seed=-5)
    with pytest.raises(ValueError, match="Invalid seed"):
        invalid_config.validate()


def test_pipeline_config_sample_count_validation():
    valid_config = PipelineConfig(model_path="dummy.pt", sample_count=100)
    valid_config.validate()

    zero_config = PipelineConfig(model_path="dummy.pt", sample_count=0)
    with pytest.raises(ValueError, match="Invalid sample_count"):
        zero_config.validate()

    neg_config = PipelineConfig(model_path="dummy.pt", sample_count=-10)
    with pytest.raises(ValueError, match="Invalid sample_count"):
        neg_config.validate()


def test_pipeline_config_metric_validation():
    valid_config = PipelineConfig(
        model_path="dummy.pt",
        evaluation_metrics=["accuracy", "f1_macro", "confusion_matrix"]
    )
    valid_config.validate()

    invalid_config = PipelineConfig(
        model_path="dummy.pt",
        evaluation_metrics=["accuracy", "invalid_metric_name"]
    )
    with pytest.raises(ValueError, match="Unsupported evaluation metric"):
        invalid_config.validate()


def test_pipeline_config_dict_serialization():
    config = PipelineConfig(
        model_path="dummy.pt",
        experiment_id="EXP-200",
        sample_count=50,
        attack_configs={"fgsm": {"epsilon": 0.05}}
    )
    d = config.to_dict()
    assert d["model_path"] == "dummy.pt"
    assert d["experiment_id"] == "EXP-200"
    assert d["sample_count"] == 50
    assert d["attack_configs"] == {"fgsm": {"epsilon": 0.05}}

    reconstructed = PipelineConfig.from_dict(d)
    assert reconstructed.model_path == "dummy.pt"
    assert reconstructed.experiment_id == "EXP-200"
    assert reconstructed.sample_count == 50
    assert reconstructed.attack_configs == {"fgsm": {"epsilon": 0.05}}


def test_pipeline_config_json_roundtrip():
    config = PipelineConfig(
        model_path="dummy.pt",
        experiment_id="EXP-JSON",
        seed=100,
        sample_count=200,
        attack_configs={"fgsm": {"epsilon": 0.1}}
    )
    json_str = config.to_json()
    reconstructed = PipelineConfig.from_json(json_str)
    assert reconstructed.experiment_id == "EXP-JSON"
    assert reconstructed.seed == 100
    assert reconstructed.sample_count == 200

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "config.json"
        config.to_json(tmp_file)
        assert tmp_file.exists()
        file_reconstructed = PipelineConfig.from_json(tmp_file)
        assert file_reconstructed.experiment_id == "EXP-JSON"


def test_pipeline_config_yaml_roundtrip():
    config = PipelineConfig(
        model_path="dummy.pt",
        experiment_id="EXP-YAML",
        seed=200,
        sample_count=300
    )
    yaml_str = config.to_yaml()
    reconstructed = PipelineConfig.from_yaml(yaml_str)
    assert reconstructed.experiment_id == "EXP-YAML"
    assert reconstructed.seed == 200
    assert reconstructed.sample_count == 300

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "config.yaml"
        config.to_yaml(tmp_file)
        assert tmp_file.exists()
        file_reconstructed = PipelineConfig.from_yaml(tmp_file)
        assert file_reconstructed.experiment_id == "EXP-YAML"


def test_pipeline_config_hashing():
    config1 = PipelineConfig(model_path="dummy.pt", dataset_name="GTSRB", sample_count=100)
    config2 = PipelineConfig(model_path="dummy.pt", dataset_name="GTSRB", sample_count=100)
    config3 = PipelineConfig(model_path="dummy.pt", dataset_name="GTSRB", sample_count=200)

    hash1 = config1.get_configuration_hash()
    hash2 = config2.get_configuration_hash()
    hash3 = config3.get_configuration_hash()

    assert isinstance(hash1, str)
    assert len(hash1) == 32
    assert hash1 == hash2
    assert hash1 != hash3

