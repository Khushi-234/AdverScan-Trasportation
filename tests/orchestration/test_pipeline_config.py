"""
Unit tests for PipelineConfig dataclass and validation logic.
"""

import pytest
from app.orchestration.pipeline_config import PipelineConfig


def test_pipeline_config_defaults():
    config = PipelineConfig(model_path="dummy_path.pt")
    assert config.model_path == "dummy_path.pt"
    assert config.mode == "full"
    assert config.attacks == ["fgsm"]
    assert config.enable_xai is False
    assert config.enable_hardening is False
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
