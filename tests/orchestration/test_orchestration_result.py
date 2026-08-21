"""
Unit tests for OrchestrationResult DTO and serialization logic.
"""

from pathlib import Path
import json
import pytest

from app.orchestration.orchestration_result import OrchestrationResult


def test_orchestration_result_to_dict_and_save(tmp_path: Path):
    result = OrchestrationResult(
        status="SUCCESS",
        execution_mode="full",
        execution_time_seconds=1.23,
        timestamp="2026-08-21 12:00:00",
        model_metadata={"model_name": "TestNet", "framework": "pytorch"},
        baseline_evaluation={"accuracy": 0.95},
        attack_results={"fgsm": {"attack_name": "fgsm"}},
        adversarial_evaluations={"fgsm": {"accuracy": 0.40}},
        vulnerability_analysis={"fgsm": {"vulnerability_score": 75.0}},
        errors=[],
    )

    res_dict = result.to_dict()
    assert res_dict["status"] == "SUCCESS"
    assert res_dict["execution_mode"] == "full"
    assert res_dict["model_metadata"]["model_name"] == "TestNet"

    out_file = tmp_path / "result.json"
    result.save_json(out_file)

    assert out_file.exists()
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "SUCCESS"
    assert data["vulnerability_analysis"]["fgsm"]["vulnerability_score"] == 75.0
