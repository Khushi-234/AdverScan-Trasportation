"""
Unit tests for EvaluationResult dataclass and serialization.
"""

import tempfile
from pathlib import Path
from app.evaluation.results import EvaluationResult


def test_evaluation_result_instantiation():
    """Test instantiating EvaluationResult with metrics."""
    result = EvaluationResult(
        dataset_name="bazyl/GTSRB",
        model_name="bazyl/gtsrb-model",
        num_samples=100,
        num_classes=43,
        accuracy=0.95,
        precision_macro=0.94,
        recall_macro=0.93,
        f1_macro=0.935,
        precision_weighted=0.95,
        recall_weighted=0.95,
        f1_weighted=0.95,
        average_confidence=0.98,
        average_entropy=0.15,
        per_class_metrics={"0": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 10}},
        confusion_matrix=[[10, 0], [0, 10]],
    )

    assert result.dataset_name == "bazyl/GTSRB"
    assert result.accuracy == 0.95
    assert result.num_samples == 100
    assert result.num_classes == 43


def test_evaluation_result_json_serialization():
    """Test to_dict and save_json methods of EvaluationResult."""
    result = EvaluationResult(
        dataset_name="bazyl/GTSRB",
        model_name="bazyl/gtsrb-model",
        num_samples=50,
        num_classes=43,
        accuracy=0.98,
        precision_macro=0.97,
        recall_macro=0.97,
        f1_macro=0.97,
        precision_weighted=0.98,
        recall_weighted=0.98,
        f1_weighted=0.98,
        average_confidence=0.99,
        average_entropy=0.05,
    )

    data_dict = result.to_dict()
    assert data_dict["accuracy"] == 0.98
    assert data_dict["dataset_name"] == "bazyl/GTSRB"

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "test_result.json"
        result.save_json(json_path)

        assert json_path.is_file()
        assert "accuracy" in json_path.read_text(encoding="utf-8")
