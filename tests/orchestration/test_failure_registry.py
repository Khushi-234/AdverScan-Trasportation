"""
Unit tests for FailureRegistry and FailureRecord in app.orchestration.failure_registry.
"""

import json
import pytest
from app.orchestration.failure_registry import FailureRecord, FailureRegistry, FailureSeverity


def test_1_failure_record_creation():
    rec = FailureRecord(
        failure_id="FAIL-101",
        timestamp="2026-09-05T12:00:00+00:00",
        module="M1_ingestion",
        operation="ingest_model",
        error_type="ValueError",
        message="Invalid model file",
        severity=FailureSeverity.ERROR,
        recoverable=False,
    )
    assert rec.failure_id == "FAIL-101"
    assert rec.module == "M1_ingestion"
    assert rec.error_type == "ValueError"
    assert rec.recoverable is False


def test_2_unique_failure_ids():
    reg = FailureRegistry()
    r1 = reg.register(module="M1", operation="op1", error_type="E1", message="m1")
    r2 = reg.register(module="M1", operation="op2", error_type="E2", message="m2")
    assert r1.failure_id != r2.failure_id
    assert r1.failure_id.startswith("FAIL-")
    assert r2.failure_id.startswith("FAIL-")


def test_3_timestamp_generation():
    reg = FailureRegistry()
    rec = reg.register(module="M1", operation="op1", error_type="E1", message="m1")
    assert isinstance(rec.timestamp, str)
    assert len(rec.timestamp) > 0


def test_4_exception_type_and_message_capture():
    reg = FailureRegistry()
    try:
        raise KeyError("Missing hyperparameter 'learning_rate'")
    except KeyError as exc:
        rec = reg.register_exception(exc, module="M3_attack_engine", operation="run_attack")
        assert rec.error_type == "KeyError"
        assert "Missing hyperparameter" in rec.message


def test_5_module_and_operation_capture():
    reg = FailureRegistry()
    rec = reg.register(module="M7_hardening", operation="harden_spatial_smoothing", error_type="RuntimeError", message="CUDA OOM")
    assert rec.module == "M7_hardening"
    assert rec.operation == "harden_spatial_smoothing"


def test_6_severity_handling():
    reg = FailureRegistry()
    rec_info = reg.register(module="M1", operation="op", error_type="E", message="msg", severity=FailureSeverity.INFO)
    rec_crit = reg.register(module="M1", operation="op", error_type="E", message="msg", severity=FailureSeverity.CRITICAL)
    assert rec_info.severity == "INFO"
    assert rec_crit.severity == "CRITICAL"


def test_7_recoverability_handling():
    reg = FailureRegistry()
    rec_fatal = reg.register(module="M1", operation="op", error_type="E", message="m", recoverable=False)
    rec_recov = reg.register(module="M3", operation="op", error_type="E", message="m", recoverable=True)
    assert rec_fatal.recoverable is False
    assert rec_recov.recoverable is True


def test_8_batch_index_capture():
    reg = FailureRegistry()
    rec = reg.register(module="M3", operation="op", error_type="E", message="m", batch_index=4)
    assert rec.batch_index == 4


def test_9_sample_index_capture():
    reg = FailureRegistry()
    rec1 = reg.register(module="M3", operation="op", error_type="E", message="m", sample_index=12)
    rec2 = reg.register(module="M3", operation="op", error_type="E", message="m", sample_index=[10, 11, 12])
    assert rec1.sample_index == 12
    assert rec2.sample_index == [10, 11, 12]


def test_10_attack_name_capture():
    reg = FailureRegistry()
    rec = reg.register(module="M3_attack_engine", operation="run_attack", error_type="E", message="m", attack_name="pgd")
    assert rec.attack_name == "pgd"


def test_11_defense_name_capture():
    reg = FailureRegistry()
    rec = reg.register(module="M7_hardening", operation="harden", error_type="E", message="m", defense_name="adversarial_training")
    assert rec.defense_name == "adversarial_training"


def test_12_traceback_capture():
    reg = FailureRegistry()
    try:
        raise ValueError("Invalid dimension size")
    except ValueError as exc:
        rec = reg.register_exception(exc, module="M1", operation="op")
        assert rec.traceback is not None
        assert "ValueError: Invalid dimension size" in rec.traceback


def test_13_optional_fields_remain_none():
    reg = FailureRegistry()
    rec = reg.register(module="M1", operation="op", error_type="E", message="m")
    assert rec.batch_index is None
    assert rec.sample_index is None
    assert rec.attack_name is None
    assert rec.defense_name is None
    assert rec.traceback is None


def test_14_registry_insertion_order():
    reg = FailureRegistry()
    r1 = reg.register(module="M1", operation="op1", error_type="E1", message="m1")
    r2 = reg.register(module="M2", operation="op2", error_type="E2", message="m2")
    r3 = reg.register(module="M3", operation="op3", error_type="E3", message="m3")
    all_recs = reg.get_all()
    assert len(all_recs) == 3
    assert all_recs[0].failure_id == r1.failure_id
    assert all_recs[1].failure_id == r2.failure_id
    assert all_recs[2].failure_id == r3.failure_id


def test_15_get_all():
    reg = FailureRegistry()
    reg.register(module="M1", operation="op1", error_type="E1", message="m1")
    reg.register(module="M2", operation="op2", error_type="E2", message="m2")
    assert len(reg.get_all()) == 2


def test_16_get_by_id():
    reg = FailureRegistry()
    r1 = reg.register(module="M1", operation="op1", error_type="E1", message="m1")
    found = reg.get_by_id(r1.failure_id)
    assert found is not None
    assert found.message == "m1"
    assert reg.get_by_id("NON_EXISTENT_ID") is None


def test_17_count():
    reg = FailureRegistry()
    assert reg.count() == 0
    reg.register(module="M1", operation="op1", error_type="E1", message="m1")
    assert reg.count() == 1


def test_18_clear():
    reg = FailureRegistry()
    reg.register(module="M1", operation="op1", error_type="E1", message="m1")
    assert reg.count() == 1
    reg.clear()
    assert reg.count() == 0


def test_19_to_dict_json_serializable():
    reg = FailureRegistry()
    reg.register(module="M1", operation="op1", error_type="E1", message="m1", batch_index=2, attack_name="fgsm")
    d_list = reg.to_dict()
    assert isinstance(d_list, list)
    assert len(d_list) == 1
    json_str = json.dumps(d_list)
    assert isinstance(json_str, str)


def test_20_register_exception_behavior():
    reg = FailureRegistry()
    try:
        raise ZeroDivisionError("division by zero")
    except ZeroDivisionError as exc:
        rec = reg.register_exception(exc, module="M5", operation="score", severity=FailureSeverity.WARNING)
        assert rec.error_type == "ZeroDivisionError"
        assert rec.severity == "WARNING"


def test_21_registry_failure_isolation():
    reg = FailureRegistry()

    # Monkey-patching internal _records to test failure isolation safety guard
    class BrokenList(list):
        def append(self, item):
            raise RuntimeError("Internal storage full")

    reg._records = BrokenList()
    # register should not crash the caller even if internal append fails
    rec = reg.register(module="M1", operation="op", error_type="E", message="m")
    assert isinstance(rec, FailureRecord)


def test_22_empty_registry_behavior():
    reg = FailureRegistry()
    assert reg.count() == 0
    assert reg.get_all() == []
    assert reg.to_dict() == []
    assert reg.get_by_id("ANY_ID") is None
