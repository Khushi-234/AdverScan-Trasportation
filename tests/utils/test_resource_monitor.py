"""
Unit tests for ResourceMonitor in app.utils.resource_monitor.
"""

import json
import pytest
import torch
from app.utils.resource_monitor import ResourceMonitor, ResourceSnapshot
from app.orchestration.orchestrator import AdverScanOrchestrator
from app.orchestration.pipeline_config import PipelineConfig
from app.orchestration.orchestration_result import OrchestrationResult


def test_1_resource_monitor_instantiation():
    monitor = ResourceMonitor()
    assert monitor is not None
    assert monitor._is_monitoring is False


def test_2_start_monitoring():
    monitor = ResourceMonitor()
    monitor.start()
    assert monitor._is_monitoring is True
    assert monitor._start_time is not None
    assert monitor._initial_snapshot is not None


def test_3_stop_monitoring():
    monitor = ResourceMonitor()
    monitor.start()
    summary = monitor.stop()
    assert monitor._is_monitoring is False
    assert isinstance(summary, dict)
    assert summary.get("monitoring_available") is True
    assert "duration_seconds" in summary
    assert "cpu" in summary
    assert "memory" in summary
    assert "process" in summary
    assert "gpu" in summary


def test_4_snapshot_structured_data():
    monitor = ResourceMonitor()
    snap = monitor.snapshot()
    assert isinstance(snap, ResourceSnapshot)
    d = snap.to_dict()
    assert isinstance(d, dict)
    assert "timestamp" in d
    assert "elapsed_seconds" in d
    assert "cpu_percent" in d
    assert "ram_total_mb" in d
    assert "process_ram_mb" in d
    assert "gpu_available" in d


def test_5_cpu_metrics():
    monitor = ResourceMonitor()
    snap = monitor.snapshot()
    assert isinstance(snap.cpu_percent, (int, float))
    assert snap.cpu_percent >= 0.0


def test_6_ram_metrics():
    monitor = ResourceMonitor()
    snap = monitor.snapshot()
    assert isinstance(snap.ram_total_mb, (int, float))
    assert isinstance(snap.ram_available_mb, (int, float))
    assert isinstance(snap.ram_percent, (int, float))
    assert snap.ram_total_mb >= 0.0


def test_7_process_memory_metrics():
    monitor = ResourceMonitor()
    snap = monitor.snapshot()
    assert isinstance(snap.process_ram_mb, (int, float))
    assert snap.process_ram_mb >= 0.0


def test_8_cpu_only_environments_do_not_fail():
    monitor = ResourceMonitor()
    monitor.start()
    summary = monitor.stop()
    assert summary["gpu"]["gpu_name"] is not None


def test_9_gpu_unavailable_graceful_fallback():
    if not torch.cuda.is_available():
        monitor = ResourceMonitor()
        snap = monitor.snapshot()
        assert snap.gpu_available is False
        assert snap.gpu_info["gpu_name"] == "N/A"
        assert snap.gpu_info["gpu_total_memory_mb"] is None


def test_10_gpu_apis_cuda_safety():
    monitor = ResourceMonitor()
    monitor.start()
    summary = monitor.stop()
    if not torch.cuda.is_available():
        assert summary["gpu"]["gpu_available"] is False
    else:
        assert summary["gpu"]["gpu_available"] is True


def test_11_resource_summary_json_serializable():
    monitor = ResourceMonitor()
    monitor.start()
    summary = monitor.stop()
    json_str = json.dumps(summary)
    assert isinstance(json_str, str)
    reconstructed = json.loads(json_str)
    assert reconstructed["monitoring_available"] is True


def test_12_peak_gpu_memory_handling_cpu_safe():
    monitor = ResourceMonitor()
    monitor.start()
    summary = monitor.stop()
    gpu_info = summary.get("gpu", {})
    assert "peak_pytorch_allocated_bytes" in gpu_info
    assert "peak_pytorch_reserved_bytes" in gpu_info
    if not torch.cuda.is_available():
        assert gpu_info["peak_pytorch_allocated_bytes"] == 0


def test_13_monitoring_failures_do_not_break_experiment():
    monitor = ResourceMonitor()

    def buggy_snapshot():
        raise RuntimeError("Simulated Hardware Sensor Malfunction")

    monitor.snapshot = buggy_snapshot  # Monkey-patch snapshot to simulate failure
    # stop() should gracefully return fallback dictionary instead of crashing
    summary = monitor.stop()
    assert isinstance(summary, dict)
    assert summary.get("monitoring_available") is False
    assert "error" in summary


def test_14_orchestration_result_resource_summary():
    res = OrchestrationResult(
        status="SUCCESS",
        execution_mode="baseline_only",
        resource_summary={"monitoring_available": True, "duration_seconds": 1.5}
    )
    d = res.to_dict()
    assert d["resource_summary"] == {"monitoring_available": True, "duration_seconds": 1.5}
