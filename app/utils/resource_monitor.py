"""
Resource Monitoring module for AdverScan (Module 10 / Utilities).
Provides lightweight, failure-safe monitoring for CPU, system RAM, process RAM/CPU,
and PyTorch CUDA GPU memory utilization.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import os
import platform
import resource
import time
from typing import Any, Dict, List, Optional, Tuple
import torch

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    psutil = None
    _HAS_PSUTIL = False


@dataclass
class ResourceSnapshot:
    """
    Structured snapshot container representing system and hardware resource metrics at a specific instant.
    """
    timestamp: str
    elapsed_seconds: float
    cpu_percent: float
    ram_total_mb: float
    ram_available_mb: float
    ram_percent: float
    process_ram_mb: float
    process_cpu_percent: Optional[float]
    gpu_available: bool
    gpu_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary representation."""
        return asdict(self)


class ResourceMonitor:
    """
    Lightweight, research-grade resource monitor for tracking computational resource usage,
    peak GPU memory, and system overhead during AdverScan execution.
    """

    def __init__(self) -> None:
        self._start_time: Optional[float] = None
        self._start_timestamp: Optional[str] = None
        self._stop_time: Optional[float] = None
        self._stop_timestamp: Optional[str] = None
        self._is_monitoring: bool = False
        self._initial_snapshot: Optional[ResourceSnapshot] = None
        self._final_snapshot: Optional[ResourceSnapshot] = None
        self._stage_snapshots: Dict[str, Dict[str, Any]] = {}
        self._peak_process_ram_mb: float = 0.0

    def start(self) -> None:
        """
        Start monitoring session. Resets PyTorch CUDA peak memory statistics if CUDA is available.
        """
        try:
            self._start_time = time.time()
            self._start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._is_monitoring = True

            # Reset PyTorch CUDA peak memory stats when CUDA is available
            if torch.cuda.is_available():
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass

            self._initial_snapshot = self.snapshot()
            if self._initial_snapshot:
                self._peak_process_ram_mb = self._initial_snapshot.process_ram_mb
        except Exception:
            # Failure-safe: monitoring initialization should never break caller
            self._is_monitoring = False

    def snapshot(self) -> ResourceSnapshot:
        """
        Capture an instantaneous snapshot of system CPU, RAM, process, and GPU memory metrics.

        Returns:
            ResourceSnapshot dataclass instance.
        """
        now = time.time()
        elapsed = round(now - self._start_time, 4) if self._start_time else 0.0
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # System & Process Metrics Collection
        cpu_pct = 0.0
        ram_total_mb = 0.0
        ram_avail_mb = 0.0
        ram_pct = 0.0
        proc_ram_mb = 0.0
        proc_cpu_pct: Optional[float] = None

        try:
            if _HAS_PSUTIL and psutil is not None:
                cpu_pct = float(psutil.cpu_percent(interval=None))
                mem = psutil.virtual_memory()
                ram_total_mb = round(mem.total / (1024 * 1024), 2)
                ram_avail_mb = round(mem.available / (1024 * 1024), 2)
                ram_pct = float(mem.percent)

                p = psutil.Process(os.getpid())
                proc_ram_mb = round(p.memory_info().rss / (1024 * 1024), 2)
                try:
                    proc_cpu_pct = float(p.cpu_percent(interval=None))
                except Exception:
                    proc_cpu_pct = None
            else:
                # Fallback to standard library / Linux /proc
                proc_ram_mb = self._get_fallback_process_ram_mb()
                ram_total_mb, ram_avail_mb, ram_pct = self._get_fallback_system_ram_mb()
                cpu_pct = 0.0
        except Exception:
            pass

        if proc_ram_mb > self._peak_process_ram_mb:
            self._peak_process_ram_mb = proc_ram_mb

        # GPU Metrics Collection (PyTorch CUDA APIs)
        gpu_avail = False
        gpu_info: Dict[str, Any] = {
            "gpu_available": False,
            "gpu_name": "N/A",
            "gpu_count": 0,
            "gpu_utilization_percent": None,
            "gpu_total_memory_mb": None,
            "gpu_used_memory_mb": None,
            "gpu_free_memory_mb": None,
            "pytorch_allocated_bytes": 0,
            "pytorch_allocated_mb": 0.0,
            "pytorch_reserved_bytes": 0,
            "pytorch_reserved_mb": 0.0,
            "peak_pytorch_allocated_bytes": 0,
            "peak_pytorch_allocated_mb": 0.0,
            "peak_pytorch_reserved_bytes": 0,
            "peak_pytorch_reserved_mb": 0.0,
        }

        try:
            if torch.cuda.is_available():
                gpu_avail = True
                gpu_info["gpu_available"] = True
                gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
                gpu_info["gpu_count"] = torch.cuda.device_count()

                alloc_bytes = torch.cuda.memory_allocated()
                reserv_bytes = torch.cuda.memory_reserved()
                max_alloc_bytes = torch.cuda.max_memory_allocated()
                max_reserv_bytes = torch.cuda.max_memory_reserved()

                gpu_info["pytorch_allocated_bytes"] = alloc_bytes
                gpu_info["pytorch_allocated_mb"] = round(alloc_bytes / (1024 * 1024), 2)
                gpu_info["pytorch_reserved_bytes"] = reserv_bytes
                gpu_info["pytorch_reserved_mb"] = round(reserv_bytes / (1024 * 1024), 2)
                gpu_info["peak_pytorch_allocated_bytes"] = max_alloc_bytes
                gpu_info["peak_pytorch_allocated_mb"] = round(max_alloc_bytes / (1024 * 1024), 2)
                gpu_info["peak_pytorch_reserved_bytes"] = max_reserv_bytes
                gpu_info["peak_pytorch_reserved_mb"] = round(max_reserv_bytes / (1024 * 1024), 2)

                try:
                    total_bytes = torch.cuda.get_device_properties(0).total_memory
                    total_mb = round(total_bytes / (1024 * 1024), 2)
                    used_mb = round(alloc_bytes / (1024 * 1024), 2)
                    free_mb = round((total_bytes - alloc_bytes) / (1024 * 1024), 2)

                    gpu_info["gpu_total_memory_mb"] = total_mb
                    gpu_info["gpu_used_memory_mb"] = used_mb
                    gpu_info["gpu_free_memory_mb"] = free_mb
                except Exception:
                    pass
        except Exception:
            gpu_avail = False

        return ResourceSnapshot(
            timestamp=timestamp_str,
            elapsed_seconds=elapsed,
            cpu_percent=cpu_pct,
            ram_total_mb=ram_total_mb,
            ram_available_mb=ram_avail_mb,
            ram_percent=ram_pct,
            process_ram_mb=proc_ram_mb,
            process_cpu_percent=proc_cpu_pct,
            gpu_available=gpu_avail,
            gpu_info=gpu_info,
        )

    def record_stage(self, stage_name: str) -> None:
        """
        Record a snapshot for a specific pipeline stage (e.g., 'baseline', 'attack', 'hardening').

        Args:
            stage_name: Name identifier for the execution stage.
        """
        try:
            snap = self.snapshot()
            self._stage_snapshots[stage_name] = snap.to_dict()
        except Exception:
            pass

    def stop(self) -> Dict[str, Any]:
        """
        Stop monitoring session and build a structured resource summary payload.

        Returns:
            Dict containing experiment-level resource usage summary.
        """
        try:
            self._stop_time = time.time()
            self._stop_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._is_monitoring = False
            self._final_snapshot = self.snapshot()

            duration = round(self._stop_time - (self._start_time or self._stop_time), 4)
            initial_dict = self._initial_snapshot.to_dict() if self._initial_snapshot else {}
            final_dict = self._final_snapshot.to_dict() if self._final_snapshot else {}

            gpu_avail = self._final_snapshot.gpu_available if self._final_snapshot else False
            gpu_info = self._final_snapshot.gpu_info if self._final_snapshot else {}

            summary = {
                "monitoring_available": True,
                "start_timestamp": self._start_timestamp,
                "stop_timestamp": self._stop_timestamp,
                "duration_seconds": duration,
                "cpu": {
                    "cpu_percent": final_dict.get("cpu_percent", 0.0),
                    "cpu_count": os.cpu_count() or 1,
                    "architecture": platform.machine(),
                },
                "memory": {
                    "ram_total_mb": final_dict.get("ram_total_mb", 0.0),
                    "ram_available_mb": final_dict.get("ram_available_mb", 0.0),
                    "ram_percent": final_dict.get("ram_percent", 0.0),
                    "peak_process_ram_mb": self._peak_process_ram_mb,
                },
                "process": {
                    "process_id": os.getpid(),
                    "process_ram_mb": final_dict.get("process_ram_mb", 0.0),
                    "process_cpu_percent": final_dict.get("process_cpu_percent"),
                },
                "gpu": gpu_info,
                "initial_snapshot": initial_dict,
                "final_snapshot": final_dict,
                "stage_snapshots": self._stage_snapshots,
            }
            return summary
        except Exception as e:
            # Fallback safe summary payload on error
            return {
                "monitoring_available": False,
                "error": str(e),
                "duration_seconds": 0.0,
                "cpu": {},
                "memory": {},
                "process": {},
                "gpu": {"gpu_available": False, "gpu_name": "N/A"},
            }

    @staticmethod
    def _get_fallback_process_ram_mb() -> float:
        """Fallback to get process RAM in MB using standard library resource module."""
        try:
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            # ru_maxrss is in KB on Linux
            return round(rusage.ru_maxrss / 1024.0, 2)
        except Exception:
            return 0.0

    @staticmethod
    def _get_fallback_system_ram_mb() -> Tuple[float, float, float]:
        """Fallback to parse system RAM from Linux /proc/meminfo if available."""
        try:
            mem_total_kb = 0
            mem_avail_kb = 0
            if os.path.exists("/proc/meminfo"):
                with open("/proc/meminfo", "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.split(":")
                        if len(parts) == 2:
                            key = parts[0].strip()
                            val_str = parts[1].strip().split()[0]
                            if key == "MemTotal":
                                mem_total_kb = int(val_str)
                            elif key in ("MemAvailable", "MemFree"):
                                if mem_avail_kb == 0:
                                    mem_avail_kb = int(val_str)

            if mem_total_kb > 0:
                tot_mb = round(mem_total_kb / 1024.0, 2)
                avail_mb = round(mem_avail_kb / 1024.0, 2)
                pct = round(((mem_total_kb - mem_avail_kb) / mem_total_kb) * 100.0, 2)
                return tot_mb, avail_mb, pct
        except Exception:
            pass
        return 0.0, 0.0, 0.0
