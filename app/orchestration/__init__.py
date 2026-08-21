"""
Module 8 — Orchestration / Integration package for AdverScan.
"""

from app.orchestration.dataset_adapter import InMemoryDatasetLoader
from app.orchestration.orchestrator import AdverScanOrchestrator
from app.orchestration.orchestration_result import OrchestrationResult
from app.orchestration.pipeline_config import PipelineConfig

__all__ = [
    "InMemoryDatasetLoader",
    "AdverScanOrchestrator",
    "OrchestrationResult",
    "PipelineConfig",
]
