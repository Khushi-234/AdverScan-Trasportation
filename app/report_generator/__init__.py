"""
Module 9 — Report Generator package for AdverScan.

Public API:
    ReportData          — Structured input data contract (M1–M8 + execution).
    ReportGenerator     — 15-section report coordinator.
    ReportResult        — Generated report DTO with serialization helpers.
    ReportWriter        — Persists reports to reports/<scan_id>/ on disk.
    ExecutionSummary    — Per-module timing and status container (section 13).
    ModuleExecutionRecord — Single module execution record.
"""

from .execution_summary import ExecutionSummary, ModuleExecutionRecord
from .report_data import ReportData
from .report_generator import ReportGenerator
from .report_result import ReportResult
from .report_writer import ReportWriter

__all__ = [
    "ExecutionSummary",
    "ModuleExecutionRecord",
    "ReportData",
    "ReportGenerator",
    "ReportResult",
    "ReportWriter",
]
