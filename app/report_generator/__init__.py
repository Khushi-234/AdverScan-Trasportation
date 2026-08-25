"""
Module 9 — Report Generator package for AdverScan.

Exposes the public reporting interface and DTOs:
- ReportGenerator: Coordinator for generating security reports.
- ReportData: Input data contract aggregating M1-M8 outputs.
- ReportResult: DTO for generated reports, JSON/text outputs, and recommendations.
"""

from .report_data import ReportData
from .report_generator import ReportGenerator
from .report_result import ReportResult

__all__ = [
    "ReportGenerator",
    "ReportData",
    "ReportResult",
]
