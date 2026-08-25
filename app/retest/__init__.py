"""
Module 8 — Re-Test & Comparison for AdverScan.

Provides coordinator engine, comparative analysis, and result DTOs for verifying model hardening.
"""

from app.retest.retest_result import ComparisonResult, RetestResult
from app.retest.comparison import ComparisonEngine, compare_results
from app.retest.retest_engine import RetestEngine

__all__ = [
    "RetestEngine",
    "ComparisonEngine",
    "compare_results",
    "RetestResult",
    "ComparisonResult",
]
