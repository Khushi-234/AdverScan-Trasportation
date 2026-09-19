"""
Module 8 — Re-Test & Comparison for AdverScan.

Provides coordinator engine, comparative analysis, result DTOs, and acceptance thresholds for verifying model hardening.
"""

from app.retest.threshold import RetestThresholds, Thresholds

__all__ = [
    "RetestEngine",
    "ComparisonEngine",
    "compare_results",
    "RetestResult",
    "ComparisonResult",
    "RetestThresholds",
    "Thresholds",
]


def __getattr__(name: str):
    if name == "RetestEngine":
        from app.retest.retest_engine import RetestEngine
        return RetestEngine
    elif name == "ComparisonEngine":
        from app.retest.comparison import ComparisonEngine
        return ComparisonEngine
    elif name == "compare_results":
        from app.retest.comparison import compare_results
        return compare_results
    elif name in ("RetestResult", "ComparisonResult"):
        import app.retest.retest_result as _rr
        return getattr(_rr, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
