"""
Custom exceptions for the Hardening module (Module 7).
"""


class HardeningError(Exception):
    """Base exception class for all hardening errors."""

    pass


class DefenseNotFoundError(HardeningError):
    """Raised when a requested defense type or strategy is not registered or found."""

    pass


class HardeningConfigurationError(HardeningError):
    """Raised when hardening configuration or parameter specifications are invalid."""

    pass


class DefenseExecutionError(HardeningError):
    """Raised when an error occurs during defense application or model hardening."""

    pass
