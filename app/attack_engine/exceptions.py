"""
Custom exceptions for the Adversarial Attack Engine module.
"""


class AttackError(Exception):
    """Base class for all attack engine exceptions."""

    pass


class AttackConfigurationError(AttackError):
    """Raised when attack configuration parameters are invalid or missing."""

    pass


class AttackExecutionError(AttackError):
    """Raised when an error occurs during attack execution."""

    pass


class UnsupportedModelError(AttackError):
    """Raised when a model is not supported by the requested attack engine implementation."""

    pass
