"""
Structured Failure Registry module for AdverScan (Module 8 / Orchestration).
Provides structured, machine-readable recording of pipeline execution failures,
exception types, tracebacks, and contextual metadata.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import traceback
from typing import Any, Dict, List, Optional, Union
import uuid


class FailureSeverity:
    """Standard severity levels for registered failures."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class FailureRecord:
    """
    Structured container representing a single registered failure event.

    Attributes:
        failure_id: Unique UUID4 record identifier (non-deterministic record key,
                   excluded from config/reproducibility hashing).
        timestamp: Timezone-aware ISO 8601 string timestamp recording when failure occurred.
        module: Name of the pipeline module/component where failure occurred.
        operation: Specific function/operation name executing when failure occurred.
        error_type: Exception class name or string description.
        message: Human-readable error description.
        severity: Severity classification (INFO, WARNING, ERROR, CRITICAL).
        recoverable: True if caller execution continues after failure, False if failure
                     terminates operation/run. Recorded directly from caller, not inferred.
        batch_index: Optional batch index if failure occurred during batch processing.
        sample_index: Optional sample index or list of indices for targeted failures.
        attack_name: Optional name of the attack if failure occurred during attack execution.
        defense_name: Optional name of defense if failure occurred during hardening.
        traceback: Optional formatted string traceback for debugging.
        metadata: Optional dictionary of additional contextual attributes.
    """
    failure_id: str
    timestamp: str
    module: str
    operation: str
    error_type: str
    message: str
    severity: str = FailureSeverity.ERROR
    recoverable: bool = True
    batch_index: Optional[int] = None
    sample_index: Optional[Union[int, List[int]]] = None
    attack_name: Optional[str] = None
    defense_name: Optional[str] = None
    traceback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert failure record to dictionary representation."""
        return asdict(self)


class FailureRegistry:
    """
    Registry for recording and inspecting structured failure events during pipeline execution.
    Maintains failure records in insertion order with failure-safe exception guards.
    """

    def __init__(self) -> None:
        self._records: List[FailureRecord] = []

    def register(
        self,
        module: str,
        operation: str,
        error_type: str,
        message: str,
        severity: str = FailureSeverity.ERROR,
        recoverable: bool = True,
        batch_index: Optional[int] = None,
        sample_index: Optional[Union[int, List[int]]] = None,
        attack_name: Optional[str] = None,
        defense_name: Optional[str] = None,
        traceback: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FailureRecord:
        """
        Register a new structured failure event.

        Returns:
            Instantiated FailureRecord object.
        """
        try:
            record_id = f"FAIL-{uuid.uuid4()}"
            ts = datetime.now().astimezone().isoformat()

            record = FailureRecord(
                failure_id=record_id,
                timestamp=ts,
                module=str(module),
                operation=str(operation),
                error_type=str(error_type),
                message=str(message),
                severity=str(severity),
                recoverable=bool(recoverable),
                batch_index=batch_index,
                sample_index=sample_index,
                attack_name=attack_name,
                defense_name=defense_name,
                traceback=traceback,
                metadata=metadata or {},
            )
            self._records.append(record)
            return record
        except Exception:
            # Failure-safe: failure registration error must never crash pipeline
            fallback_record = FailureRecord(
                failure_id=f"FAIL-{uuid.uuid4()}",
                timestamp=datetime.now().astimezone().isoformat(),
                module=str(module),
                operation=str(operation),
                error_type=str(error_type),
                message=str(message),
                severity=severity,
                recoverable=recoverable,
            )
            return fallback_record

    def register_exception(
        self,
        exc: Exception,
        module: str,
        operation: str,
        severity: str = FailureSeverity.ERROR,
        recoverable: bool = True,
        batch_index: Optional[int] = None,
        sample_index: Optional[Union[int, List[int]]] = None,
        attack_name: Optional[str] = None,
        defense_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FailureRecord:
        """
        Extract exception type, message, and traceback automatically from an Exception instance.

        Returns:
            Instantiated FailureRecord object.
        """
        try:
            err_type = type(exc).__name__
            msg = str(exc)
            tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        except Exception:
            err_type = "Exception"
            msg = str(exc)
            tb_str = None

        return self.register(
            module=module,
            operation=operation,
            error_type=err_type,
            message=msg,
            severity=severity,
            recoverable=recoverable,
            batch_index=batch_index,
            sample_index=sample_index,
            attack_name=attack_name,
            defense_name=defense_name,
            traceback=tb_str,
            metadata=metadata,
        )

    def get_all(self) -> List[FailureRecord]:
        """Get all registered failure records in insertion order."""
        return list(self._records)

    def get_by_id(self, failure_id: str) -> Optional[FailureRecord]:
        """Find a failure record by its unique failure_id."""
        for r in self._records:
            if r.failure_id == failure_id:
                return r
        return None

    def count(self) -> int:
        """Get total count of registered failures."""
        return len(self._records)

    def clear(self) -> None:
        """Clear all registered failure records."""
        self._records.clear()

    def to_dict(self) -> List[Dict[str, Any]]:
        """Serialize all failure records to a list of dictionaries."""
        return [r.to_dict() for r in self._records]
