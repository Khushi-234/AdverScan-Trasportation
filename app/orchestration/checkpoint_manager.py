"""
Evaluation Checkpoint and Resume State Manager for Module 8 (Orchestration).

Provides a reliable, file-safe, atomic checkpoint/resume mechanism for persisting
evaluation progress during multi-batch assessment runs.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import uuid
from typing import Any, Dict, List, Optional, Union


# ============================================================================
# Exception Hierarchy
# ============================================================================

class CheckpointError(Exception):
    """Base exception class for checkpoint operation failures."""
    pass


class CheckpointCorruptedError(CheckpointError):
    """Raised when a checkpoint file is missing required fields, empty, or contains malformed JSON."""
    pass


class CheckpointIncompatibleError(CheckpointError):
    """Raised when loading a checkpoint with an incompatible configuration hash or dataset parameters."""
    pass


# ============================================================================
# Configuration Hash Helper
# ============================================================================

def compute_configuration_hash(
    dataset_name: str,
    total_samples: int,
    batch_size: int,
    attacks: List[str],
    extra_params: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a stable, deterministic SHA256 configuration hash.
    
    Args:
        dataset_name: Dataset identifier.
        total_samples: Target sample count.
        batch_size: Batch size used for evaluation.
        attacks: List of attack names.
        extra_params: Optional extra configuration parameters.

    Returns:
        32-character hexadecimal digest string.
    """
    normalized_attacks = sorted([str(a).lower().strip() for a in attacks])
    normalized_config = {
        "dataset_name": str(dataset_name).strip().lower(),
        "total_samples": int(total_samples),
        "batch_size": int(batch_size),
        "attacks": normalized_attacks,
    }
    if extra_params:
        normalized_config["extra_params"] = extra_params

    raw_json = json.dumps(normalized_config, sort_keys=True)
    return hashlib.sha256(raw_json.encode("utf-8")).hexdigest()[:32]


# ============================================================================
# Evaluation Checkpoint State
# ============================================================================

@dataclass
class EvaluationCheckpointState:
    """
    Data container representing persisted evaluation progress.
    """
    run_id: str
    dataset_name: str
    total_samples: int
    batch_size: int
    attacks: List[str]
    configuration_hash: str
    completed_batch_indices: List[int] = field(default_factory=list)
    processed_samples_count: int = 0
    last_completed_batch_index: Optional[int] = None
    status: str = "in_progress"  # "in_progress", "completed"
    schema_version: int = 1
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    updated_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    accumulated_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationCheckpointState":
        """Reconstruct EvaluationCheckpointState instance from dictionary."""
        known_fields = {
            "run_id",
            "dataset_name",
            "total_samples",
            "batch_size",
            "attacks",
            "configuration_hash",
            "completed_batch_indices",
            "processed_samples_count",
            "last_completed_batch_index",
            "status",
            "schema_version",
            "created_at",
            "updated_at",
            "accumulated_results",
            "metadata",
        }
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)


# ============================================================================
# Evaluation Checkpoint Manager
# ============================================================================

class EvaluationCheckpointManager:
    """
    Manager class responsible for creating, loading, saving, and managing checkpoint states.
    Uses atomic writes (temporary file + os.replace) to guarantee checkpoint file integrity.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = "results/checkpoints",
        filename_prefix: str = "eval_checkpoint",
    ) -> None:
        """
        Initialize EvaluationCheckpointManager.

        Args:
            checkpoint_dir: Path to directory storing checkpoint files.
            filename_prefix: Default filename prefix for checkpoint files.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.filename_prefix = filename_prefix

    def get_checkpoint_path(self, run_id: str) -> Path:
        """Get standard checkpoint filepath for a given run_id."""
        clean_id = str(run_id).replace("/", "_").replace("\\", "_")
        return self.checkpoint_dir / f"{self.filename_prefix}_{clean_id}.json"

    def create_checkpoint(
        self,
        dataset_name: str,
        total_samples: int,
        batch_size: int,
        attacks: List[str],
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> EvaluationCheckpointState:
        """
        Initialize a new EvaluationCheckpointState with a stable configuration hash.

        Args:
            dataset_name: Dataset identifier.
            total_samples: Requested sample evaluation count.
            batch_size: Evaluation batch size.
            attacks: List of attack names.
            run_id: Optional unique run ID. If None, a UUID is generated.
            metadata: Optional user metadata dict.
            extra_config: Optional extra configuration key-values included in config hash.

        Returns:
            Newly instantiated EvaluationCheckpointState.
        """
        final_run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
        config_hash = compute_configuration_hash(
            dataset_name=dataset_name,
            total_samples=total_samples,
            batch_size=batch_size,
            attacks=attacks,
            extra_params=extra_config,
        )

        return EvaluationCheckpointState(
            run_id=final_run_id,
            dataset_name=dataset_name,
            total_samples=total_samples,
            batch_size=batch_size,
            attacks=list(attacks),
            configuration_hash=config_hash,
            metadata=metadata or {},
        )

    def save(
        self,
        state: EvaluationCheckpointState,
        custom_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Persist state to JSON file using atomic file replace strategy.

        Args:
            state: EvaluationCheckpointState instance.
            custom_path: Optional custom destination file path.

        Returns:
            Path object pointing to saved checkpoint file.

        Raises:
            CheckpointError: If atomic write operation fails.
        """
        target_path = Path(custom_path) if custom_path else self.get_checkpoint_path(state.run_id)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        state.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tmp_filename = f"{target_path.name}.tmp_{uuid.uuid4().hex[:6]}"
        tmp_path = target_path.parent / tmp_filename

        try:
            dict_data = state.to_dict()
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(dict_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_path, target_path)
            return target_path

        except Exception as e:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise CheckpointError(f"Failed to save checkpoint file atomically to '{target_path}': {e}") from e

    def load(
        self,
        checkpoint_filepath: Union[str, Path],
        expected_config: Optional[Dict[str, Any]] = None,
    ) -> EvaluationCheckpointState:
        """
        Load, deserialize, and validate an existing checkpoint file.

        Args:
            checkpoint_filepath: Path to checkpoint JSON file.
            expected_config: Optional configuration dict containing keys 'dataset_name',
                            'total_samples', 'batch_size', 'attacks' to check compatibility.

        Returns:
            Deserialized EvaluationCheckpointState.

        Raises:
            CheckpointCorruptedError: If file is missing, empty, malformed, or missing required keys.
            CheckpointIncompatibleError: If expected_config hash does not match state hash.
        """
        path = Path(checkpoint_filepath)
        if not path.exists():
            raise CheckpointCorruptedError(f"Checkpoint file does not exist: '{path}'")

        if path.stat().st_size == 0:
            raise CheckpointCorruptedError(f"Checkpoint file is empty: '{path}'")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise CheckpointCorruptedError(f"Checkpoint file contains malformed JSON: {e}") from e
        except Exception as e:
            raise CheckpointCorruptedError(f"Failed to read checkpoint file: {e}") from e

        if not isinstance(data, dict):
            raise CheckpointCorruptedError("Checkpoint file root must be a JSON object.")

        required_keys = [
            "run_id",
            "dataset_name",
            "total_samples",
            "batch_size",
            "attacks",
            "completed_batch_indices",
            "schema_version",
        ]
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            raise CheckpointCorruptedError(f"Checkpoint missing required schema keys: {missing_keys}")

        schema_ver = data.get("schema_version", 1)
        if schema_ver > 1:
            raise CheckpointCorruptedError(
                f"Unsupported schema version {schema_ver}. Maximum supported version is 1."
            )

        try:
            state = EvaluationCheckpointState.from_dict(data)
        except Exception as e:
            raise CheckpointCorruptedError(f"Failed to deserialize checkpoint state: {e}") from e

        # Validate configuration compatibility if expected_config is provided
        if expected_config is not None:
            expected_hash = compute_configuration_hash(
                dataset_name=expected_config.get("dataset_name", state.dataset_name),
                total_samples=expected_config.get("total_samples", state.total_samples),
                batch_size=expected_config.get("batch_size", state.batch_size),
                attacks=expected_config.get("attacks", state.attacks),
                extra_params=expected_config.get("extra_config"),
            )
            if state.configuration_hash and state.configuration_hash != expected_hash:
                raise CheckpointIncompatibleError(
                    f"Checkpoint configuration hash mismatch! File hash='{state.configuration_hash}', "
                    f"Expected hash='{expected_hash}'."
                )

        return state

    def mark_batch_completed(
        self,
        state: EvaluationCheckpointState,
        batch_index: int,
        samples_processed: int,
        batch_results: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Idempotently mark a batch as completed and update sample counts.

        Args:
            state: Active EvaluationCheckpointState instance.
            batch_index: Index of completed batch.
            samples_processed: Number of samples contained in this batch.
            batch_results: Optional intermediate result dictionary to persist.

        Returns:
            True if batch was newly added, False if it was already marked completed.
        """
        if batch_index in state.completed_batch_indices:
            return False

        state.completed_batch_indices.append(batch_index)
        state.processed_samples_count += max(0, samples_processed)

        if state.last_completed_batch_index is None or batch_index > state.last_completed_batch_index:
            state.last_completed_batch_index = batch_index

        if batch_results is not None:
            state.accumulated_results[str(batch_index)] = batch_results

        return True

    def get_next_incomplete_batch_index(
        self,
        state: EvaluationCheckpointState,
        total_batches: int,
    ) -> Optional[int]:
        """
        Find the smallest uncompleted batch index in range [0, total_batches).

        Args:
            state: EvaluationCheckpointState instance.
            total_batches: Total number of batches in evaluation.

        Returns:
            Next batch index to process, or None if all batches are completed.
        """
        if state.status == "completed":
            return None

        completed_set = set(state.completed_batch_indices)
        for idx in range(total_batches):
            if idx not in completed_set:
                return idx
        return None

    def finalize(
        self,
        state: EvaluationCheckpointState,
        status: str = "completed",
    ) -> Path:
        """
        Mark checkpoint as completed/finalized and save state.

        Args:
            state: EvaluationCheckpointState instance.
            status: Final status string (default "completed").

        Returns:
            Path to saved final checkpoint file.
        """
        state.status = status
        return self.save(state)

    def cleanup(self, checkpoint_filepath: Union[str, Path]) -> bool:
        """
        Safely delete checkpoint file from disk.

        Args:
            checkpoint_filepath: Path to checkpoint file.

        Returns:
            True if file was deleted, False if file did not exist.
        """
        path = Path(checkpoint_filepath)
        if path.exists() and path.is_file():
            try:
                path.unlink()
                return True
            except OSError as e:
                raise CheckpointError(f"Failed to delete checkpoint file '{path}': {e}") from e
        return False
