"""
AdverScan SQLite Storage Module.

Provides normalized database persistence, schema definitions, and artifact offloading.
"""

from app.storage.db import (
    Base,
    backup_db,
    create_sqlite_engine,
    get_db_path,
    get_db_session,
    get_engine,
    init_db,
    reset_db,
)
from app.storage.models import (
    AssessmentRecord,
    AttackExecutionRecord,
    DatasetRecord,
    DefenseExecutionRecord,
    DefenseRecommendationRecord,
    ModelRecord,
    ReportRecord,
    VulnerabilityFindingRecord,
    XAIResultRecord,
)
from app.storage.storage import (
    ArtifactManager,
    StorageRepository,
)

__all__ = [
    # Database Management
    "Base",
    "get_engine",
    "get_db_session",
    "init_db",
    "reset_db",
    "backup_db",
    "get_db_path",
    "create_sqlite_engine",
    # Models
    "AssessmentRecord",
    "ModelRecord",
    "DatasetRecord",
    "AttackExecutionRecord",
    "DefenseExecutionRecord",
    "XAIResultRecord",
    "VulnerabilityFindingRecord",
    "DefenseRecommendationRecord",
    "ReportRecord",
    # Repository & Artifacts
    "StorageRepository",
    "ArtifactManager",
]
