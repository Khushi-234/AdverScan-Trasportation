"""
Tests for AdverScan SQLite database engine, pragmas, transactions, and backups.
"""

import os
from pathlib import Path
import sqlite3
import pytest
from sqlalchemy import text

from app.storage.db import (
    backup_db,
    create_sqlite_engine,
    get_db_path,
    get_db_session,
    init_db,
    reset_db,
)
from app.storage.models import AssessmentRecord


@pytest.fixture
def temp_db_path(tmp_path):
    """Provide an isolated temporary SQLite database path."""
    db_file = tmp_path / "test_adverscan.db"
    return str(db_file)


def test_sqlite_engine_pragmas(temp_db_path):
    """Verify that SQLite connections strictly enforce WAL mode, foreign keys, and busy timeout."""
    engine = create_sqlite_engine(temp_db_path)
    with engine.connect() as conn:
        fk_result = conn.execute(text("PRAGMA foreign_keys;")).scalar()
        assert fk_result == 1, "Foreign keys must be enabled (PRAGMA foreign_keys = 1)."

        timeout_result = conn.execute(text("PRAGMA busy_timeout;")).scalar()
        assert timeout_result == 30000, "Busy timeout must be set to 30,000ms."

        journal_result = conn.execute(text("PRAGMA journal_mode;")).scalar()
        assert journal_result.upper() == "WAL", "Journal mode must be WAL."


def test_init_db_and_reset(temp_db_path):
    """Verify database initialization creates all tables and reset_db clears them cleanly."""
    engine = init_db(temp_db_path)
    assert Path(temp_db_path).is_file()

    with engine.connect() as conn:
        tables = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        ).scalars().all()
        expected = [
            "assessment_reports",
            "assessments",
            "attack_executions",
            "datasets",
            "defense_executions",
            "defense_recommendations",
            "models",
            "vulnerability_findings",
            "xai_results",
        ]
        for tbl in expected:
            assert tbl in tables, f"Expected table '{tbl}' was not created."

    # Test reset_db drops and recreates
    reset_db(temp_db_path)
    with engine.connect() as conn:
        tables_after = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        ).scalars().all()
        for tbl in expected:
            assert tbl in tables_after


def test_session_transaction_and_rollback(temp_db_path):
    """Verify that get_db_session commits on success and rolls back on exception."""
    init_db(temp_db_path)

    # 1. Successful commit
    with get_db_session(temp_db_path) as session:
        scan = AssessmentRecord(scan_id="SCAN-TEST-001", assessment_name="Test Commit", status="pending")
        session.add(scan)

    # Verify record was persisted
    with get_db_session(temp_db_path) as session:
        found = session.query(AssessmentRecord).filter_by(scan_id="SCAN-TEST-001").first()
        assert found is not None
        assert found.assessment_name == "Test Commit"

    # 2. Rollback on exception
    with pytest.raises(RuntimeError):
        with get_db_session(temp_db_path) as session:
            scan_bad = AssessmentRecord(scan_id="SCAN-TEST-BAD", assessment_name="Bad Commit", status="running")
            session.add(scan_bad)
            raise RuntimeError("Forced simulation error")

    # Verify record was rolled back
    with get_db_session(temp_db_path) as session:
        found_bad = session.query(AssessmentRecord).filter_by(scan_id="SCAN-TEST-BAD").first()
        assert found_bad is None


def test_database_backup(temp_db_path, tmp_path):
    """Verify the atomic SQLite backup utility copies data faithfully."""
    init_db(temp_db_path)
    with get_db_session(temp_db_path) as session:
        session.add(AssessmentRecord(scan_id="SCAN-BACKUP-TEST", assessment_name="Before Backup", status="completed"))

    backup_dir = tmp_path / "backups"
    backup_file = backup_db(backup_dir=str(backup_dir), source_db_path=temp_db_path)

    assert Path(backup_file).is_file()
    assert Path(backup_file).stat().st_size > 0

    # Query the backup database directly
    conn = sqlite3.connect(backup_file)
    cursor = conn.cursor()
    cursor.execute("SELECT scan_id, assessment_name FROM assessments WHERE scan_id = 'SCAN-BACKUP-TEST';")
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "SCAN-BACKUP-TEST"
    assert row[1] == "Before Backup"
