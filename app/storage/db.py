"""
Database engine and connection management for AdverScan.

Provides SQLite configuration with:
- WAL (Write-Ahead Logging) journal mode for multi-threaded read/write concurrency
- Busy timeout of 30,000ms to eliminate 'database is locked' errors
- Explicit foreign key constraint enforcement
- Safe scoped sessions with context manager support
- Atomic SQLite backup utilities
"""

from __future__ import annotations

import contextlib
from datetime import datetime
import os
from pathlib import Path
import sqlite3
from typing import Generator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Base directory for storage and SQLite database
DEFAULT_STORAGE_DIR = Path(__file__).resolve().parent.parent.parent / "storage_data"
DEFAULT_DB_FILE = DEFAULT_STORAGE_DIR / "adverscan.db"

# Declarative Base for all AdverScan storage models
Base = declarative_base()

_ENGINE: Optional[Engine] = None
_SESSION_FACTORY: Optional[sessionmaker] = None
_CURRENT_DB_PATH: Optional[str] = None


def get_db_path(custom_path: Optional[str] = None) -> str:
    """Resolve the absolute database file path."""
    if custom_path:
        return str(Path(custom_path).resolve())
    env_path = os.environ.get("ADVERSCAN_DB_PATH")
    if env_path:
        return str(Path(env_path).resolve())
    return str(DEFAULT_DB_FILE.resolve())


def create_sqlite_engine(db_path: Optional[str] = None, echo: bool = False) -> Engine:
    """
    Create a SQLAlchemy engine configured for SQLite with WAL mode,
    extended busy timeout, and foreign key enforcement.
    """
    target_path = get_db_path(db_path)
    is_memory = target_path == ":memory:" or "mode=memory" in target_path

    if not is_memory:
        p = Path(target_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        sqlite_url = f"sqlite:///{target_path}"
    else:
        sqlite_url = "sqlite:///:memory:"

    engine = create_engine(
        sqlite_url,
        echo=echo,
        connect_args={
            "check_same_thread": False,
            "timeout": 30.0,
        },
        future=True,
    )

    @event.listens_for(engine, "connect")
    def _configure_sqlite_connection(dbapi_connection, connection_record):
        """Enforce strict foreign keys, WAL mode, and reasonable timeout."""
        if isinstance(dbapi_connection, sqlite3.Connection):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys = ON;")
            cursor.execute("PRAGMA busy_timeout = 30000;")
            if not is_memory:
                try:
                    cursor.execute("PRAGMA journal_mode = WAL;")
                    cursor.execute("PRAGMA synchronous = NORMAL;")
                except Exception:
                    pass
            cursor.close()

    return engine


def get_engine(db_path: Optional[str] = None) -> Engine:
    """Retrieve or create the singleton engine."""
    global _ENGINE, _CURRENT_DB_PATH, _SESSION_FACTORY
    target_path = get_db_path(db_path)
    if _ENGINE is None or _CURRENT_DB_PATH != target_path:
        _ENGINE = create_sqlite_engine(target_path)
        _CURRENT_DB_PATH = target_path
        _SESSION_FACTORY = sessionmaker(autocommit=False, autoflush=False, bind=_ENGINE)
    return _ENGINE


def get_session_factory(db_path: Optional[str] = None) -> sessionmaker:
    """Retrieve or create the session factory bound to the active engine."""
    global _SESSION_FACTORY
    get_engine(db_path)
    assert _SESSION_FACTORY is not None
    return _SESSION_FACTORY


@contextlib.contextmanager
def get_db_session(db_path: Optional[str] = None) -> Generator[Session, None, None]:
    """
    Context manager for database sessions with automatic commit and rollback.

    Usage:
        with get_db_session() as session:
            session.add(record)
    """
    session_factory = get_session_factory(db_path)
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(db_path: Optional[str] = None) -> Engine:
    """
    Initialize SQLite database schema and tables.
    Safe to call multiple times (creates tables only if missing).
    """
    # Import models here to ensure they are registered with Base.metadata
    from app.storage import models  # noqa: F401

    engine = get_engine(db_path)
    Base.metadata.create_all(bind=engine)
    return engine


def reset_db(db_path: Optional[str] = None) -> Engine:
    """Drop and recreate all tables in the specified database (primarily for testing)."""
    from app.storage import models  # noqa: F401

    engine = get_engine(db_path)
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    return engine


def backup_db(
    backup_dir: Optional[str] = None,
    source_db_path: Optional[str] = None,
) -> str:
    """
    Perform an online, atomic zero-downtime backup of the SQLite database.

    Args:
        backup_dir: Directory where the backup file will be created.
        source_db_path: Path to the source database.

    Returns:
        Absolute path to the created backup file.
    """
    src_path = get_db_path(source_db_path)
    if src_path == ":memory:":
        raise ValueError("Cannot backup an in-memory database to disk via file path.")

    if not Path(src_path).is_file():
        raise FileNotFoundError(f"Source database file '{src_path}' does not exist.")

    out_dir = Path(backup_dir) if backup_dir else Path(src_path).parent / "backups"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"adverscan_backup_{timestamp}.db"
    dest_path = out_dir / backup_filename

    # Perform atomic backup using SQLite backup API
    src_conn = sqlite3.connect(src_path)
    dest_conn = sqlite3.connect(str(dest_path))
    try:
        with dest_conn:
            src_conn.backup(dest_conn, pages=250, sleep=0.01)
    finally:
        dest_conn.close()
        src_conn.close()

    return str(dest_path.resolve())
