"""
In-memory SQLite connection manager for QueryGym.
A single shared connection is created at startup and recycled on reset().
"""

import os
import sqlite3
from pathlib import Path

from db.seed import seed
from db.bugs import inject_bugs

_DB_PATH = Path(__file__).parent / "schema.sql"

# Module-level connection — shared across all requests (single-session assumption)
_conn: sqlite3.Connection | None = None


def init_db() -> None:
    """Create schema, seed data, inject bugs. Called at FastAPI startup."""
    global _conn
    _conn = sqlite3.connect(":memory:", check_same_thread=False)
    _conn.row_factory = sqlite3.Row
    _apply_schema(_conn)
    seed(_conn)
    inject_bugs(_conn)


def reset_db() -> None:
    """
    Wipe all tables and re-initialise from scratch.
    Called by POST /reset to guarantee a clean deterministic episode.
    """
    global _conn
    if _conn is None:
        init_db()
        return
    cur = _conn.cursor()
    # Drop in reverse FK order
    cur.executescript(
        """
        PRAGMA foreign_keys = OFF;
        DROP TABLE IF EXISTS events;
        DROP TABLE IF EXISTS invoices;
        DROP TABLE IF EXISTS subscriptions;
        DROP TABLE IF EXISTS customers;
        DROP TABLE IF EXISTS plans;
        """
    )
    _conn.commit()
    _apply_schema(_conn)
    seed(_conn)
    inject_bugs(_conn)


def get_db() -> sqlite3.Connection:
    """Return the shared in-memory connection."""
    if _conn is None:
        raise RuntimeError("Database not initialised — call init_db() first")
    return _conn


def _apply_schema(conn: sqlite3.Connection) -> None:
    """Execute the DDL from schema.sql."""
    ddl = _DB_PATH.read_text(encoding="utf-8")
    conn.executescript(ddl)
    conn.commit()
