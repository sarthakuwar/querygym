"""
Pydantic v2 models for the QueryGym OpenEnv interface.
All models used across app.py, env.py, and inference.py.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class QueryAction(BaseModel):
    """Action submitted by the agent — a single SQL string."""
    sql: str = Field(..., description="A valid SQLite SELECT statement")


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class QueryObservation(BaseModel):
    """Observation returned after reset() or step()."""
    task_id: str = Field(..., description="Active task name")
    step: int = Field(..., description="Current step number (0-indexed)")
    result: list[dict[str, Any]] | None = Field(
        None,
        description="Rows returned by agent's last query (None on reset)",
    )
    error: str | None = Field(
        None,
        description="SQL error message if query failed, else null",
    )
    schema_hint: str = Field(
        ...,
        description="Comma-separated table names as a minimal schema hint",
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class QueryEnvState(BaseModel):
    """Full episode state snapshot returned by GET /state."""
    task_id: str
    step: int
    max_steps: int
    done: bool
    total_reward: float
    last_action: str | None
    history: list[str] = Field(
        default_factory=list,
        description="SQL queries executed this episode (most recent last)",
    )


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """Body for POST /reset.  task_id is optional."""
    task_id: str = "schema-explorer"
