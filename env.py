"""
Core QueryGym environment logic.
Manages episode state, SQL execution, reward shaping, and task dispatch.

Fixes applied per review:
- Whitelist approach for SQL guard (only SELECT/WITH/EXPLAIN/PRAGMA reads allowed)
- threading.Lock around step() and reset() for concurrent safety
"""

from __future__ import annotations

import re
import sqlite3
import threading
from typing import Any

from db.connection import get_db, reset_db
from models import QueryAction, QueryObservation, QueryEnvState
from tasks.base import BaseTask
from tasks.schema_explorer import SchemaExplorerTask
from tasks.kpi_analyst import KpiAnalystTask
from tasks.data_debugger import DataDebuggerTask

# ---------------------------------------------------------------------------
# SQL safety whitelist
# ---------------------------------------------------------------------------
# Only queries starting with these tokens are permitted.
_ALLOWED_PREFIXES = re.compile(
    r"^\s*(select|with|explain|pragma\s+\w+\s*(?!\s*=))",
    re.IGNORECASE,
)
# Explicitly block known PRAGMA write forms regardless of prefix check
_BLOCKED_PRAGMA_WRITES = re.compile(
    r"pragma\s+\w+\s*=",
    re.IGNORECASE,
)

_SCHEMA_HINT = "plans, customers, subscriptions, invoices, events"

_TASK_REGISTRY: dict[str, type[BaseTask]] = {
    "schema-explorer": SchemaExplorerTask,
    "kpi-analyst": KpiAnalystTask,
    "data-debugger": DataDebuggerTask,
}

# Reward deltas
PENALTY_DESTRUCTIVE = -0.03
PENALTY_SYNTAX_ERROR = -0.005
PENALTY_REPEAT_QUERY = -0.01

# Hard bounds for the cumulative episode score — strictly (0, 1) required by OpenEnv
_SCORE_MIN: float = 0.05   # never returned as 0.0 even with zero agent progress
_SCORE_MAX: float = 0.95   # cap away from 1.0 even if agent is perfect

# Query execution timeout (seconds)
_QUERY_TIMEOUT = 5


def _clamp_score(value: float) -> float:
    """Return value strictly inside (0, 1), hard-clamped to [_SCORE_MIN, _SCORE_MAX]."""
    clamped = max(_SCORE_MIN, min(_SCORE_MAX, value))
    # Belt-and-suspenders: guarantee the literal float is not 0.0 or 1.0
    if clamped <= 0.0:
        clamped = _SCORE_MIN
    elif clamped >= 1.0:
        clamped = _SCORE_MAX
    return round(clamped, 4)


def _is_allowed_sql(sql: str) -> bool:
    """Return True only if the SQL starts with a whitelisted, non-write prefix."""
    if _BLOCKED_PRAGMA_WRITES.search(sql):
        return False
    return bool(_ALLOWED_PREFIXES.match(sql.strip()))


def _execute_sql(conn: sqlite3.Connection, sql: str) -> tuple[list[dict[str, Any]], str | None]:
    """
    Run sql and return (rows, error).
    rows is a list of dicts; error is a string or None.
    """
    try:
        conn.execute(f"PRAGMA busy_timeout = {_QUERY_TIMEOUT * 1000}")
        cur = conn.execute(sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = [dict(zip(columns, row)) for row in cur.fetchall()]
        return rows, None
    except (sqlite3.OperationalError, sqlite3.DatabaseError, sqlite3.ProgrammingError) as exc:
        return [], str(exc)


class QueryEnv:
    """Single-session OpenEnv-style environment."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._task: BaseTask | None = None
        self._task_id: str = "schema-explorer"
        self._step: int = 0
        self._done: bool = False
        self._total_reward: float = _SCORE_MIN
        self._last_action: str | None = None
        self._history: list[str] = []  # SQL strings this episode

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, task_id: str = "schema-explorer") -> QueryObservation:
        with self._lock:
            if task_id not in _TASK_REGISTRY:
                raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(_TASK_REGISTRY)}")

            reset_db()  # Wipe and re-seed each episode

            self._task_id = task_id
            self._task = _TASK_REGISTRY[task_id]()
            self._task.reset()
            self._step = 0
            self._done = False
            self._total_reward = _SCORE_MIN
            self._last_action = None
            self._history = []

            return QueryObservation(
                task_id=task_id,
                step=0,
                result=None,
                error=None,
                schema_hint=_SCHEMA_HINT,
            )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action: QueryAction) -> tuple[QueryObservation, float, bool, dict[str, Any]]:
        with self._lock:
            if self._done:
                raise RuntimeError("Episode is done. Call /reset to start a new episode.")

            sql = action.sql.strip()
            step_reward = 0.0
            error: str | None = None
            result: list[dict[str, Any]] | None = None

            # 1. Whitelist check (fix #3: replaces narrow blacklist)
            if not _is_allowed_sql(sql):
                error = (
                    "Destructive or disallowed SQL detected. "
                    "Only SELECT / WITH / EXPLAIN / PRAGMA reads are permitted."
                )
                self._total_reward = _clamp_score(self._total_reward + PENALTY_DESTRUCTIVE)
                self._step += 1
                self._last_action = sql
                obs = QueryObservation(
                    task_id=self._task_id,
                    step=self._step,
                    result=None,
                    error=error,
                    schema_hint=_SCHEMA_HINT,
                )
                done = self._step >= self._task.max_steps
                self._done = done
                return obs, _clamp_score(self._total_reward), done, self._get_info_dict(error)

            # 2. Repeat query penalty
            if sql in self._history:
                step_reward += PENALTY_REPEAT_QUERY

            # 3. Execute SQL
            conn = get_db()
            result, error = _execute_sql(conn, sql)

            # 4. Syntax error penalty
            if error:
                step_reward += PENALTY_SYNTAX_ERROR
                result = None

            # 5. Task grader (only called on successful execution)
            if self._task and not error:
                delta = self._task.check_progress(sql, result, error)
                step_reward += delta

            # 6. Update state
            self._history.append(sql)
            # Accumulate and strictly clamp total_reward to (0, 1) — validator requires this
            self._total_reward = _clamp_score(self._total_reward + step_reward)
            self._step += 1
            self._last_action = sql

            task_done = self._task.is_done() if self._task else False
            max_steps_hit = self._step >= self._task.max_steps
            self._done = task_done or max_steps_hit

            obs = QueryObservation(
                task_id=self._task_id,
                step=self._step,
                result=result,
                error=error,
                schema_hint=_SCHEMA_HINT,
            )
            # Return total_reward as the score — always strictly in (0.01, 0.99)
            return obs, _clamp_score(self._total_reward), self._done, self._get_info_dict(error)

    # ------------------------------------------------------------------
    # state snapshot
    # ------------------------------------------------------------------
    def get_state(self) -> QueryEnvState:
        with self._lock:
            return QueryEnvState(
                task_id=self._task_id,
                step=self._step,
                max_steps=self._task.max_steps if self._task else 8,
                done=self._done,
                total_reward=_clamp_score(self._total_reward),
                last_action=self._last_action,
                history=list(self._history[-10:]),  # last 10 queries
            )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _get_info_dict(self, error: str | None) -> dict[str, Any]:
        task_info = self._task.get_info() if self._task else {}
        return {
            **task_info,
            "error": error,
            "total_reward": _clamp_score(self._total_reward),
        }
