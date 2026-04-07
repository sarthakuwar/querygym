"""
schema-explorer task grader (easy).
Agent must discover all table names, all column names, and row counts.

11 milestones × 0.08 reward each = 0.88 max per episode.
Reward is incremental: only newly achieved milestones earn reward each step.
"""

from __future__ import annotations

import re
from typing import Any

from tasks.base import BaseTask

TABLES = ["plans", "customers", "subscriptions", "invoices", "events"]

# Milestone keys
MILESTONE_TABLE_LIST = "table_list"
MILESTONE_COLS = {t: f"cols_{t}" for t in TABLES}
MILESTONE_COUNT = {t: f"count_{t}" for t in TABLES}

ALL_MILESTONES: list[str] = (
    [MILESTONE_TABLE_LIST]
    + list(MILESTONE_COLS.values())
    + list(MILESTONE_COUNT.values())
)
REWARD_PER_MILESTONE = 0.08  # max 0.88 per episode


class SchemaExplorerTask(BaseTask):
    task_id = "schema-explorer"
    max_steps = 8

    def __init__(self) -> None:
        self._achieved: set[str] = set()

    def reset(self) -> None:
        self._achieved = set()

    # ------------------------------------------------------------------
    # Grading helpers
    # ------------------------------------------------------------------

    def _check_result(
        self,
        sql: str,
        result: list[dict[str, Any]] | None,
    ) -> set[str]:
        """Return the set of milestones newly covered by this (sql, result) pair."""
        new: set[str] = set()
        if result is None:
            return new

        sql_lower = sql.lower()
        col_names_lower = (
            {k.lower() for row in result for k in row.keys()} if result else set()
        )
        values_lower = {
            str(v).lower() for row in result for v in row.values() if v is not None
        }

        # ---- Milestone: all table names listed ----
        if MILESTONE_TABLE_LIST not in self._achieved:
            found_tables = {t for t in TABLES if t in values_lower}
            if found_tables == set(TABLES):
                new.add(MILESTONE_TABLE_LIST)

        # ---- Milestones: column discovery per table ----
        for table in TABLES:
            key = MILESTONE_COLS[table]
            if key not in self._achieved:
                # PRAGMA table_info or SELECT * returning column names
                if (
                    f"pragma table_info({table})" in sql_lower
                    or f"pragma table_info('{table}')" in sql_lower
                    or (table in sql_lower and col_names_lower - {"name", "type", "notnull", "dflt_value", "pk"})
                ):
                    # Accept if the query targets this table and returned rows
                    if table in sql_lower and result:
                        new.add(key)

        # ---- Milestones: row count per table ----
        for table in TABLES:
            key = MILESTONE_COUNT[table]
            if key not in self._achieved:
                # Single-row result with a numeric value where sql mentions COUNT and table
                if (
                    "count" in sql_lower
                    and table in sql_lower
                    and len(result) == 1
                ):
                    vals = list(result[0].values())
                    if vals and isinstance(vals[0], (int, float)) and vals[0] > 0:
                        new.add(key)

        return new

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def check_progress(
        self,
        sql: str,
        result: list[dict[str, Any]] | None,
        error: str | None,
    ) -> float:
        if error:
            return 0.0
        new = self._check_result(sql, result)
        new -= self._achieved  # only genuinely new ones
        self._achieved |= new
        return round(len(new) * REWARD_PER_MILESTONE, 4)

    def is_done(self) -> bool:
        return self._achieved >= set(ALL_MILESTONES)

    def get_info(self) -> dict[str, Any]:
        return {
            "milestones_achieved": sorted(self._achieved),
            "milestones_remaining": sorted(set(ALL_MILESTONES) - self._achieved),
            "progress": f"{len(self._achieved)}/{len(ALL_MILESTONES)}",
        }
