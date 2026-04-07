"""
data-debugger task grader (hard).
Agent must find and report all 4 seeded integrity bugs via SQL queries.

Each bug = 0.25 reward, max 1.0 per episode.
Bug detection uses predicate matching against query results.
"""

from __future__ import annotations

from typing import Any

from db.bugs import ALL_BUGS, BUG_BAD_DATES, BUG_DUP_INVOICE, BUG_NEGATIVE_AMOUNT, BUG_ORPHAN_SUB
from tasks.base import BaseTask

REWARD_PER_BUG = 0.24  # max 0.96 per episode


def _check_orphan_sub(result: list[dict[str, Any]]) -> bool:
    """Result contains a row where customer_id = 9999 (orphaned subscription)."""
    for row in result:
        vals = {str(k).lower(): v for k, v in row.items()}
        cid = vals.get("customer_id") or vals.get("customerid") or vals.get("cid")
        if cid is not None:
            try:
                if int(cid) == 9999:
                    return True
            except (ValueError, TypeError):
                pass
        # Also accept if a column named 'id' has value 9999 and context suggests subscription
        for v in row.values():
            try:
                if int(v) == 9999:
                    return True
            except (ValueError, TypeError):
                pass
    return False


def _check_dup_invoice(result: list[dict[str, Any]]) -> bool:
    """Result contains two rows with identical (subscription_id, amount, due_date)."""
    seen: set[tuple] = set()
    for row in result:
        vals = {str(k).lower(): v for k, v in row.items()}
        key = (
            vals.get("subscription_id"),
            vals.get("amount"),
            vals.get("due_date"),
        )
        if None not in key:
            if key in seen:
                return True
            seen.add(key)
    return False


def _check_negative_amount(result: list[dict[str, Any]]) -> bool:
    """Result contains a row where amount < 0."""
    for row in result:
        vals = {str(k).lower(): v for k, v in row.items()}
        amount = vals.get("amount")
        if amount is not None:
            try:
                if float(amount) < 0:
                    return True
            except (ValueError, TypeError):
                pass
    return False


def _check_bad_dates(result: list[dict[str, Any]]) -> bool:
    """Result contains a row where end_date < start_date (lexicographic, ISO-8601 safe)."""
    for row in result:
        vals = {str(k).lower(): v for k, v in row.items()}
        start = vals.get("start_date")
        end = vals.get("end_date")
        if start and end:
            try:
                if str(end) < str(start):
                    return True
            except TypeError:
                pass
    return False


_DETECTORS = {
    BUG_ORPHAN_SUB: _check_orphan_sub,
    BUG_DUP_INVOICE: _check_dup_invoice,
    BUG_NEGATIVE_AMOUNT: _check_negative_amount,
    BUG_BAD_DATES: _check_bad_dates,
}


class DataDebuggerTask(BaseTask):
    task_id = "data-debugger"
    max_steps = 8

    def __init__(self) -> None:
        self._found: set[str] = set()

    def reset(self) -> None:
        self._found = set()

    def check_progress(
        self,
        sql: str,
        result: list[dict[str, Any]] | None,
        error: str | None,
    ) -> float:
        if error or result is None:
            return 0.0

        delta = 0.0
        for bug_id, detector in _DETECTORS.items():
            if bug_id not in self._found and detector(result):
                self._found.add(bug_id)
                delta += REWARD_PER_BUG

        return round(delta, 4)

    def is_done(self) -> bool:
        return self._found >= set(ALL_BUGS)

    def get_info(self) -> dict[str, Any]:
        return {
            "bugs_found": sorted(self._found),
            "bugs_remaining": sorted(set(ALL_BUGS) - self._found),
            "progress": f"{len(self._found)}/{len(ALL_BUGS)}",
        }
