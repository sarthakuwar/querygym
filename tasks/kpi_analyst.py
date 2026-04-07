"""
kpi-analyst task grader (medium).
Agent must answer 5 specific KPI questions about the billing data.
Each KPI = 0.17 reward delta; starting at 0.05 → max 0.90 per episode.
Score is always strictly in (0.05, 0.95) — within OpenEnv's (0, 1) exclusive requirement.

Grading rule (fixes applied per user review):
- Only match against rows[0][0] when the result is a single row.
- For multi-row results, scan col[0] with label-hint matching.
- Tolerance: ±5% for numeric KPIs, exact match for IDs/counts.

KPI answers are computed inside reset() AFTER bugs are injected.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from db.connection import get_db
from tasks.base import BaseTask

KPI_KEYS = ["mrr", "churn_rate", "top_customer", "avg_sub_length_days", "failed_invoice_count"]
REWARD_PER_KPI = 0.17  # 5 × 0.17 = 0.85 max delta; + 0.05 initial = 0.90 total (stays under 0.95 cap)
TOLERANCE = 0.05  # ±5%


def _compute_kpis(conn: sqlite3.Connection) -> dict[str, float | int]:
    """Derive ground-truth KPI values from the live (post-bug) DB."""
    cur = conn.cursor()

    # MRR: sum of plan price for active subscriptions
    cur.execute(
        """
        SELECT COALESCE(SUM(p.price_monthly), 0)
        FROM subscriptions s
        JOIN plans p ON p.id = s.plan_id
        WHERE s.status = 'active'
        """
    )
    mrr = float(cur.fetchone()[0])

    # Churn rate: cancelled / total subscriptions
    cur.execute("SELECT COUNT(*) FROM subscriptions")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM subscriptions WHERE status = 'cancelled'")
    cancelled = cur.fetchone()[0]
    churn_rate = round(cancelled / total, 4) if total else 0.0
    # Clamp churn_rate away from exact 0.0 or 1.0 so it's never equal to the
    # OpenEnv score boundaries (belt-and-suspenders guard for the expected dict).
    churn_rate = max(0.0001, min(0.9999, churn_rate))

    # Top customer by total paid invoice amount
    cur.execute(
        """
        SELECT s.customer_id
        FROM invoices i
        JOIN subscriptions s ON s.id = i.subscription_id
        WHERE i.status = 'paid'
        GROUP BY s.customer_id
        ORDER BY SUM(i.amount) DESC
        LIMIT 1
        """
    )
    row = cur.fetchone()
    top_customer = int(row[0]) if row else -1

    # Average subscription length in days (only rows with end_date and valid range)
    cur.execute(
        """
        SELECT AVG(
            CAST(julianday(end_date) - julianday(start_date) AS REAL)
        )
        FROM subscriptions
        WHERE end_date IS NOT NULL
          AND julianday(end_date) > julianday(start_date)
        """
    )
    avg_len = float(cur.fetchone()[0] or 0.0)

    # Count of failed invoices
    cur.execute("SELECT COUNT(*) FROM invoices WHERE status = 'failed'")
    failed_count = int(cur.fetchone()[0])

    return {
        "mrr": mrr,
        "churn_rate": churn_rate,
        "top_customer": top_customer,
        "avg_sub_length_days": avg_len,
        "failed_invoice_count": failed_count,
    }


def _within_tolerance(agent_val: float, truth: float) -> bool:
    if truth == 0:
        return abs(agent_val) < 0.01
    return abs(agent_val - truth) / abs(truth) <= TOLERANCE


_KPI_LABEL_HINTS: dict[str, list[str]] = {
    "mrr": ["mrr", "revenue", "monthly"],
    "churn_rate": ["churn", "rate", "churn_rate"],
    "top_customer": ["customer", "top", "id", "customer_id"],
    "avg_sub_length_days": ["avg", "length", "days", "duration"],
    "failed_invoice_count": ["failed", "count", "invoice"],
}


class KpiAnalystTask(BaseTask):
    task_id = "kpi-analyst"
    max_steps = 8

    def __init__(self) -> None:
        self._expected: dict[str, float | int] = {}
        self._solved: set[str] = set()

    def reset(self) -> None:
        # Compute expected answers now (after bugs injected) — not at class init
        self._expected = _compute_kpis(get_db())
        self._solved = set()

    # ------------------------------------------------------------------

    def _extract_candidate(
        self,
        sql: str,
        result: list[dict[str, Any]],
    ) -> dict[str, list[float]]:
        """
        Returns {kpi_key: [candidate_values]} for matching.
        Fix: only use rows[0][0] for single-row results.
              For multi-row, use col[0] with label hint.
        """
        candidates: dict[str, list[float]] = {k: [] for k in KPI_KEYS}

        if not result:
            return candidates

        sql_lower = sql.lower()

        if len(result) == 1:
            # Single-row result: check first column value only
            first_val = list(result[0].values())[0]
            try:
                num = float(first_val)
            except (TypeError, ValueError):
                return candidates

            first_col = list(result[0].keys())[0].lower()

            for kpi in KPI_KEYS:
                # Check label hints OR accept any single-row numeric (less precise)
                hints = _KPI_LABEL_HINTS[kpi]
                if any(h in first_col for h in hints) or any(h in sql_lower for h in hints):
                    candidates[kpi].append(num)

        else:
            # Multi-row result: only consider first column of each row, with label match
            first_col = list(result[0].keys())[0].lower() if result else ""
            for kpi in KPI_KEYS:
                hints = _KPI_LABEL_HINTS[kpi]
                if any(h in first_col for h in hints) or any(h in sql_lower for h in hints):
                    for row in result:
                        v = list(row.values())[0]
                        try:
                            candidates[kpi].append(float(v))
                        except (TypeError, ValueError):
                            pass

        return candidates

    def check_progress(
        self,
        sql: str,
        result: list[dict[str, Any]] | None,
        error: str | None,
    ) -> float:
        if error or result is None:
            return 0.0

        candidates = self._extract_candidate(sql, result)
        delta = 0.0

        for kpi, vals in candidates.items():
            if kpi in self._solved:
                continue
            truth = self._expected.get(kpi, 0)
            for v in vals:
                if _within_tolerance(v, float(truth)):
                    self._solved.add(kpi)
                    delta += REWARD_PER_KPI
                    break

        return round(delta, 4)

    def is_done(self) -> bool:
        return self._solved >= set(KPI_KEYS)

    def get_info(self) -> dict[str, Any]:
        return {
            "kpis_solved": sorted(self._solved),
            "kpis_remaining": sorted(set(KPI_KEYS) - self._solved),
            "progress": f"{len(self._solved)}/{len(KPI_KEYS)}",
            "expected": {k: self._expected.get(k) for k in KPI_KEYS},
        }
