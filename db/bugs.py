"""
Injects 4 deterministic integrity bugs into the already-seeded DB.
Called once per reset, after seed.py finishes.

Bug constants are exported so task graders can reference them.
"""

import sqlite3

# Ground-truth bug identifiers used by data_debugger grader
BUG_ORPHAN_SUB = "orphan_sub"
BUG_DUP_INVOICE = "dup_invoice"
BUG_NEGATIVE_AMOUNT = "negative_amount"
BUG_BAD_DATES = "bad_dates"

ALL_BUGS = [BUG_ORPHAN_SUB, BUG_DUP_INVOICE, BUG_NEGATIVE_AMOUNT, BUG_BAD_DATES]


def inject_bugs(conn: sqlite3.Connection) -> None:
    """
    Insert / mutate exactly 4 integrity bugs.
    This must run AFTER seed() so row ids are stable.
    FK enforcement is temporarily disabled for the orphan bug insert.
    """
    cur = conn.cursor()

    # ------------------------------------------------------------------
    # Bug 1: Orphaned subscription — customer_id points to a non-existent customer
    # Disable FK checks so the intentionally bad insert succeeds.
    # ------------------------------------------------------------------
    cur.execute("PRAGMA foreign_keys = OFF")
    cur.execute(
        """
        INSERT INTO subscriptions (customer_id, plan_id, status, start_date, end_date)
        VALUES (9999, 1, 'active', '2024-01-01', NULL)
        """
    )
    cur.execute("PRAGMA foreign_keys = ON")

    # ------------------------------------------------------------------
    # Bug 2: Duplicate invoice — exact copy of the first invoice in the DB
    # ------------------------------------------------------------------
    cur.execute("SELECT subscription_id, amount, status, due_date FROM invoices LIMIT 1")
    row = cur.fetchone()
    if row:
        sub_id, amount, status, due_date = row
        cur.execute(
            """
            INSERT INTO invoices (subscription_id, amount, status, due_date, paid_date)
            VALUES (?, ?, ?, ?, NULL)
            """,
            (sub_id, amount, status, due_date),
        )

    # ------------------------------------------------------------------
    # Bug 3: Negative invoice amount — mutate the invoice with id=2
    # ------------------------------------------------------------------
    cur.execute("UPDATE invoices SET amount = -149.99 WHERE id = 2")

    # ------------------------------------------------------------------
    # Bug 4: end_date before start_date — mutate subscription with id=2
    # ------------------------------------------------------------------
    cur.execute(
        "UPDATE subscriptions SET end_date = '2020-01-01' WHERE id = 2"
    )

    conn.commit()
