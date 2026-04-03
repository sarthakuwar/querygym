"""
Seeds the QueryGym SQLite database with ~200 rows of realistic fake SaaS data.
random.seed(42) and Faker.seed(42) are called INSIDE seed() to ensure
full determinism across multiple reset() calls.
"""

import json
import random
import sqlite3
from datetime import date, timedelta

from faker import Faker


PLANS = [
    ("Starter",    9.99,  "monthly"),
    ("Pro",        29.99, "monthly"),
    ("Business",   79.99, "monthly"),
    ("Enterprise", 249.99, "annual"),
]

NUM_CUSTOMERS = 60
STATUSES = ["active", "cancelled", "trialing"]
EVENT_TYPES = ["signup", "upgrade", "downgrade", "churn"]
INVOICE_STATUSES = ["paid", "failed", "pending"]


def _rand_date(fake: Faker, start: date, end: date) -> str:
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).isoformat()


def seed(conn: sqlite3.Connection) -> None:
    """
    Populate plans, customers, subscriptions, invoices, events.
    Resets PRNG seeds at the top so re-seeding after reset() is deterministic.
    """
    random.seed(42)
    Faker.seed(42)
    fake = Faker()

    cur = conn.cursor()

    # ---- plans -------------------------------------------------------
    plan_ids: list[int] = []
    for name, price, cycle in PLANS:
        cur.execute(
            "INSERT INTO plans (name, price_monthly, billing_cycle) VALUES (?, ?, ?)",
            (name, price, cycle),
        )
        plan_ids.append(cur.lastrowid)

    # ---- customers ---------------------------------------------------
    customer_ids: list[int] = []
    created_ats: list[date] = []
    start_window = date(2021, 1, 1)
    end_window = date(2023, 12, 31)

    for _ in range(NUM_CUSTOMERS):
        created = fake.date_between(start_date=start_window, end_date=end_window)
        cur.execute(
            "INSERT INTO customers (name, email, created_at, country) VALUES (?, ?, ?, ?)",
            (fake.name(), fake.unique.email(), created.isoformat(), fake.country_code()),
        )
        customer_ids.append(cur.lastrowid)
        created_ats.append(created)

    # ---- subscriptions & invoices ------------------------------------
    sub_ids: list[int] = []
    for idx, cust_id in enumerate(customer_ids):
        # 1–2 subscriptions per customer
        num_subs = random.choices([1, 2], weights=[0.7, 0.3])[0]
        cust_start = created_ats[idx]

        for s in range(num_subs):
            plan_id = random.choice(plan_ids)
            status = random.choices(STATUSES, weights=[0.55, 0.35, 0.10])[0]

            sub_start = cust_start + timedelta(days=random.randint(0, 60))
            if status == "cancelled":
                sub_end_days = random.randint(30, 500)
                sub_end = sub_start + timedelta(days=sub_end_days)
                end_date = sub_end.isoformat()
            else:
                end_date = None

            cur.execute(
                """
                INSERT INTO subscriptions (customer_id, plan_id, status, start_date, end_date)
                VALUES (?, ?, ?, ?, ?)
                """,
                (cust_id, plan_id, status, sub_start.isoformat(), end_date),
            )
            sub_id = cur.lastrowid
            sub_ids.append(sub_id)

            # Fetch plan price for invoice generation
            cur.execute("SELECT price_monthly FROM plans WHERE id = ?", (plan_id,))
            price = cur.fetchone()[0]

            # Generate 1–4 monthly invoices per subscription
            num_invoices = random.randint(1, 4)
            for i in range(num_invoices):
                due = sub_start + timedelta(days=30 * (i + 1))
                inv_status = random.choices(
                    INVOICE_STATUSES, weights=[0.75, 0.15, 0.10]
                )[0]
                paid_date = (
                    (due + timedelta(days=random.randint(0, 5))).isoformat()
                    if inv_status == "paid"
                    else None
                )
                cur.execute(
                    """
                    INSERT INTO invoices (subscription_id, amount, status, due_date, paid_date)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (sub_id, round(price, 2), inv_status, due.isoformat(), paid_date),
                )

    # ---- events ------------------------------------------------------
    for idx, cust_id in enumerate(customer_ids):
        # Always emit a signup event
        signup_date = created_ats[idx]
        cur.execute(
            "INSERT INTO events (customer_id, event_type, occurred_at, metadata) VALUES (?, ?, ?, ?)",
            (cust_id, "signup", signup_date.isoformat(), json.dumps({"source": fake.bs()})),
        )
        # 0–2 additional lifecycle events
        extra = random.randint(0, 2)
        for _ in range(extra):
            ev_type = random.choices(
                ["upgrade", "downgrade", "churn"], weights=[0.4, 0.3, 0.3]
            )[0]
            ev_date = signup_date + timedelta(days=random.randint(30, 700))
            cur.execute(
                "INSERT INTO events (customer_id, event_type, occurred_at, metadata) VALUES (?, ?, ?, ?)",
                (cust_id, ev_type, ev_date.isoformat(), json.dumps({"reason": fake.sentence()})),
            )

    conn.commit()
