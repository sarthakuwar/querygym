-- QueryGym SQLite schema
-- All dates stored as ISO-8601 strings (TEXT)

CREATE TABLE IF NOT EXISTS plans (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,           -- e.g. Starter, Pro, Business, Enterprise
    price_monthly   REAL    NOT NULL,
    billing_cycle   TEXT    NOT NULL            -- 'monthly' | 'annual'
);

CREATE TABLE IF NOT EXISTS customers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    email       TEXT    NOT NULL UNIQUE,
    created_at  TEXT    NOT NULL,               -- ISO-8601
    country     TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS subscriptions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    plan_id     INTEGER NOT NULL REFERENCES plans(id),
    status      TEXT    NOT NULL,               -- 'active' | 'cancelled' | 'trialing'
    start_date  TEXT    NOT NULL,               -- ISO-8601
    end_date    TEXT                            -- NULL = still active
);

CREATE TABLE IF NOT EXISTS invoices (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    subscription_id INTEGER NOT NULL REFERENCES subscriptions(id),
    amount          REAL    NOT NULL,
    status          TEXT    NOT NULL,           -- 'paid' | 'failed' | 'pending'
    due_date        TEXT    NOT NULL,           -- ISO-8601
    paid_date       TEXT                        -- NULL if not paid
);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    event_type  TEXT    NOT NULL,               -- 'signup' | 'upgrade' | 'downgrade' | 'churn'
    occurred_at TEXT    NOT NULL,               -- ISO-8601
    metadata    TEXT                            -- JSON blob
);
