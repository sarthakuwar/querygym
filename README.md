---
title: QueryGym
emoji: 🗄️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# QueryGym 🗄️

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant reinforcement-learning gym where an AI agent learns to query a SaaS subscription billing SQLite database using SQL.

## Overview

QueryGym presents an agent with a realistic SaaS billing database (SQLite, seeded with ~200 rows) and three progressively harder tasks. The agent interacts via SQL queries; the gym grades each query and returns a reward signal.

| Table | Description |
|---|---|
| `plans` | Subscription plans (Starter, Pro, Business, Enterprise) |
| `customers` | 60 customer accounts |
| `subscriptions` | 80 subscriptions with status + date ranges |
| `invoices` | ~120 billing invoices |
| `events` | ~150 lifecycle events (signup, upgrade, churn) |

---

## Action Space

```json
{ "sql": "SELECT name FROM sqlite_master WHERE type='table'" }
```

`sql` — A single SQLite **read-only** query (`SELECT`, `WITH`, `EXPLAIN`, or PRAGMA reads). Destructive statements are refused and penalised.

## Observation Space

```json
{
  "task_id":    "schema-explorer",
  "step":       1,
  "result":     [{"name": "plans"}, {"name": "customers"}, ...],
  "error":      null,
  "schema_hint": "plans, customers, subscriptions, invoices, events"
}
```

---

## Tasks

### 1. `schema-explorer` (easy)
Discover all table names, column names, and row counts.  
**10 milestones × 0.10 reward** | max 8 steps

### 2. `kpi-analyst` (medium)
Answer 5 key KPI questions:
- Monthly Recurring Revenue (MRR)
- Churn rate
- Top customer by total paid revenue
- Average subscription length (days)
- Count of failed invoices

**5 KPIs × 0.20 reward** | max 8 steps | ±5% tolerance on numeric answers

### 3. `data-debugger` (hard)
Find all 4 seeded data integrity bugs:
- Orphaned subscription (no matching customer)
- Duplicate invoice
- Negative invoice amount
- Subscription with `end_date` before `start_date`

**4 bugs × 0.25 reward** | max 8 steps

---

## Reward Shape

| Trigger | Reward |
|---|---|
| Task milestone / KPI solved / bug found | +0.10 – +0.25 (per task) |
| Destructive SQL (ALTER, DROP, DELETE, INSERT, UPDATE, PRAGMA writes) | -0.05 (query refused) |
| SQL syntax / runtime error | -0.01 |
| Repeated identical query | -0.02 |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start a new episode. Body: `{"task_id": "schema-explorer"}` |
| `POST` | `/step` | Submit an SQL action. Body: `{"sql": "..."}` |
| `GET` | `/state` | Get current episode state |
| `GET` | `/healthz` | Health check |

---

## Setup

### Local Development

```bash
git clone <repo>
cd querygym
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

### Docker

```bash
docker build -t querygym .
docker run -p 7860:7860 querygym
```

### Run Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen3-30B-A3B:novita"
export HF_TOKEN="hf_..."
python inference.py
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `API_BASE_URL` | Yes | — | OpenAI-compatible base URL |
| `MODEL_NAME` | Yes | — | Model identifier |
| `HF_TOKEN` | Yes | — | Hugging Face access token |
| `GYM_BASE_URL` | No | `http://localhost:7860` | Gym server URL for inference |

---

## Baseline Scores (placeholder)

| Task | Model | Avg Reward | Avg Steps |
|---|---|---|---|
| schema-explorer | — | — | — |
| kpi-analyst | — | — | — |
| data-debugger | — | — | — |
