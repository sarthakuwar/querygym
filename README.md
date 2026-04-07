---
title: QueryGym
emoji: 🗄️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - sql
  - agent-evaluation
---

# QueryGym 🗄️

> **An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant reinforcement-learning environment for evaluating SQL-reasoning agents on a realistic SaaS billing database.**

QueryGym challenges agents to navigate a real-world-style SQLite database through a structured sequence of progressively harder analytical tasks — from schema discovery to KPI computation to data-integrity forensics. The environment provides dense, step-level reward signals, deterministic episode reset, and full OpenEnv protocol compliance.

---

## Motivation

Most SQL benchmarks (Spider, BIRD, WikiSQL) evaluate end-to-end query generation against a static ground truth. They do not evaluate an agent's ability to *explore* an unfamiliar schema, *iteratively refine* queries based on observed results, or *detect anomalies* in live data. QueryGym fills this gap by framing SQL interaction as a multi-step RL episode — rewarding incremental progress rather than a single correct answer.

This makes QueryGym directly useful for evaluating:
- Tool-use agents that call SQL in an agentic loop
- Reasoning models being fine-tuned with RL on tabular data
- Multi-step planning capabilities of LLMs in analytical tasks

---

## Environment Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Agent (LLM)                       │
│               POST /step  {"sql": "..."}             │
└────────────────────────┬─────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────┐
│                  QueryEnv (FastAPI)                  │
│                                                      │
│  ┌─────────────┐   ┌──────────────┐  ┌────────────┐ │
│  │  SQL Guard  │→  │ SQLite Exec  │→ │  Grader    │ │
│  │ (whitelist) │   │ (in-memory)  │  │ (per-task) │ │
│  └─────────────┘   └──────────────┘  └─────┬──────┘ │
│                                             │        │
│  total_reward ← clamp(prev + Δreward, 0.05, 0.95)   │
└─────────────────────────────────────────────┼────────┘
                                              │
                              {observation, reward, done, info}
```

**Database schema (SQLite, in-memory, deterministic seed 42):**

| Table | Rows | Description |
|---|---|---|
| `plans` | 4 | Subscription tiers: Starter ($9.99), Pro ($29.99), Business ($79.99), Enterprise ($249.99) |
| `customers` | 60 | Accounts with `name`, `email`, `country`, `created_at` |
| `subscriptions` | ~80 | Per-customer plan subscriptions; `status ∈ {active, cancelled, trialing}` |
| `invoices` | ~120 | Monthly billing records; `status ∈ {paid, failed, pending}` |
| `events` | ~150 | Lifecycle signals: `signup`, `upgrade`, `downgrade`, `churn` |

Four deterministic data-integrity bugs are injected after each `reset()` to support the `data-debugger` task:
1. **Orphaned subscription** — `customer_id = 9999` references a non-existent customer
2. **Duplicate invoice** — exact copy of the first invoice record
3. **Negative invoice amount** — `invoices.amount = -149.99`
4. **Chronologically invalid subscription** — `end_date ('2020-01-01') < start_date`

---

## Action Space

Each step accepts a single SQL string. Only **read-only** operations are permitted:

```json
{ "sql": "SELECT s.customer_id, SUM(i.amount) AS total_paid FROM invoices i JOIN subscriptions s ON s.id = i.subscription_id WHERE i.status = 'paid' GROUP BY s.customer_id ORDER BY total_paid DESC LIMIT 1" }
```

**Permitted prefixes:** `SELECT`, `WITH` (CTEs), `EXPLAIN`, `PRAGMA <key>` (read-only)  
**Blocked:** `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `PRAGMA <key> = <value>`

Blocked queries are refused before execution and incur a penalty without consuming a database I/O.

---

## Observation Space

Every `/step` response returns a structured JSON observation:

```json
{
  "observation": {
    "task_id":     "kpi-analyst",
    "step":        3,
    "result":      [{"total_mrr": 2847.62}],
    "error":       null,
    "schema_hint": "plans, customers, subscriptions, invoices, events"
  },
  "reward": 0.22,
  "done":   false,
  "info": {
    "kpis_solved":    ["mrr"],
    "kpis_remaining": ["churn_rate", "top_customer", "avg_sub_length_days", "failed_invoice_count"],
    "progress":       "1/5",
    "total_reward":   0.22,
    "error":          null
  }
}
```

`result` is a list of row-dicts (column name → value). `null` on reset or SQL error.  
`info` exposes task-specific progress metadata to aid agent reasoning.

---

## Tasks

### Task 1 — `schema-explorer` *(easy)*

**Objective:** Fully map the database schema without prior knowledge.

The agent must discover: (a) all 5 table names, (b) the column structure of each table, and (c) the row count of each table — through 11 discrete milestones.

| Milestone | Condition | Reward Δ |
|---|---|---|
| `table_list` | All 5 table names appear in a single result | +0.08 |
| `cols_<table>` × 5 | `PRAGMA table_info(<t>)` or `SELECT *` with real column names | +0.08 each |
| `count_<table>` × 5 | `SELECT COUNT(*)` returning a positive integer | +0.08 each |

**Max reward delta:** 11 × 0.08 = 0.88 | **Max steps:** 8 | **Starting score:** 0.05  
**Episode max score:** 0.05 + 0.88 = **0.93**

---

### Task 2 — `kpi-analyst` *(medium)*

**Objective:** Derive five business KPIs from first principles using SQL.

Ground-truth KPI values are recomputed from the live (bug-injected) database after each `reset()` — agents cannot hard-code answers.

| KPI | SQL Pattern | Tolerance |
|---|---|---|
| `mrr` | `SUM(price_monthly)` for `active` subscriptions | ±5% |
| `churn_rate` | `cancelled_count / total_count` | ±5% |
| `top_customer` | Customer ID with highest total paid invoice amount | Exact |
| `avg_sub_length_days` | `AVG(julianday(end_date) - julianday(start_date))` | ±5% |
| `failed_invoice_count` | `COUNT(*)` where `status = 'failed'` | Exact |

**Reward Δ per KPI:** +0.17 | **Max delta:** 5 × 0.17 = 0.85 | **Max steps:** 8  
**Episode max score:** 0.05 + 0.85 = **0.90**

Partial credit is awarded per KPI; the agent does not need to answer all five to earn reward.

---

### Task 3 — `data-debugger` *(hard)*

**Objective:** Identify all four seeded data-integrity anomalies via exploratory SQL.

The agent must produce query results that *contain evidence* of each bug. Detection is purely result-driven — the agent is not told how many bugs exist or what they are.

| Bug | Detection Predicate | Reward Δ |
|---|---|---|
| Orphaned subscription | Result contains row with `customer_id = 9999` | +0.21 |
| Duplicate invoice | Result has two rows with identical `(subscription_id, amount, due_date)` | +0.21 |
| Negative amount | Result has row where `amount < 0` | +0.21 |
| Bad date range | Result has row where `end_date < start_date` (ISO-8601 string compare) | +0.21 |

**Max delta:** 4 × 0.21 = 0.84 | **Max steps:** 8  
**Episode max score:** 0.05 + 0.84 = **0.89**

This task genuinely challenges frontier models: no schema hint reveals the bugs, and the agent must reason about data-quality heuristics to formulate the right detection queries.

---

## Reward Function

The environment maintains a **cumulative episode score** `S ∈ (0.05, 0.95)`:

```
S_0 = 0.05    (episode baseline — strictly > 0 as required by OpenEnv)

At each step t:
  Δ_task  = grader output (positive when progress made, 0 otherwise)
  Δ_step  = Δ_task + penalties

  S_t = clamp(S_{t-1} + Δ_step, 0.05, 0.95)
```

**Reward components:**

| Event | Δ Reward |
|---|---|
| Task milestone solved (schema) | +0.08 per milestone |
| KPI correctly answered (kpi-analyst) | +0.17 per KPI |
| Data bug detected (data-debugger) | +0.21 per bug |
| Destructive / disallowed SQL (refused) | −0.03 |
| SQL syntax or runtime error | −0.005 |
| Repeated identical query | −0.01 |

Scores are clamped to `[0.05, 0.95]` at every step, satisfying the OpenEnv Phase 2 requirement that task scores are **strictly in (0, 1)**.

---

## API Reference

### `POST /reset`

Starts a fresh, deterministic episode. The database is wiped and re-seeded with `random.seed(42)`.

**Request body** (optional):
```json
{ "task_id": "schema-explorer" }
```
Valid `task_id` values: `schema-explorer`, `kpi-analyst`, `data-debugger`

**Response:** `QueryObservation` — initial observation with `step=0`, `result=null`.

---

### `POST /step`

Submit one SQL action and receive the environment's response.

**Request body:**
```json
{ "sql": "SELECT COUNT(*) FROM invoices WHERE status = 'failed'" }
```

**Response:**
```json
{
  "observation": { "task_id": "...", "step": 2, "result": [...], "error": null, "schema_hint": "..." },
  "reward": 0.26,
  "done": false,
  "info": { "bugs_found": ["negative_amount"], "bugs_remaining": [...], "progress": "1/4", "total_reward": 0.26 }
}
```

---

### `GET /state`

Returns the full current episode state snapshot (task, step, cumulative reward, action history).

### `GET /healthz`

Returns `{"status": "ok"}` — used by HF Spaces for liveness checks.

---

## Setup & Usage

### Docker (recommended)

```bash
docker build -t querygym .
docker run -p 7860:7860 querygym
# Server available at http://localhost:7860
```

### Local Development

```bash
git clone https://github.com/sarthakuwar/querygym
cd querygym
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

### Run Inference Agent

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen3-30B-A3B:novita"
export HF_TOKEN="hf_..."
export GYM_BASE_URL="http://localhost:7860"
python inference.py
```

### Quick API Smoke-Test

```bash
# Reset to kpi-analyst task
curl -s -X POST http://localhost:7860/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id": "kpi-analyst"}' | python -m json.tool

# Submit a query
curl -s -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{"sql": "SELECT SUM(p.price_monthly) AS mrr FROM subscriptions s JOIN plans p ON p.id = s.plan_id WHERE s.status = '\''active'\''"}' | python -m json.tool
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `API_BASE_URL` | Yes | `https://router.huggingface.co/v1` | OpenAI-compatible API endpoint for the LLM |
| `MODEL_NAME` | Yes | `Qwen/Qwen3-30B-A3B:novita` | Model identifier passed to the OpenAI client |
| `HF_TOKEN` | Yes | — | Hugging Face access token (used as API key) |
| `GYM_BASE_URL` | No | `http://localhost:7860` | Base URL of the running QueryGym server |

---

## Baseline Scores

Measured over 5 deterministic episodes (`random.seed(42)`, `temperature=0.2`) using `Qwen/Qwen3-30B-A3B` via the HuggingFace inference router. Score = final cumulative reward at episode end (strictly in (0.05, 0.95)).

| Task | Difficulty | Model | Avg Score | Avg Steps | Success Rate |
|---|---|---|---|---|---|
| `schema-explorer` | Easy | Qwen3-30B-A3B | 0.61 | 7.2 | 60% |
| `kpi-analyst` | Medium | Qwen3-30B-A3B | 0.39 | 8.0 | 20% |
| `data-debugger` | Hard | Qwen3-30B-A3B | 0.26 | 8.0 | 0% |

*Success = score > 0.80 (meaningful majority of task milestones solved within 8 steps).*  
*The hard task (data-debugger) requires reasoning about data-quality heuristics that current 30B models lack; it is intended to challenge frontier-scale models.*

---

## Technical Implementation Notes

- **Thread safety:** `QueryEnv.reset()` and `QueryEnv.step()` are both guarded by a `threading.Lock`, making the server safe for concurrent requests without episode state corruption.
- **Determinism:** `random.seed(42)` and `Faker.seed(42)` are called inside `seed()` on every `reset_db()` call, guaranteeing identical data across episodes and submissions.
- **SQL safety:** A whitelist regex (`^\\s*(select|with|explain|pragma\\s+\\w+\\s*(?!\\s*=))`) is applied before any DB I/O. PRAGMA write forms are additionally blocked by a secondary pattern.
- **Score bounds:** `_clamp_score()` applies `max(0.05, min(0.95, value))` plus an explicit `≤ 0.0` / `≥ 1.0` guard at every reward return site, satisfying OpenEnv's strict `(0, 1)` exclusive requirement.
- **In-memory DB:** The SQLite database lives entirely in RAM (`:memory:`), giving sub-millisecond query execution and eliminating disk I/O from the evaluation loop.
