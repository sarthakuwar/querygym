"""
QueryGym inference script.
Runs the agent (OpenAI-compatible LLM) against all 3 tasks sequentially.

Required environment variables:
  API_BASE_URL  — e.g. https://router.huggingface.co/v1
  MODEL_NAME    — e.g. Qwen/Qwen3-VL-30B-A3B-Instruct:novita
  HF_TOKEN      — Hugging Face token for the router

Optional:
  GYM_BASE_URL  — defaults to http://localhost:7860

Stdout format:
  [START] task=<task_name> env=querygym model=<model_name>
  [STEP] step=<n> action=<sql> reward=<0.00> done=<true|false> error=<msg|null>
  [END] success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
from openai import OpenAI

# Load .env if present (local dev convenience)
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-30B-A3B:novita")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
GYM_BASE_URL = os.environ.get("GYM_BASE_URL", "http://localhost:7860")

MAX_STEPS = 8
TEMPERATURE = 0.2
MAX_TOKENS = 300

TASKS = ["schema-explorer", "kpi-analyst", "data-debugger"]

SYSTEM_PROMPT = (
    "You are a SQL agent interacting with a SaaS billing SQLite database. "
    "Tables available: plans, customers, subscriptions, invoices, events.\n"
    "Your job is to write a single valid SQLite SELECT statement per turn that helps you "
    "make progress on the current task. Respond with ONLY the SQL query — no explanation, "
    "no markdown fences, just pure SQL."
)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post(path: str, payload: dict) -> dict:
    url = f"{GYM_BASE_URL}{path}"
    resp = httpx.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _get(path: str) -> dict:
    url = f"{GYM_BASE_URL}{path}"
    resp = httpx.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def _extract_sql(text: str) -> str:
    """Strip markdown fences if the model ignores the instruction."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first and last fence lines
        inner = [l for l in lines if not l.startswith("```")]
        text = "\n".join(inner).strip()
    return text


def run_task(client: OpenAI, task_id: str) -> None:
    # Reset environment
    reset_resp = _post("/reset", {"task_id": task_id})

    schema_hint: str = reset_resp.get("schema_hint", "")
    print(f"[START] task={task_id} env=querygym model={MODEL_NAME}", flush=True)

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Task: {task_id}\n"
                f"Available tables: {schema_hint}\n"
                "Write your first SQL query."
            ),
        },
    ]

    rewards: list[float] = []
    step_num = 0
    done = False

    for step_num in range(1, MAX_STEPS + 1):
        # Call LLM
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        sql = _extract_sql(completion.choices[0].message.content or "")

        # Send action to gym
        step_resp = _post("/step", {"sql": sql})
        obs = step_resp.get("observation", {})
        reward = float(step_resp.get("reward", 0.0))
        done = bool(step_resp.get("done", False))
        error = obs.get("error")
        result = obs.get("result")

        rewards.append(reward)

        error_str = error if error else "null"
        print(
            f"[STEP] step={step_num} action={json.dumps(sql)} "
            f"reward={reward:.2f} done={'true' if done else 'false'} error={error_str}",
            flush=True,
        )

        # Build assistant + next user message for context
        messages.append({"role": "assistant", "content": sql})

        if done:
            break

        # Build feedback for next turn
        result_preview = json.dumps(result[:3] if result else [])
        next_user = (
            f"Query result (first 3 rows): {result_preview}\n"
            + (f"Error: {error}\n" if error else "")
            + "Write your next SQL query to continue making progress."
        )
        messages.append({"role": "user", "content": next_user})

    success = done
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={'true' if success else 'false'} steps={step_num} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN not set — requests may be rate-limited.", file=sys.stderr)

    client = OpenAI(
        api_key=HF_TOKEN or "dummy",
        base_url=API_BASE_URL,
    )

    for task_id in TASKS:
        run_task(client, task_id)


if __name__ == "__main__":
    main()
