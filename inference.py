"""
QueryGym inference script.
Runs the agent (OpenAI-compatible LLM) against all 3 tasks sequentially.

Required environment variables:
  API_BASE_URL  — e.g. https://router.huggingface.co/v1
  MODEL_NAME    — e.g. Qwen/Qwen3-30B-A3B:novita
  HF_TOKEN      — Hugging Face token for the router

Optional:
  GYM_BASE_URL  — defaults to http://localhost:7860

STDOUT FORMAT (mandatory per OpenEnv spec):
  [START] task=<task_name> env=querygym model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after the episode ends, ALWAYS emitted (even on exception).
    - reward and rewards formatted to 2 decimal places.
    - score formatted to 4 decimal places; strictly in (0.0, 1.0) exclusive.
    - done and success are lowercase booleans: true or false.
    - error is the raw error string, or null if none.
    - All fields on a single line with no newlines within a line.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Load .env (local dev convenience)
# ---------------------------------------------------------------------------
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
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen3-30B-A3B:novita")
HF_TOKEN     = os.environ.get("HF_TOKEN")
GYM_BASE_URL = os.environ.get("GYM_BASE_URL", "http://localhost:7860")

MAX_STEPS   = 8
TEMPERATURE = 0.2
MAX_TOKENS  = 300

TASKS = ["schema-explorer", "kpi-analyst", "data-debugger"]

# Bounds that env.py guarantees via _clamp_score():
#   reward from /step is always in [_SCORE_MIN, _SCORE_MAX] = [0.05, 0.95]
# We use the FINAL cumulative reward as the task score, so it is always
# strictly in (0, 1) without any further normalisation needed.
_SCORE_MIN = 0.05   # must match env.py _SCORE_MIN
_SCORE_MAX = 0.95   # must match env.py _SCORE_MAX

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


# ---------------------------------------------------------------------------
# Logging helpers (strict OpenEnv format)
# ---------------------------------------------------------------------------

def _log_start(task_id: str) -> None:
    print(f"[START] task={task_id} env=querygym model={MODEL_NAME}", flush=True)


def _log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error else "null"
    done_val  = "true" if done else "false"
    # Inline any newlines in action so the line stays single-line
    safe_action = action.replace("\n", " ").replace("\r", " ")
    print(
        f"[STEP] step={step} action={json.dumps(safe_action)} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def _log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = "true" if success else "false"
    print(
        f"[END] success={success_val} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Reward / score helpers
# ---------------------------------------------------------------------------

def _safe_reward(raw: object) -> float:
    """
    Convert the raw reward value from /step into a float.
    The environment guarantees it is in [_SCORE_MIN, _SCORE_MAX], but we
    add a belt-and-suspenders clamp here too so inference is never the
    source of an out-of-range score.
    """
    try:
        val = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        val = _SCORE_MIN
    return max(_SCORE_MIN, min(_SCORE_MAX, val))


def _compute_score(rewards: list[float]) -> float:
    """
    Task score = the final (cumulative) reward value for this episode.

    Because env.py uses _clamp_score() everywhere, the last reward is
    always in [_SCORE_MIN, _SCORE_MAX] = [0.05, 0.95], which is
    strictly within (0.0, 1.0) as required by OpenEnv Phase 2.

    Fall back to _SCORE_MIN if rewards is somehow empty.
    """
    if not rewards:
        return _SCORE_MIN
    score = rewards[-1]   # cumulative total_reward from last step
    # Hard guarantee: strictly inside (0, 1)
    score = max(_SCORE_MIN, min(_SCORE_MAX, score))
    if score <= 0.0:
        score = _SCORE_MIN
    elif score >= 1.0:
        score = _SCORE_MAX
    return round(score, 4)


# ---------------------------------------------------------------------------
# SQL extraction helper
# ---------------------------------------------------------------------------

def _extract_sql(text: str) -> str:
    """Strip markdown fences if the model ignores the instruction."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = [ln for ln in lines if not ln.startswith("```")]
        text = "\n".join(inner).strip()
    return text or "SELECT 1"


# ---------------------------------------------------------------------------
# Agent episode loop — one task at a time
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_id: str) -> None:
    _log_start(task_id)

    rewards: list[float] = []
    step_num = 0
    score    = _SCORE_MIN
    success  = False

    try:
        # -- Reset --
        reset_resp  = _post("/reset", {"task_id": task_id})
        schema_hint: str = reset_resp.get("schema_hint", "")

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

        done = False

        for step_num in range(1, MAX_STEPS + 1):
            # -- LLM call --
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                sql = _extract_sql(completion.choices[0].message.content or "")
            except Exception as llm_exc:
                print(f"[DEBUG] LLM error on step {step_num}: {llm_exc}", file=sys.stderr, flush=True)
                sql = "SELECT 1"

            # -- Env step --
            try:
                step_resp = _post("/step", {"sql": sql})
                obs    = step_resp.get("observation", {})
                reward = _safe_reward(step_resp.get("reward", _SCORE_MIN))
                done   = bool(step_resp.get("done", False))
                error  = obs.get("error") or None
                result = obs.get("result")
            except Exception as env_exc:
                print(f"[DEBUG] Env step error on step {step_num}: {env_exc}", file=sys.stderr, flush=True)
                reward = _SCORE_MIN
                done   = True
                error  = str(env_exc)
                result = None

            rewards.append(reward)
            _log_step(step=step_num, action=sql, reward=reward, done=done, error=error)

            messages.append({"role": "assistant", "content": sql})

            if done:
                break

            # Build feedback for next LLM turn
            result_preview = json.dumps(result[:3] if result else [])
            next_user = (
                f"Query result (first 3 rows): {result_preview}\n"
                + (f"Error: {error}\n" if error else "")
                + "Write your next SQL query to continue making progress."
            )
            messages.append({"role": "user", "content": next_user})

        # -- Compute final score --
        score   = _compute_score(rewards)
        # "success" if agent achieved meaningful progress beyond the floor
        success = score > (_SCORE_MIN + 0.01)

    except Exception as outer_exc:
        print(f"[DEBUG] Unhandled error in run_task({task_id}): {outer_exc}", file=sys.stderr, flush=True)
        score = _SCORE_MIN

    finally:
        _log_end(success=success, steps=step_num, score=score, rewards=rewards)


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
