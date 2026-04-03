"""
QueryGym FastAPI application.
Exposes: POST /reset, POST /step, GET /state, GET /healthz
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from db.connection import init_db
from env import QueryEnv
from models import QueryAction, QueryObservation, QueryEnvState, ResetRequest


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="QueryGym",
    description="OpenEnv-compliant gym for SQL agents on a SaaS billing database.",
    version="1.0.0",
    lifespan=lifespan,
)

# Single shared environment instance (single-session, guarded by Lock inside QueryEnv)
_env = QueryEnv()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/healthz", tags=["meta"])
def healthz() -> dict[str, str]:
    """Health check — returns 200 OK so HF Spaces cold-start detection works."""
    return {"status": "ok"}


@app.post("/reset", response_model=QueryObservation, tags=["openenv"])
def reset(body: ResetRequest = ResetRequest()) -> QueryObservation:
    """
    Start a new episode.
    Body is optional; task_id defaults to 'schema-explorer'.
    """
    try:
        obs = _env.reset(task_id=body.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return obs


@app.post("/step", tags=["openenv"])
def step(action: QueryAction) -> dict[str, Any]:
    """
    Submit one SQL action.
    Returns observation + reward + done + info.
    """
    try:
        obs, reward, done, info = _env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state", response_model=QueryEnvState, tags=["openenv"])
def state() -> QueryEnvState:
    """Return the full current episode state."""
    return _env.get_state()
