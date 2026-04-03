"""
Abstract base class for all QueryGym task graders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTask(ABC):
    """Interface all task graders must implement."""

    task_id: str          # overridden in subclasses
    max_steps: int = 8    # overridden in subclasses

    @abstractmethod
    def reset(self) -> None:
        """Called at the start of each episode to reset grader state."""
        ...

    @abstractmethod
    def check_progress(
        self,
        sql: str,
        result: list[dict[str, Any]] | None,
        error: str | None,
    ) -> float:
        """
        Inspect the latest query result and return the incremental reward
        delta earned this step (0.0 if nothing new was discovered).
        """
        ...

    @abstractmethod
    def is_done(self) -> bool:
        """Return True when the task completion criterion has been met."""
        ...

    @abstractmethod
    def get_info(self) -> dict[str, Any]:
        """Return task-specific diagnostic info for the /step response."""
        ...
