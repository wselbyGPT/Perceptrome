from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import JobEngine, JobEvent, JobResult, JobSpec

__all__ = ["JobEngine", "JobEvent", "JobResult", "JobSpec"]


def __getattr__(name: str):
    if name in __all__:
        from .engine import JobEngine, JobEvent, JobResult, JobSpec

        mapping = {
            "JobEngine": JobEngine,
            "JobEvent": JobEvent,
            "JobResult": JobResult,
            "JobSpec": JobSpec,
        }
        return mapping[name]
    raise AttributeError(name)
