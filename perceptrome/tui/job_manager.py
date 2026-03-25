"""Small job registry for TUI-visible jobs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class Job:
    id: str
    title: str
    status: JobStatus = JobStatus.QUEUED


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def add(self, job: Job) -> None:
        self._jobs[job.id] = job

    def set_status(self, job_id: str, status: JobStatus) -> None:
        if job_id in self._jobs:
            self._jobs[job_id].status = status

    def list_jobs(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda job: job.id)
