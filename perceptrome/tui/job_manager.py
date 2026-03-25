"""Threaded TUI job manager wrapping :class:`perceptrome.jobs.engine.JobEngine`."""

from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from perceptrome.jobs.engine import JobEvent, JobResult, JobSpec

from .events import (
    JobArtifactEmittedEvent,
    JobCanceledEvent,
    JobCompletedEvent,
    JobErrorEvent,
    JobFailedEvent,
    JobMetricUpdatedEvent,
    JobStageUpdatedEvent,
    JobStartedEvent,
    JobWarningEvent,
)

EventCallback = Callable[[object], None]


class JobStatus(str, Enum):
    HEALTHY = "healthy"
    BUSY = "busy"
    STALLED = "stalled"
    WARNING = "warning"
    FAILED = "failed"


@dataclass(slots=True)
class JobMetrics:
    latest_loss: float | None = None
    rolling_loss: float | None = None


@dataclass(slots=True)
class Job:
    id: str
    run_id: str
    title: str
    kind: str
    status: JobStatus = JobStatus.BUSY
    message: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    artifacts: list[str] = field(default_factory=list)
    metrics: JobMetrics = field(default_factory=JobMetrics)


@dataclass(slots=True)
class _RuntimeState:
    cancel_event: threading.Event
    thread: threading.Thread
    losses: deque[float] = field(default_factory=lambda: deque(maxlen=20))


class JobManager:
    """Owns JobEngine workers, event fanout, and reconnect state."""

    def __init__(self, *, persist_path: str = "state/tui_jobs.json") -> None:
        self._jobs: dict[str, Job] = {}
        self._runtime: dict[str, _RuntimeState] = {}
        self._subscribers: dict[int, tuple[EventCallback, str | None]] = {}
        self._next_token = 1
        self._lock = threading.RLock()
        self._persist_path = Path(persist_path)

    def submit(self, spec: "JobSpec", *, run_id: str | None = None, title: str | None = None) -> str:
        job_id = str(run_id or f"job-{datetime.now(timezone.utc).timestamp()}")
        card = Job(id=job_id, run_id=job_id, title=title or spec.kind.replace("_", " ").title(), kind=spec.kind)
        cancel_event = threading.Event()

        worker = threading.Thread(
            target=self._run_job,
            kwargs={"job_id": job_id, "spec": spec, "cancel_event": cancel_event},
            name=f"tui-{job_id}",
            daemon=True,
        )

        with self._lock:
            self._jobs[job_id] = card
            self._runtime[job_id] = _RuntimeState(cancel_event=cancel_event, thread=worker)
        self._publish(JobStartedEvent(job_id=job_id, run_id=job_id, title=card.title))
        self._persist_registry()
        worker.start()
        return job_id

    def _run_job(self, *, job_id: str, spec: "JobSpec", cancel_event: threading.Event) -> None:
        from perceptrome.jobs.engine import JobEngine

        engine = JobEngine(event_sink=lambda event: self._bridge_engine_event(job_id, event), cancel_event=cancel_event)
        result = engine.run(spec)
        self._finalize_job(job_id, result)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            runtime = self._runtime.get(job_id)
            if runtime is None:
                return False
            runtime.cancel_event.set()
        self._publish(JobCanceledEvent(job_id=job_id, run_id=job_id, message="Cancellation requested"))
        return True

    def subscribe(self, callback: EventCallback, *, job_id: str | None = None) -> int:
        with self._lock:
            token = self._next_token
            self._next_token += 1
            self._subscribers[token] = (callback, job_id)
            return token

    def unsubscribe(self, token: int) -> None:
        with self._lock:
            self._subscribers.pop(token, None)

    def list_jobs(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.updated_at, reverse=True)

    def reconnect_on_startup(self, *, runs_dir: str = "runs", limit: int = 25) -> None:
        snapshot = self._load_snapshot()
        active = set(str(x) for x in snapshot.get("active", []))

        manifests = sorted(Path(runs_dir).glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for manifest in manifests[:limit]:
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception:
                continue
            run_id = str(payload.get("run_id") or manifest.parent.name)
            job = Job(
                id=run_id,
                run_id=run_id,
                title=str(payload.get("run_kind") or "Recovered job"),
                kind=str(payload.get("run_kind") or "unknown"),
                status=JobStatus.STALLED if run_id in active else JobStatus.HEALTHY,
                message="Recovered from persisted manifest",
            )
            for artifact in payload.get("artifacts", []) or []:
                if isinstance(artifact, dict) and artifact.get("path"):
                    job.artifacts.append(str(artifact["path"]))
            if payload.get("error"):
                job.status = JobStatus.FAILED
            with self._lock:
                self._jobs.setdefault(run_id, job)

    def _bridge_engine_event(self, job_id: str, event: "JobEvent") -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            card = self._jobs.get(job_id)
            runtime = self._runtime.get(job_id)
            if card is None:
                return
            card.updated_at = now
            card.message = event.message

        self._publish(JobStageUpdatedEvent(job_id=job_id, run_id=card.run_id, stage=event.stage, message=event.message, data=event.data))

        if event.stage in {"warn", "warning"}:
            self._set_status(job_id, JobStatus.WARNING)
            self._publish(JobWarningEvent(job_id=job_id, run_id=card.run_id, message=event.message))

        if event.stage == "error":
            self._set_status(job_id, JobStatus.FAILED)
            self._publish(JobErrorEvent(job_id=job_id, run_id=card.run_id, message=event.message, error=str(event.data.get("error") or "")))

        metric_loss = event.data.get("loss")
        if isinstance(metric_loss, (int, float)) and runtime is not None:
            runtime.losses.append(float(metric_loss))
            rolling = sum(runtime.losses) / len(runtime.losses)
            with self._lock:
                card.metrics.latest_loss = float(metric_loss)
                card.metrics.rolling_loss = float(rolling)
            self._publish(
                JobMetricUpdatedEvent(
                    job_id=job_id,
                    run_id=card.run_id,
                    metric_name="loss",
                    latest_value=float(metric_loss),
                    rolling_value=float(rolling),
                    step=int(event.data.get("epoch")) if isinstance(event.data.get("epoch"), int) else None,
                )
            )

        for key in ("path", "output", "manifest_path", "summary_path", "scorecard_path"):
            maybe_path = event.data.get(key)
            if isinstance(maybe_path, str) and maybe_path:
                with self._lock:
                    if maybe_path not in card.artifacts:
                        card.artifacts.append(maybe_path)
                self._publish(JobArtifactEmittedEvent(job_id=job_id, run_id=card.run_id, path=maybe_path, role=key))

        if event.stage in {"start", "encode", "train", "stream_step", "pretrain", "generate", "validate", "design_loop", "manifest"}:
            self._set_status(job_id, JobStatus.BUSY)

    def _finalize_job(self, job_id: str, result: "JobResult") -> None:
        with self._lock:
            card = self._jobs.get(job_id)
            runtime = self._runtime.pop(job_id, None)
            if card is None:
                return
            card.finished_at = datetime.now(timezone.utc)
            card.updated_at = card.finished_at
        if result.ok:
            self._set_status(job_id, JobStatus.HEALTHY)
            self._publish(JobCompletedEvent(job_id=job_id, run_id=card.run_id, message=result.message, data=result.data))
        elif result.exit_code == 130:
            self._set_status(job_id, JobStatus.STALLED)
            self._publish(JobCanceledEvent(job_id=job_id, run_id=card.run_id, message=result.message))
        else:
            self._set_status(job_id, JobStatus.FAILED)
            self._publish(JobFailedEvent(job_id=job_id, run_id=card.run_id, message=result.message, error=result.message))
        self._persist_registry()
        _ = runtime

    def _set_status(self, job_id: str, status: JobStatus) -> None:
        with self._lock:
            card = self._jobs.get(job_id)
            if card is not None:
                card.status = status
                card.updated_at = datetime.now(timezone.utc)

    def _publish(self, event: object) -> None:
        with self._lock:
            subscribers = list(self._subscribers.items())
        for _, (callback, only_job_id) in subscribers:
            if only_job_id is not None and getattr(event, "job_id", None) != only_job_id:
                continue
            callback(event)

    def _persist_registry(self) -> None:
        payload = {
            "active": list(self._runtime.keys()),
            "recent": [job.id for job in self.list_jobs()[:50]],
        }
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _load_snapshot(self) -> dict[str, Any]:
        if not self._persist_path.exists():
            return {"active": [], "recent": []}
        try:
            return json.loads(self._persist_path.read_text(encoding="utf-8"))
        except Exception:
            return {"active": [], "recent": []}
