from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic, sleep
from types import SimpleNamespace

from perceptrome.tui.events import (
    JobArtifactEmittedEvent,
    JobCanceledEvent,
    JobCompletedEvent,
    JobMetricUpdatedEvent,
    JobRecoveryEvent,
    JobStageUpdatedEvent,
    JobStartedEvent,
    JobWarningEvent,
)
from perceptrome.tui.job_manager import JobManager, JobStatus


@dataclass
class FakeJobEvent:
    stage: str
    message: str
    data: dict[str, object] = field(default_factory=dict)


@dataclass
class FakeJobResult:
    ok: bool
    exit_code: int
    message: str
    data: dict[str, object] = field(default_factory=dict)


class FakeJobEngineSuccess:
    def __init__(self, event_sink, cancel_event) -> None:
        self._event_sink = event_sink
        self._cancel_event = cancel_event

    def run(self, spec):
        del spec
        self._event_sink(FakeJobEvent(stage="start", message="job started"))
        self._event_sink(FakeJobEvent(stage="train", message="step 1", data={"loss": 2.0, "step": 1, "epoch": 1, "total_steps": 2, "log_path": "runs/r1/train.log"}))
        self._event_sink(FakeJobEvent(stage="warn", message="soft warning"))
        self._event_sink(FakeJobEvent(stage="train", message="step 2", data={"loss": 1.0, "step": 2, "epoch": 2, "total_steps": 2, "path": "runs/r1/artifacts/model.pt"}))
        return FakeJobResult(ok=True, exit_code=0, message="done")


class FakeJobEngineCancelable:
    def __init__(self, event_sink, cancel_event) -> None:
        self._event_sink = event_sink
        self._cancel_event = cancel_event
        self._event_sink(FakeJobEvent(stage="start", message="job started"))

    def run(self, spec):
        del spec
        self._cancel_event.wait(timeout=1.0)
        return FakeJobResult(ok=False, exit_code=130, message="canceled")


def _wait_for_status(manager: JobManager, job_id: str, expected: JobStatus, timeout: float = 2.0) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        jobs = {job.id: job for job in manager.list_jobs()}
        if jobs.get(job_id) is not None and jobs[job_id].status == expected:
            return
        sleep(0.01)
    jobs = {job.id: job for job in manager.list_jobs()}
    raise AssertionError(f"job {job_id} did not reach {expected!r}; got {jobs.get(job_id)}")


def test_submit_event_flow_and_status_updates(monkeypatch, tmp_path: Path) -> None:
    fake_engine_module = SimpleNamespace(JobEngine=FakeJobEngineSuccess)
    monkeypatch.setitem(sys.modules, "perceptrome.jobs.engine", fake_engine_module)
    manager = JobManager(persist_path=str(tmp_path / "state" / "tui_jobs.json"))

    seen: list[object] = []
    manager.subscribe(seen.append)

    job_id = manager.submit(SimpleNamespace(kind="train_one"), run_id="run_success", title="Train")
    _wait_for_status(manager, job_id, JobStatus.HEALTHY)

    jobs = manager.list_jobs()
    assert jobs[0].id == "run_success"
    assert jobs[0].status == JobStatus.HEALTHY
    assert "runs/r1/artifacts/model.pt" in jobs[0].artifacts
    assert jobs[0].metrics.latest_loss == 1.0
    assert jobs[0].metrics.rolling_loss == 1.5
    assert jobs[0].current_stage == "train"
    assert jobs[0].progress.step == 2
    assert jobs[0].last_warning == "soft warning"

    assert any(isinstance(event, JobStartedEvent) for event in seen)
    assert any(isinstance(event, JobWarningEvent) for event in seen)
    assert any(isinstance(event, JobMetricUpdatedEvent) for event in seen)
    assert any(isinstance(event, JobArtifactEmittedEvent) for event in seen)
    assert any(isinstance(event, JobStageUpdatedEvent) and event.stage == "progress" for event in seen)
    assert any(isinstance(event, JobStageUpdatedEvent) and event.stage == "log" for event in seen)
    assert any(isinstance(event, JobCompletedEvent) for event in seen)


def test_cancel_flow_marks_stalled_and_emits_cancel_event(monkeypatch, tmp_path: Path) -> None:
    fake_engine_module = SimpleNamespace(JobEngine=FakeJobEngineCancelable)
    monkeypatch.setitem(sys.modules, "perceptrome.jobs.engine", fake_engine_module)
    manager = JobManager(persist_path=str(tmp_path / "state" / "tui_jobs.json"))

    seen: list[object] = []
    manager.subscribe(seen.append)

    job_id = manager.submit(SimpleNamespace(kind="train_one"), run_id="run_cancel", title="Train")
    assert manager.cancel(job_id) is True
    _wait_for_status(manager, job_id, JobStatus.STALLED)

    jobs = manager.list_jobs()
    assert jobs[0].id == "run_cancel"
    assert jobs[0].status == JobStatus.STALLED
    assert any(isinstance(event, JobCanceledEvent) for event in seen)


def test_reconnect_hydrates_artifacts_and_status(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run_dir = runs / "run_123"
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": "run_123",
        "run_kind": "stream",
        "artifacts": [{"path": "outputs/out.fasta"}, {"path": "artifacts/metrics.json"}],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    persist = tmp_path / "state" / "tui_jobs.json"
    persist.parent.mkdir(parents=True)
    persist.write_text(
        json.dumps(
            {
                "active": ["run_123"],
                "recent": ["run_123"],
                "cards": {
                    "run_123": {
                        "id": "run_123",
                        "run_id": "run_123",
                        "title": "stream",
                        "kind": "stream",
                        "status": "busy",
                        "message": "in progress",
                        "created_at": "2026-04-01T00:00:00+00:00",
                        "updated_at": "2026-04-01T00:00:00+00:00",
                        "finished_at": None,
                        "artifacts": ["outputs/out.fasta"],
                        "metrics": {"latest_loss": 1.25, "rolling_loss": 1.4},
                        "current_stage": "train",
                        "progress": {"step": 4, "epoch": 1, "total_steps": 10, "total_epochs": 4, "percent": 0.4},
                        "last_warning": "watch loss",
                        "last_error": "",
                        "status_reason": "",
                        "status_metadata": {},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    jm = JobManager(persist_path=str(persist))
    seen: list[object] = []
    jm.subscribe(seen.append)
    jm.reconnect_on_startup(runs_dir=str(runs))

    jobs = jm.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].id == "run_123"
    assert jobs[0].status == JobStatus.STALLED
    assert jobs[0].status_reason == "recovered_orphaned_active"
    assert jobs[0].current_stage == "train"
    assert jobs[0].progress.percent == 0.4
    assert "outputs/out.fasta" in jobs[0].artifacts
    assert any(isinstance(event, JobRecoveryEvent) for event in seen)
