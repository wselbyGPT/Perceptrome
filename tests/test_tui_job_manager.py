from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from perceptrome.tui.events import JobMetricUpdatedEvent
from perceptrome.tui.job_manager import Job, JobManager, JobStatus, _RuntimeState


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


def test_rolling_loss_metrics_emitted() -> None:
    jm = JobManager(persist_path="state/test_tui_jobs.json")
    now = datetime.now(timezone.utc)
    jm._jobs["r1"] = Job(id="r1", run_id="r1", title="Train", kind="train", created_at=now, updated_at=now)
    jm._runtime["r1"] = _RuntimeState(cancel_event=threading.Event(), thread=threading.Thread())

    seen: list[object] = []
    jm.subscribe(seen.append)

    jm._bridge_engine_event("r1", FakeJobEvent(stage="train", message="step", data={"loss": 1.0, "epoch": 1}))
    jm._bridge_engine_event("r1", FakeJobEvent(stage="train", message="step", data={"loss": 3.0, "epoch": 2}))

    metrics = [event for event in seen if isinstance(event, JobMetricUpdatedEvent)]
    assert metrics
    assert metrics[-1].latest_value == 3.0
    assert metrics[-1].rolling_value == 2.0
    assert jm._jobs["r1"].metrics.rolling_loss == 2.0


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
    persist.write_text(json.dumps({"active": ["run_123"], "recent": ["run_123"]}), encoding="utf-8")

    jm = JobManager(persist_path=str(persist))
    jm.reconnect_on_startup(runs_dir=str(runs))

    jobs = jm.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].id == "run_123"
    assert jobs[0].status == JobStatus.STALLED
    assert "outputs/out.fasta" in jobs[0].artifacts


def test_finalize_failed_marks_status() -> None:
    jm = JobManager(persist_path="state/test_tui_jobs.json")
    jm._jobs["r2"] = Job(id="r2", run_id="r2", title="Train", kind="train")
    jm._runtime["r2"] = _RuntimeState(cancel_event=threading.Event(), thread=threading.Thread())
    jm._finalize_job("r2", FakeJobResult(ok=False, exit_code=1, message="boom"))
    assert jm._jobs["r2"].status == JobStatus.FAILED
