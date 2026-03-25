from __future__ import annotations

import json
from pathlib import Path

from perceptrome.tui.state_store import FailureSummary, JobRecord, SCHEMA_VERSION, StateStore


def test_state_store_load_save_and_corruption_fallback(tmp_path: Path) -> None:
    root = tmp_path / "state" / "tui"
    root.mkdir(parents=True)
    (root / "session.json").write_text("{broken", encoding="utf-8")
    (root / "jobs.json").write_text("[broken", encoding="utf-8")

    store = StateStore(state_root=str(root))
    assert store.active_view == "overview"
    assert store.list_jobs() == []

    store.set_active_view("jobs")
    store.set_active_focus("job-card", active_job_id="run_ok")
    store.set_drawer_toggle("logs", True)
    store.add_launcher_history("open_panel", panel="jobs")

    ok_job = JobRecord(
        id="run_ok",
        run_id="run_ok",
        kind="train",
        status="healthy",
        title="Healthy train run",
        config={"command": "perceptrome train --config config.yaml"},
        artifacts=[{"role": "checkpoint", "path": str(tmp_path / "runs" / "run_ok" / "artifacts" / "ckpt.pt")}],
    )
    failed_job = JobRecord(
        id="run_bad",
        run_id="run_bad",
        kind="train",
        status="failed",
        failure_summary=FailureSummary(
            stage="train",
            latest_warning_or_error="OOM",
            traceback_path="runs/run_bad/artifacts/traceback.txt",
            suggested_next_action="lower_batch_size",
        ),
    )
    store.upsert_job(ok_job)
    store.upsert_job(failed_job)

    reloaded = StateStore(state_root=str(root))
    session = reloaded.get_session()
    assert session.last_panel == "jobs"
    assert session.active_job_id == "run_ok"
    assert session.drawer_toggles["logs"] is True

    assert reloaded.open_active_job() is not None
    assert reloaded.open_active_job().id == "run_ok"
    assert reloaded.open_last_failed_job() is not None
    assert reloaded.open_last_failed_job().id == "run_bad"
    assert reloaded.rerun_last_job() is not None
    assert reloaded.launcher_history(limit=1)


def test_state_store_schema_migration_falls_back_to_defaults(tmp_path: Path) -> None:
    root = tmp_path / "state" / "tui"
    root.mkdir(parents=True)

    legacy_session = {
        "schema_version": SCHEMA_VERSION - 1,
        "last_panel": "train",
        "drawer_toggles": {"diagnostics": True},
        "active_job_id": "old_job",
    }
    legacy_jobs = {
        "schema_version": SCHEMA_VERSION - 1,
        "jobs": [{"id": "legacy", "run_id": "legacy", "kind": "train", "status": "healthy"}],
    }
    (root / "session.json").write_text(json.dumps(legacy_session), encoding="utf-8")
    (root / "jobs.json").write_text(json.dumps(legacy_jobs), encoding="utf-8")

    store = StateStore(state_root=str(root))

    # Unsupported schema should gracefully reset in-memory state.
    assert store.active_view == "overview"
    assert store.list_jobs() == []

    # Any save operation should rewrite with current schema.
    store.set_active_view("train")
    session_payload = json.loads((root / "session.json").read_text(encoding="utf-8"))
    assert session_payload["schema_version"] == SCHEMA_VERSION
    assert session_payload["last_panel"] == "train"
