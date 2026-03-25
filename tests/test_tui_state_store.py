from __future__ import annotations

import json
from pathlib import Path

from perceptrome.tui.history import HistoryIndexer
from perceptrome.tui.state_store import FailureSummary, JobRecord, StateStore


def test_state_store_corruption_fallback_and_helpers(tmp_path: Path) -> None:
    root = tmp_path / "state" / "tui"
    root.mkdir(parents=True)
    (root / "session.json").write_text("{broken", encoding="utf-8")

    store = StateStore(state_root=str(root))
    assert store.active_view == "overview"

    store.set_active_view("jobs")
    store.set_active_focus("job-card", active_job_id="run_ok")
    store.add_launcher_history("open_panel", panel="jobs")

    ok_job = JobRecord(
        id="run_ok",
        run_id="run_ok",
        kind="train",
        status="healthy",
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

    assert store.open_active_job() is not None
    assert store.open_active_job().id == "run_ok"
    assert store.open_last_failed_job() is not None
    assert store.open_last_failed_job().id == "run_bad"
    assert store.rerun_last_job() is not None

    assert store.launcher_history(limit=1)


def test_history_indexer_merges_persisted_jobs_and_manifests(tmp_path: Path) -> None:
    store = StateStore(state_root=str(tmp_path / "state" / "tui"))
    store.upsert_job(
        JobRecord(
            id="run_123",
            run_id="run_123",
            kind="stream",
            status="failed",
            title="Persisted stream",
            artifacts=[{"id": "persisted", "role": "output", "path": "outputs/persisted.fasta"}],
            failure_summary=FailureSummary(stage="validate", latest_warning_or_error="score below threshold"),
        )
    )

    run_dir = tmp_path / "runs" / "run_123"
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": "run_123",
        "run_kind": "stream",
        "updated_at": "2026-03-25T00:00:00+00:00",
        "artifacts": [{"id": "manifest", "role": "checkpoint", "path": "artifacts/model.pt"}],
        "checkpoints": {"latest": "artifacts/model_latest.pt"},
        "error": "validation failed",
        "traceback_path": "artifacts/traceback.txt",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    indexer = HistoryIndexer(store, runs_dir=str(tmp_path / "runs"))
    rows = indexer.merged_jobs()

    assert rows
    merged = rows[0]
    assert merged.run_id == "run_123"
    paths = {str(item.get("path")) for item in merged.artifacts}
    assert "outputs/persisted.fasta" in paths
    assert "artifacts/model.pt" in paths
    assert merged.failure_summary is not None
    assert merged.failure_summary.traceback_path == "artifacts/traceback.txt"
