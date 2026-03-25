from __future__ import annotations

import json
from pathlib import Path

from perceptrome.tui.history import HistoryIndexer
from perceptrome.tui.state_store import FailureSummary, JobRecord, StateStore


def test_history_index_reconciles_persisted_and_manifest_artifacts(tmp_path: Path) -> None:
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
    assert "artifacts/model_latest.pt" in paths
    assert merged.failure_summary is not None
    assert merged.failure_summary.traceback_path == "artifacts/traceback.txt"


def test_history_index_includes_manifest_only_runs(tmp_path: Path) -> None:
    store = StateStore(state_root=str(tmp_path / "state" / "tui"))

    run_dir = tmp_path / "runs" / "run_manifest_only"
    run_dir.mkdir(parents=True)
    manifest = {
        "run": {"id": "run_manifest_only", "kind": "train_one"},
        "created_at": "2026-03-24T00:00:00+00:00",
        "completed_at": "2026-03-24T00:05:00+00:00",
        "artifacts": [{"id": "weights", "role": "checkpoint", "path": "artifacts/weights.pt"}],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    rows = HistoryIndexer(store, runs_dir=str(tmp_path / "runs")).merged_jobs()
    manifest_only = next(item for item in rows if item.run_id == "run_manifest_only")

    assert manifest_only.kind == "train_one"
    assert manifest_only.status == "healthy"
    assert manifest_only.manifest_path is not None
    assert manifest_only.artifacts[0]["path"] == "artifacts/weights.pt"
