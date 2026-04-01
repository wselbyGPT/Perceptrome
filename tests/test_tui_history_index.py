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
    assert any(path.endswith("/runs/run_123/artifacts/model.pt") for path in paths)
    assert any(path.endswith("/runs/run_123/artifacts/model_latest.pt") for path in paths)
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
    assert manifest_only.artifacts[0]["path"].endswith("/runs/run_manifest_only/artifacts/weights.pt")


def test_history_index_artifacts_group_and_checkpoint_inspector(tmp_path: Path) -> None:
    store = StateStore(state_root=str(tmp_path / "state" / "tui"))
    run_dir = tmp_path / "runs" / "run_x"
    art_dir = run_dir / "artifacts"
    art_dir.mkdir(parents=True)
    ckpt = art_dir / "model_latest.pt"
    ckpt.write_text("weights", encoding="utf-8")

    manifest = {
        "run": {"id": "run_x", "kind": "train_one", "created_at": "2026-03-31T00:00:00+00:00"},
        "artifacts": [{"id": "legacy", "role": "checkpoint", "path": "model_latest.pt"}],
        "provenance_metadata": {
            "software": {"git_sha": "abc123"},
            "config": {"path": "config/train.yaml", "sha256": "deadbeef"},
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    indexer = HistoryIndexer(store, runs_dir=str(tmp_path / "runs"))
    grouped = indexer.artifacts_grouped()
    assert "run_x" in grouped
    checkpoint_rows = grouped["run_x"]["checkpoint"]
    assert checkpoint_rows[0].exists
    assert checkpoint_rows[0].path.endswith("runs/run_x/artifacts/model_latest.pt")

    inspection = indexer.inspect_checkpoint(checkpoint_rows[0].path)
    assert inspection is not None
    assert inspection.exists
    assert inspection.run_kind == "train_one"
    assert inspection.metadata["git_sha"] == "abc123"


def test_history_index_resolves_log_and_traceback_paths(tmp_path: Path) -> None:
    store = StateStore(state_root=str(tmp_path / "state" / "tui"))
    run_dir = tmp_path / "runs" / "run_logs"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)
    log_path = artifacts / "stdout.log"
    log_path.write_text("hello\nworld\n", encoding="utf-8")
    tb_path = artifacts / "traceback.txt"
    tb_path.write_text("Traceback...", encoding="utf-8")

    manifest = {
        "run": {"id": "run_logs", "kind": "train"},
        "artifacts": [
            {"role": "log", "path": "artifacts/stdout.log"},
            {"role": "traceback", "path": "artifacts/traceback.txt"},
        ],
        "error": "boom",
        "traceback_path": "artifacts/traceback.txt",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    indexer = HistoryIndexer(store, runs_dir=str(tmp_path / "runs"))
    assert indexer.resolve_log_path() is not None
    assert indexer.resolve_log_path().endswith("runs/run_logs/artifacts/stdout.log")
    assert indexer.resolve_traceback_path() is not None
    assert indexer.resolve_traceback_path().endswith("runs/run_logs/artifacts/traceback.txt")
