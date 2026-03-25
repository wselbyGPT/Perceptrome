"""History utilities for TUI events and run/job indexing."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .events import TUIEvent
from .state_store import FailureSummary, JobRecord, StateStore


class HistoryLog:
    """Tracks a bounded rolling history of events for display in the UI."""

    def __init__(self, max_entries: int = 200) -> None:
        self._events: deque[TUIEvent] = deque(maxlen=max_entries)

    def add(self, event: TUIEvent) -> None:
        self._events.append(event)

    def latest(self, limit: int = 25) -> list[TUIEvent]:
        return list(self._events)[-limit:]

    def all(self) -> Iterable[TUIEvent]:
        return tuple(self._events)


@dataclass(slots=True)
class IndexedJob:
    id: str
    run_id: str
    kind: str
    status: str
    title: str = ""
    created_at: str = ""
    updated_at: str = ""
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    failure_summary: FailureSummary | None = None
    manifest_path: str | None = None


class HistoryIndexer:
    """Merges persisted TUI job records with run manifests/artifact metadata."""

    def __init__(self, state_store: StateStore, *, runs_dir: str = "runs") -> None:
        self._state_store = state_store
        self._runs_dir = Path(runs_dir)

    def merged_jobs(self, *, limit: int = 200) -> list[IndexedJob]:
        merged: dict[str, IndexedJob] = {}

        for persisted in self._state_store.list_jobs():
            merged[persisted.run_id] = IndexedJob(
                id=persisted.id,
                run_id=persisted.run_id,
                kind=persisted.kind,
                status=persisted.status,
                title=persisted.title,
                created_at=persisted.created_at,
                updated_at=persisted.updated_at,
                artifacts=[dict(item) for item in persisted.artifacts],
                config=dict(persisted.config),
                failure_summary=persisted.failure_summary,
            )

        if self._runs_dir.exists():
            manifests = sorted(self._runs_dir.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            for manifest_path in manifests[:limit]:
                manifest = _load_manifest(manifest_path)
                if manifest is None:
                    continue
                run_id = _manifest_run_id(manifest, default=manifest_path.parent.name)
                run_kind = _manifest_run_kind(manifest)
                status = _manifest_status(manifest)
                artifacts = _manifest_artifacts(manifest)
                indexed = merged.get(run_id)
                manifest_failure = _manifest_failure_summary(manifest)
                failure = indexed.failure_summary if indexed else None
                failure = _merge_failure_summary(failure, manifest_failure)

                if indexed is None:
                    merged[run_id] = IndexedJob(
                        id=run_id,
                        run_id=run_id,
                        kind=run_kind,
                        status=status,
                        title=run_kind.replace("_", " ").title(),
                        created_at=str(manifest.get("created_at") or ""),
                        updated_at=str(manifest.get("updated_at") or manifest.get("created_at") or ""),
                        artifacts=artifacts,
                        failure_summary=failure,
                        manifest_path=str(manifest_path),
                    )
                    continue

                indexed.kind = indexed.kind or run_kind
                if indexed.status in {"unknown", ""}:
                    indexed.status = status
                indexed.artifacts = _merge_artifacts(indexed.artifacts, artifacts)
                indexed.failure_summary = failure
                indexed.manifest_path = str(manifest_path)

        rows = sorted(merged.values(), key=lambda row: row.updated_at or row.created_at, reverse=True)
        return rows[:limit]


def _merge_artifacts(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in (left, right):
        for item in source:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or "")
            marker = str(item.get("id") or path)
            if not marker or marker in seen:
                continue
            seen.add(marker)
            merged.append(dict(item))
    return merged


def _load_manifest(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _manifest_run_id(payload: dict[str, Any], *, default: str) -> str:
    run = payload.get("run")
    if isinstance(run, dict) and run.get("id"):
        return str(run["id"])
    return str(payload.get("run_id") or default)


def _manifest_run_kind(payload: dict[str, Any]) -> str:
    run = payload.get("run")
    if isinstance(run, dict) and run.get("kind"):
        return str(run["kind"])
    return str(payload.get("run_kind") or "unknown")


def _manifest_artifacts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    rows = payload.get("artifacts")
    if isinstance(rows, list):
        out.extend(dict(row) for row in rows if isinstance(row, dict))

    checkpoints = payload.get("checkpoints")
    if isinstance(checkpoints, dict):
        for key, value in checkpoints.items():
            if isinstance(value, str):
                out.append({"id": f"checkpoint:{key}", "role": "checkpoint", "path": value})
    return out


def _manifest_status(payload: dict[str, Any]) -> str:
    if payload.get("error"):
        return "failed"
    if payload.get("completed_at") or payload.get("finished_at"):
        return "healthy"
    return "unknown"


def _manifest_failure_summary(payload: dict[str, Any]) -> FailureSummary | None:
    error = payload.get("error")
    if not error:
        return None
    text = str(error)
    traceback_path = None
    for candidate in (payload.get("traceback_path"), (payload.get("paths") or {}).get("traceback")):
        if isinstance(candidate, str) and candidate:
            traceback_path = candidate
            break
    return FailureSummary(
        stage=str(payload.get("stage") or "error"),
        latest_warning_or_error=text,
        traceback_path=traceback_path,
        suggested_next_action="open_traceback" if traceback_path else "rerun_with_diagnostics",
    )


def _merge_failure_summary(existing: FailureSummary | None, incoming: FailureSummary | None) -> FailureSummary | None:
    if existing is None:
        return incoming
    if incoming is None:
        return existing
    return FailureSummary(
        stage=existing.stage or incoming.stage,
        latest_warning_or_error=existing.latest_warning_or_error or incoming.latest_warning_or_error,
        traceback_path=existing.traceback_path or incoming.traceback_path,
        suggested_next_action=existing.suggested_next_action or incoming.suggested_next_action,
    )


def to_job_record(indexed: IndexedJob) -> JobRecord:
    return JobRecord(
        id=indexed.id,
        run_id=indexed.run_id,
        kind=indexed.kind,
        status=indexed.status,
        title=indexed.title,
        artifacts=[dict(item) for item in indexed.artifacts],
        config=dict(indexed.config),
        created_at=indexed.created_at,
        updated_at=indexed.updated_at,
        failure_summary=indexed.failure_summary,
    )
