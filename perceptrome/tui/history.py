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


@dataclass(slots=True)
class ArtifactRow:
    run_id: str
    run_kind: str
    role: str
    path: str
    exists: bool
    created_at: str
    source: str
    manifest_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ArtifactInspection:
    path: str
    exists: bool
    mtime: str
    run_id: str
    run_kind: str
    role: str
    manifest_path: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


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
                artifacts = _manifest_artifacts(manifest, manifest_path=manifest_path)
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

    def artifacts_grouped(self, *, limit_runs: int = 100) -> dict[str, dict[str, list[ArtifactRow]]]:
        grouped: dict[str, dict[str, list[ArtifactRow]]] = {}
        for row in self.merged_jobs(limit=limit_runs):
            by_role: dict[str, list[ArtifactRow]] = {}
            for item in row.artifacts:
                if not isinstance(item, dict):
                    continue
                path = str(item.get("path") or "")
                if not path:
                    continue
                resolved = _resolve_artifact_path(path, run_id=row.run_id, runs_dir=self._runs_dir, manifest_path=row.manifest_path)
                role = str(item.get("role") or "artifact")
                by_role.setdefault(role, []).append(
                    ArtifactRow(
                        run_id=row.run_id,
                        run_kind=row.kind,
                        role=role,
                        path=str(resolved),
                        exists=resolved.exists(),
                        created_at=str(item.get("created_at") or row.updated_at or row.created_at),
                        source=str(item.get("source") or "history"),
                        manifest_path=row.manifest_path,
                        metadata=dict(item.get("metadata") or {}),
                    )
                )
            if by_role:
                grouped[row.run_id] = by_role
        return grouped

    def inspect_checkpoint(self, artifact_path: str | None = None) -> ArtifactInspection | None:
        grouped = self.artifacts_grouped(limit_runs=150)
        selected = str(artifact_path or "").strip() or self._state_store.get_session().selected_artifact_path or ""
        candidate: ArtifactRow | None = None
        for _run_id, by_role in grouped.items():
            checkpoints = by_role.get("checkpoint", [])
            for item in checkpoints:
                if selected and item.path == selected:
                    candidate = item
                    break
            if candidate is not None:
                break
        if candidate is None:
            latest: tuple[float, ArtifactRow] | None = None
            for _run_id, by_role in grouped.items():
                for item in by_role.get("checkpoint", []):
                    mtime = Path(item.path).stat().st_mtime if Path(item.path).exists() else -1.0
                    if latest is None or mtime > latest[0]:
                        latest = (mtime, item)
            if latest is None:
                return None
            candidate = latest[1]

        payload = _load_manifest(Path(candidate.manifest_path)) if candidate.manifest_path else None
        manifest_metadata = _manifest_metadata(payload) if payload else {}
        mtime_value = "missing"
        target_path = Path(candidate.path)
        if target_path.exists():
            mtime_value = _fmt_mtime(target_path.stat().st_mtime)
        return ArtifactInspection(
            path=candidate.path,
            exists=target_path.exists(),
            mtime=mtime_value,
            run_id=candidate.run_id,
            run_kind=candidate.run_kind,
            role=candidate.role,
            manifest_path=candidate.manifest_path,
            metadata=manifest_metadata,
        )

    def resolve_log_path(self, *, run_id: str | None = None) -> str | None:
        for row in self.merged_jobs(limit=100):
            if run_id and row.run_id != run_id:
                continue
            path = self._first_existing_log_path(row)
            if path:
                return path
        return None

    def resolve_traceback_path(self, *, run_id: str | None = None) -> str | None:
        for row in self.merged_jobs(limit=100):
            if run_id and row.run_id != run_id:
                continue
            if row.failure_summary and row.failure_summary.traceback_path:
                resolved = _resolve_artifact_path(
                    row.failure_summary.traceback_path,
                    run_id=row.run_id,
                    runs_dir=self._runs_dir,
                    manifest_path=row.manifest_path,
                )
                return str(resolved)
            for item in row.artifacts:
                if not isinstance(item, dict):
                    continue
                path = str(item.get("path") or "")
                if "traceback" in path.lower():
                    return path
        return None

    def resolve_manifest_path(self, *, run_id: str | None = None) -> str | None:
        for row in self.merged_jobs(limit=150):
            if run_id and row.run_id != run_id:
                continue
            if row.manifest_path:
                return row.manifest_path
        return None

    def resolve_config_snapshot_path(self, *, run_id: str | None = None) -> str | None:
        for row in self.merged_jobs(limit=150):
            if run_id and row.run_id != run_id:
                continue
            for item in row.artifacts:
                role = str(item.get("role") or "")
                if "config_snapshot" in role or str(item.get("path") or "").endswith("resolved_config.json"):
                    return str(item.get("path") or "")
        return None

    def latest_artifact_path(self, *, run_id: str | None = None) -> str | None:
        for row in self.merged_jobs(limit=150):
            if run_id and row.run_id != run_id:
                continue
            paths = [str(item.get("path") or "") for item in row.artifacts if isinstance(item, dict) and item.get("path")]
            if paths:
                return paths[-1]
        return None

    def _first_existing_log_path(self, row: IndexedJob) -> str | None:
        preferred = ("log", "stdout", "stderr")
        for item in row.artifacts:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").lower()
            path = str(item.get("path") or "")
            if not path:
                continue
            if role in preferred or any(token in path.lower() for token in preferred):
                return path
        if row.manifest_path:
            candidate = Path(row.manifest_path).with_name("run.log")
            if candidate.exists():
                return str(candidate)
        return None


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


def _manifest_artifacts(payload: dict[str, Any], *, manifest_path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    rows = payload.get("artifacts")
    if isinstance(rows, list):
        out.extend(dict(row) for row in rows if isinstance(row, dict))

    checkpoints = payload.get("checkpoints")
    if isinstance(checkpoints, dict):
        for key, value in checkpoints.items():
            if isinstance(value, str):
                out.append({"id": f"checkpoint:{key}", "role": "checkpoint", "path": value})

    run_id = _manifest_run_id(payload, default=manifest_path.parent.name)
    for artifact in out:
        path = str(artifact.get("path") or "")
        if not path:
            continue
        resolved = _resolve_artifact_path(path, run_id=run_id, runs_dir=manifest_path.parent.parent, manifest_path=str(manifest_path))
        artifact["path"] = str(resolved)
        artifact.setdefault("source", "manifest")
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


def _resolve_artifact_path(path: str, *, run_id: str, runs_dir: Path, manifest_path: str | None) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    run_root = runs_dir / run_id
    if candidate.is_absolute():
        return candidate
    by_run = run_root / candidate
    if by_run.exists():
        return by_run
    artifacts_variant = run_root / "artifacts" / candidate.name
    if artifacts_variant.exists():
        return artifacts_variant
    if manifest_path:
        manifest_parent = Path(manifest_path).parent
        beside_manifest = manifest_parent / candidate
        if beside_manifest.exists():
            return beside_manifest
    return by_run


def _fmt_mtime(raw: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(raw, tz=timezone.utc).replace(microsecond=0).isoformat()


def _manifest_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    run = payload.get("run") if isinstance(payload.get("run"), dict) else {}
    provenance = payload.get("provenance_metadata") if isinstance(payload.get("provenance_metadata"), dict) else {}
    config = provenance.get("config") if isinstance(provenance.get("config"), dict) else {}
    software = provenance.get("software") if isinstance(provenance.get("software"), dict) else {}
    return {
        "run_created_at": run.get("created_at") or payload.get("created_at"),
        "config_path": config.get("path"),
        "config_sha": config.get("sha256"),
        "git_sha": software.get("git_sha"),
    }


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
