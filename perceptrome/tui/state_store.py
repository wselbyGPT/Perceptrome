"""Persistent on-disk state store for the Perceptrome TUI."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .events import (
    JobArtifactEmittedEvent,
    JobCanceledEvent,
    JobCompletedEvent,
    JobErrorEvent,
    JobEventBase,
    JobFailedEvent,
    JobRecoveryEvent,
    JobStageUpdatedEvent,
    JobStartedEvent,
    JobWarningEvent,
    TUIEvent,
)

SCHEMA_VERSION = 3


@dataclass(slots=True)
class FailureSummary:
    """Compact error summary persisted for failed jobs."""

    stage: str = ""
    latest_warning_or_error: str = ""
    traceback_path: str | None = None
    suggested_next_action: str = "inspect_logs"


@dataclass(slots=True)
class JobRecord:
    """Persisted TUI-centric metadata for a job/run."""

    id: str
    run_id: str
    kind: str
    status: str
    title: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: _utc_now())
    updated_at: str = field(default_factory=lambda: _utc_now())
    finished_at: str | None = None
    failure_summary: FailureSummary | None = None


@dataclass(slots=True)
class SessionState:
    """UI session metadata."""

    last_panel: str = "overview"
    drawer_toggles: dict[str, bool] = field(default_factory=dict)
    active_focus: str | None = None
    active_job_id: str | None = None
    selected_job_id: str | None = None
    selected_artifact_path: str | None = None
    active_detail_surface: str | None = None
    panel_scroll_positions: dict[str, int] = field(default_factory=dict)
    detail_scroll_positions: dict[str, int] = field(default_factory=dict)


class StateStore:
    """Persistent state root with corruption-safe reads and atomic writes."""

    def __init__(self, *, state_root: str | None = None) -> None:
        resolved_root = state_root or os.environ.get("PERCEPTROME_TUI_STATE_ROOT") or "state/tui"
        self._root = Path(resolved_root)
        self._lock = threading.RLock()
        self._root.mkdir(parents=True, exist_ok=True)

        self._session_path = self._root / "session.json"
        self._jobs_path = self._root / "jobs.json"
        self._events_path = self._root / "events.jsonl"
        self._launcher_history_path = self._root / "launcher_history.json"
        self._startup_context_path = self._root / "startup_context.json"

        self._session = self._load_session()
        self._apply_startup_context_if_present()

    @property
    def active_view(self) -> str:
        return self._session.last_panel

    def set_active_view(self, view: str) -> None:
        self._session.last_panel = view
        self._save_session()
        self.append_event(TUIEvent(kind="view", message=f"Switched to {view}"))

    def set_value(self, key: str, value: Any) -> None:
        payload = {"key": key, "value": value}
        self.append_event(TUIEvent(kind="state", message=f"Updated {key}", payload=payload))

    def set_drawer_toggle(self, drawer_id: str, open_state: bool) -> None:
        self._session.drawer_toggles[drawer_id] = bool(open_state)
        self._save_session()

    def set_active_focus(self, focus_id: str | None, *, active_job_id: str | None = None) -> None:
        self._session.active_focus = focus_id
        if active_job_id is not None:
            self._session.active_job_id = active_job_id
            self._session.selected_job_id = active_job_id
        self._save_session()

    def set_selected_job(self, job_id: str | None) -> None:
        self._session.selected_job_id = job_id
        self._save_session()

    def set_selected_artifact_path(self, artifact_path: str | None) -> None:
        self._session.selected_artifact_path = artifact_path
        self._save_session()

    def set_active_detail_surface(self, surface: str | None) -> None:
        self._session.active_detail_surface = surface
        self._save_session()

    def set_panel_scroll_position(self, panel_id: str, offset: int) -> None:
        self._session.panel_scroll_positions[panel_id] = max(0, int(offset))
        self._save_session()

    def set_detail_scroll_position(self, surface: str, offset: int) -> None:
        self._session.detail_scroll_positions[surface] = max(0, int(offset))
        self._save_session()

    def get_session(self) -> SessionState:
        return SessionState(**asdict(self._session))

    def list_jobs(self) -> list[JobRecord]:
        payload = self._load_json(self._jobs_path, default={"schema_version": SCHEMA_VERSION, "jobs": []})
        rows = payload.get("jobs") if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            return []
        out: list[JobRecord] = []
        for row in rows:
            record = _parse_job_record(row)
            if record is not None:
                out.append(record)
        return sorted(out, key=lambda item: item.updated_at, reverse=True)

    def upsert_job(self, record: JobRecord) -> None:
        with self._lock:
            payload = self._load_json(self._jobs_path, default={"schema_version": SCHEMA_VERSION, "jobs": []})
            rows = payload.get("jobs") if isinstance(payload, dict) else []
            if not isinstance(rows, list):
                rows = []
            record.updated_at = _utc_now()
            if record.finished_at and record.status in {"failed", "healthy", "completed", "canceled"}:
                record.finished_at = record.finished_at
            row = _job_record_to_json(record)
            replaced = False
            for idx, old in enumerate(rows):
                if isinstance(old, dict) and str(old.get("id")) == record.id:
                    rows[idx] = row
                    replaced = True
                    break
            if not replaced:
                rows.append(row)
            self._atomic_write_json(self._jobs_path, {"schema_version": SCHEMA_VERSION, "jobs": rows})

    def append_event(self, event: TUIEvent) -> None:
        encoded = {
            "schema_version": SCHEMA_VERSION,
            "created_at": event.created_at.isoformat(),
            "kind": event.kind,
            "message": event.message,
            "payload": event.payload,
        }
        line = json.dumps(encoded, sort_keys=True)
        with self._lock:
            with self._events_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def apply_job_event(self, event: JobEventBase) -> JobRecord | None:
        if not event.job_id:
            return None
        existing = {row.id: row for row in self.list_jobs()}
        record = existing.get(event.job_id) or JobRecord(id=event.job_id, run_id=event.run_id or event.job_id, kind="unknown", status="busy")
        record.run_id = event.run_id or record.run_id or event.job_id

        if isinstance(event, JobStartedEvent):
            record.title = event.title
            record.status = "busy"
            record.finished_at = None
        elif isinstance(event, JobStageUpdatedEvent):
            if event.message:
                record.config["last_message"] = event.message
            if event.stage:
                record.config["stage"] = event.stage
            if event.stage == "error":
                record.status = "failed"
        elif isinstance(event, JobArtifactEmittedEvent):
            artifact = {"path": event.path, "role": event.role, "created_at": _utc_now()}
            if event.path:
                record.artifacts.append(artifact)
                self.set_selected_artifact_path(event.path)
        elif isinstance(event, JobWarningEvent):
            record.config["last_warning"] = event.message
        elif isinstance(event, JobErrorEvent):
            record.status = "failed"
            record.failure_summary = FailureSummary(latest_warning_or_error=event.error or event.message, stage="error")
        elif isinstance(event, JobCompletedEvent):
            record.status = "completed"
            record.finished_at = _utc_now()
        elif isinstance(event, JobFailedEvent):
            record.status = "failed"
            record.finished_at = _utc_now()
            record.failure_summary = FailureSummary(latest_warning_or_error=event.error or event.message, stage="failed")
        elif isinstance(event, JobCanceledEvent):
            record.status = "canceled"
            record.finished_at = _utc_now()
        elif isinstance(event, JobRecoveryEvent):
            record.config["last_recovery"] = event.message
            if record.status == "failed":
                record.status = "busy"

        self.upsert_job(record)
        self.set_selected_job(record.id)
        self.set_active_focus(self._session.active_focus, active_job_id=record.id)
        return record

    def add_launcher_history(self, action: str, *, command: str | None = None, panel: str | None = None) -> None:
        payload = self._load_json(self._launcher_history_path, default={"schema_version": SCHEMA_VERSION, "history": []})
        rows = payload.get("history") if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            rows = []
        rows.append({"at": _utc_now(), "action": action, "command": command, "panel": panel})
        rows = rows[-100:]
        self._atomic_write_json(self._launcher_history_path, {"schema_version": SCHEMA_VERSION, "history": rows})

    def launcher_history(self, limit: int = 25) -> list[dict[str, Any]]:
        payload = self._load_json(self._launcher_history_path, default={"schema_version": SCHEMA_VERSION, "history": []})
        rows = payload.get("history") if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            return []
        return [dict(row) for row in rows[-limit:] if isinstance(row, dict)]

    def launcher_history_tail(self, *, limit: int = 25, offset_from_end: int = 0) -> list[dict[str, Any]]:
        payload = self._load_json(self._launcher_history_path, default={"schema_version": SCHEMA_VERSION, "history": []})
        rows = payload.get("history") if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            return []
        head = max(0, len(rows) - max(0, int(offset_from_end)))
        tail = rows[:head]
        if limit <= 0:
            return []
        return [dict(row) for row in tail[-limit:] if isinstance(row, dict)]

    def read_events_tail(self, *, limit: int = 100, offset_from_end: int = 0) -> list[dict[str, Any]]:
        if limit <= 0 or not self._events_path.exists():
            return []
        try:
            lines = self._events_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []
        head = max(0, len(lines) - max(0, int(offset_from_end)))
        chunk = lines[:head][-limit:]
        out: list[dict[str, Any]] = []
        for line in chunk:
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                out.append(payload)
        return out

    def open_active_job(self) -> JobRecord | None:
        active_id = self._session.active_job_id
        if not active_id:
            return None
        return next((job for job in self.list_jobs() if job.id == active_id), None)

    def open_last_failed_job(self) -> JobRecord | None:
        for job in self.list_jobs():
            if job.status.lower() == "failed" or job.failure_summary is not None:
                return job
        return None

    def rerun_last_job(self) -> dict[str, Any] | None:
        jobs = self.list_jobs()
        if not jobs:
            return None
        recent = jobs[0]
        return {
            "job_id": recent.id,
            "run_id": recent.run_id,
            "kind": recent.kind,
            "config": dict(recent.config),
            "rerun_command": recent.config.get("command"),
        }

    def open_latest_checkpoint_output(self, job_id: str | None = None) -> str | None:
        jobs = self.list_jobs()
        job = self.open_active_job() if job_id is None else next((x for x in jobs if x.id == job_id), None)
        if job is None and jobs:
            job = jobs[0]
        if job is None:
            return None

        preferred_roles = ("checkpoint", "output")
        artifacts = [dict(item) for item in job.artifacts if isinstance(item, dict)]

        def _artifact_key(item: dict[str, Any]) -> tuple[str, float]:
            created_at = str(item.get("created_at") or "")
            path = Path(str(item.get("path") or ""))
            mtime = path.stat().st_mtime if path.exists() else -1.0
            return created_at, mtime

        filtered = [
            item
            for item in artifacts
            if str(item.get("role") or "").lower() in preferred_roles or any(token in str(item.get("path") or "").lower() for token in preferred_roles)
        ]
        if not filtered:
            return None
        filtered.sort(key=_artifact_key)
        latest = filtered[-1]
        path = str(latest.get("path") or "")
        return path or None

    def _load_session(self) -> SessionState:
        payload = self._load_json(
            self._session_path,
            default={
                "schema_version": SCHEMA_VERSION,
                "last_panel": "overview",
                "drawer_toggles": {},
                "active_focus": None,
                "active_job_id": None,
                "selected_job_id": None,
                "selected_artifact_path": None,
                "active_detail_surface": None,
                "panel_scroll_positions": {},
                "detail_scroll_positions": {},
            },
        )
        return SessionState(
            last_panel=str(payload.get("last_panel") or "overview"),
            drawer_toggles=dict(payload.get("drawer_toggles") or {}),
            active_focus=payload.get("active_focus"),
            active_job_id=payload.get("active_job_id"),
            selected_job_id=payload.get("selected_job_id") or payload.get("active_job_id"),
            selected_artifact_path=payload.get("selected_artifact_path"),
            active_detail_surface=payload.get("active_detail_surface"),
            panel_scroll_positions={str(k): int(v) for k, v in dict(payload.get("panel_scroll_positions") or {}).items()},
            detail_scroll_positions={str(k): int(v) for k, v in dict(payload.get("detail_scroll_positions") or {}).items()},
        )

    def _save_session(self) -> None:
        payload = {"schema_version": SCHEMA_VERSION, **asdict(self._session)}
        self._atomic_write_json(self._session_path, payload)

    def _apply_startup_context_if_present(self) -> None:
        if not self._startup_context_path.exists():
            return
        try:
            payload = json.loads(self._startup_context_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        panel = payload.get("panel")
        if panel:
            self._session.last_panel = str(panel)
        job_id = payload.get("job_id")
        run_id = payload.get("run_id")
        selected = str(job_id or run_id or "").strip() or None
        if selected:
            self._session.active_job_id = selected
            self._session.selected_job_id = selected
        detail_surface = payload.get("detail_surface")
        if detail_surface:
            self._session.active_detail_surface = str(detail_surface)
        self._save_session()

    def _load_json(self, path: Path, *, default: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return dict(default)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return dict(default)
        if not isinstance(payload, dict):
            return dict(default)
        return self._migrate_payload(path=path, payload=payload, default=default)

    def _migrate_payload(self, *, path: Path, payload: dict[str, Any], default: dict[str, Any]) -> dict[str, Any]:
        version = int(payload.get("schema_version") or 1)
        if version == SCHEMA_VERSION:
            return payload
        migrations: dict[str, dict[int, Any]] = {
            "session.json": {1: _migrate_session_v1_to_v2, 2: _migrate_passthrough_v2_to_v3},
            "jobs.json": {1: _migrate_passthrough_v1_to_v2, 2: _migrate_passthrough_v2_to_v3},
            "launcher_history.json": {1: _migrate_passthrough_v1_to_v2, 2: _migrate_passthrough_v2_to_v3},
        }
        current = dict(payload)
        migrators = migrations.get(path.name, {})
        while version < SCHEMA_VERSION:
            migrate = migrators.get(version)
            if migrate is None:
                return dict(default)
            current = migrate(current)
            version = int(current.get("schema_version") or 0)
        if version != SCHEMA_VERSION:
            return dict(default)
        return current

    def _atomic_write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_job_record(value: Any) -> JobRecord | None:
    if not isinstance(value, dict):
        return None
    job_id = str(value.get("id") or "")
    if not job_id:
        return None
    summary = value.get("failure_summary")
    failure_summary: FailureSummary | None = None
    if isinstance(summary, dict):
        failure_summary = FailureSummary(
            stage=str(summary.get("stage") or ""),
            latest_warning_or_error=str(summary.get("latest_warning_or_error") or ""),
            traceback_path=str(summary.get("traceback_path")) if summary.get("traceback_path") else None,
            suggested_next_action=str(summary.get("suggested_next_action") or "inspect_logs"),
        )
    return JobRecord(
        id=job_id,
        run_id=str(value.get("run_id") or job_id),
        kind=str(value.get("kind") or "unknown"),
        status=str(value.get("status") or "unknown"),
        title=str(value.get("title") or ""),
        config=dict(value.get("config") or {}),
        artifacts=[dict(item) for item in value.get("artifacts") or [] if isinstance(item, dict)],
        created_at=str(value.get("created_at") or _utc_now()),
        updated_at=str(value.get("updated_at") or _utc_now()),
        finished_at=str(value.get("finished_at")) if value.get("finished_at") else None,
        failure_summary=failure_summary,
    )


def _job_record_to_json(record: JobRecord) -> dict[str, Any]:
    payload = asdict(record)
    if record.failure_summary is None:
        payload["failure_summary"] = None
    return payload


def _migrate_passthrough_v1_to_v2(payload: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(payload)
    migrated["schema_version"] = 2
    return migrated


def _migrate_passthrough_v2_to_v3(payload: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(payload)
    migrated["schema_version"] = 3
    return migrated


def _migrate_session_v1_to_v2(payload: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(payload)
    active_job_id = migrated.get("active_job_id")
    migrated.setdefault("selected_job_id", active_job_id)
    migrated.setdefault("selected_artifact_path", None)
    migrated.setdefault("active_detail_surface", None)
    migrated.setdefault("panel_scroll_positions", {})
    migrated.setdefault("detail_scroll_positions", {})
    migrated["schema_version"] = 2
    return migrated
