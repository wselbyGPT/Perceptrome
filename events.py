"""Typed events emitted inside the Perceptrome TUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


@dataclass(slots=True)
class EventBase:
    """Base event containing timestamp metadata."""

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class JobEventBase(EventBase):
    """Base class for job-scoped events emitted to panels."""

    job_id: str = ""
    run_id: str = ""


@dataclass(slots=True)
class JobStartedEvent(JobEventBase):
    kind: str = "job_started"
    title: str = ""


@dataclass(slots=True)
class JobStageUpdatedEvent(JobEventBase):
    kind: str = "stage_update"
    stage: str = ""
    message: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class JobMetricUpdatedEvent(JobEventBase):
    kind: str = "metric_update"
    metric_name: str = ""
    latest_value: float = 0.0
    rolling_value: float = 0.0
    step: int | None = None


@dataclass(slots=True)
class JobArtifactEmittedEvent(JobEventBase):
    kind: str = "artifact_emitted"
    path: str = ""
    role: str = "artifact"


@dataclass(slots=True)
class JobWarningEvent(JobEventBase):
    kind: str = "warning"
    message: str = ""


@dataclass(slots=True)
class JobErrorEvent(JobEventBase):
    kind: str = "error"
    message: str = ""
    error: str = ""


@dataclass(slots=True)
class JobCompletedEvent(JobEventBase):
    kind: str = "completed"
    message: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class JobFailedEvent(JobEventBase):
    kind: str = "failed"
    message: str = ""
    error: str = ""


@dataclass(slots=True)
class JobCanceledEvent(JobEventBase):
    kind: str = "canceled"
    message: str = ""


@dataclass(slots=True)
class JobRecoveryEvent(JobEventBase):
    kind: str = "recovery"
    message: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TUIEvent(EventBase):
    """Legacy event payload used by state/history subsystems."""

    kind: str = "generic"
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
