from datetime import datetime

from pydantic import BaseModel, Field


class RunStartRequest(BaseModel):
    config: dict = Field(default_factory=dict)


class RunArtifactOut(BaseModel):
    id: int
    phase: str | None = None
    path: str
    label: str | None = None
    download_url: str
    created_at: datetime


class ConfigSnapshotOut(BaseModel):
    path: str
    sha256: str
    format: str = "json"


class RunResultOut(BaseModel):
    config_snapshot: ConfigSnapshotOut | None = None


class RunOut(BaseModel):
    run_id: str
    user_id: str
    kind: str
    state: str
    message: str | None = None
    config: dict = Field(default_factory=dict)
    result: RunResultOut | dict | None = None
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    artifacts: list[RunArtifactOut] = Field(default_factory=list)


class RunSummaryOut(BaseModel):
    total_runs: int
    state_counts: dict[str, int] = Field(default_factory=dict)
    queued: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    canceled: int = 0
    latest_failed_run_id: str | None = None
    latest_failed_at: datetime | None = None


class RunsBoardOut(BaseModel):
    generated_at: datetime
    runs: list[RunOut] = Field(default_factory=list)


class LineageNodeOut(BaseModel):
    id: str
    kind: str
    label: str
    depth: int = 0
    run_id: str | None = None
    artifact_id: str | None = None
    artifact_type: str | None = None
    run_state: str | None = None
    path: str | None = None
    relation: str | None = None
    hash: str | None = None
    config_snapshot: ConfigSnapshotOut | None = None
    payload: dict = Field(default_factory=dict)


class LineageEdgeOut(BaseModel):
    source: str
    target: str
    relation: str


class RunLineageOut(BaseModel):
    run_id: str
    depth_limit: int
    artifact_type_filter: str | None = None
    run_state_filter: list[str] = Field(default_factory=list)
    nodes: list[LineageNodeOut] = Field(default_factory=list)
    edges: list[LineageEdgeOut] = Field(default_factory=list)
