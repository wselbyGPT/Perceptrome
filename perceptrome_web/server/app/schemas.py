# server/app/schemas.py
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=256)
    username: str | None = Field(default=None, min_length=3, max_length=64)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)


class VerifyEmailRequest(BaseModel):
    token: str = Field(min_length=8, max_length=512)


class ResendVerificationRequest(BaseModel):
    email: EmailStr


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str = Field(min_length=8, max_length=512)
    new_password: str = Field(min_length=8, max_length=256)


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=12, max_length=256)


class AdminCreateUserRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=256)
    username: str | None = Field(default=None, min_length=3, max_length=64)
    role: str
    is_active: bool = True
    must_change_password: bool = True


class UserOut(BaseModel):
    id: str
    email: EmailStr
    username: str | None
    role: str
    is_active: bool
    must_change_password: bool
    email_verified_at: datetime | None

    @classmethod
    def from_model(cls, u):
        return cls(
            id=u.id,
            email=u.email,
            username=u.username,
            role=u.role,
            is_active=u.is_active,
            must_change_password=u.must_change_password,
            email_verified_at=u.email_verified_at,
        )


class MessageOut(BaseModel):
    message: str


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


class DatasetSplitOut(BaseModel):
    name: str
    count: int


class DatasetCatalogItemOut(BaseModel):
    dataset_id: str
    source: str
    sequence_count: int
    split_metadata: list[DatasetSplitOut] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    last_updated_hash: str


class DatasetDetailOut(DatasetCatalogItemOut):
    manifest_path: str


class DatasetPreviewOut(BaseModel):
    dataset_id: str
    source: str
    preview: list[str] = Field(default_factory=list)
    total_rows: int
