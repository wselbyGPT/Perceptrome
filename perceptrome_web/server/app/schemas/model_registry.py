from datetime import datetime

from pydantic import BaseModel, Field


class ModelArtifactOut(BaseModel):
    id: int
    role: str
    path: str
    label: str | None = None
    download_url: str
    created_at: datetime


class ModelVersionOut(BaseModel):
    id: str
    model_id: str
    source_run_id: str | None = None
    version_label: str
    status: str
    architecture: str | None = None
    tokenizer: str | None = None
    checkpoint_path: str | None = None
    config_snapshot_path: str | None = None
    manifest_path: str | None = None
    metrics: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
    created_at: datetime
    promoted_at: datetime | None = None
    artifacts: list[ModelArtifactOut] = Field(default_factory=list)


class RegisteredModelOut(BaseModel):
    id: str
    owner_user_id: str
    name: str
    description: str | None = None
    visibility: str
    status: str
    tags: list[str] = Field(default_factory=list)
    current_version_id: str | None = None
    created_at: datetime
    updated_at: datetime
    versions: list[ModelVersionOut] = Field(default_factory=list)
    current_version: ModelVersionOut | None = None


class ModelRegisterFromRunRequest(BaseModel):
    run_id: str
    model_id: str | None = None
    name: str | None = None
    description: str | None = None
    visibility: str | None = None
    tags: list[str] | None = None
    version_label: str | None = None
    version_status: str = "candidate"


class ModelUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    visibility: str | None = None
    status: str | None = None
    tags: list[str] | None = None
    current_version_id: str | None = None


class ModelVersionUpdateRequest(BaseModel):
    version_label: str | None = None
    status: str | None = None
    promote_current: bool = False


class ModelRegistrySummaryOut(BaseModel):
    total_models: int
    total_versions: int
    architecture_counts: dict[str, int] = Field(default_factory=dict)
    tokenizer_counts: dict[str, int] = Field(default_factory=dict)
