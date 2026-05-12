import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..db import Base


class RegisteredModel(Base):
    __tablename__ = "registered_models"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(160), index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    visibility: Mapped[str] = mapped_column(String(32), default="private", index=True)
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    tags_json: Mapped[str] = mapped_column(Text, default="[]")
    current_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now(), onupdate=func.now(), index=True)

    versions: Mapped[list["ModelVersion"]] = relationship(back_populates="model", cascade="all, delete-orphan")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id: Mapped[str] = mapped_column(ForeignKey("registered_models.id", ondelete="CASCADE"), index=True)
    source_run_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    version_label: Mapped[str] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(32), default="candidate", index=True)
    architecture: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    tokenizer: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    checkpoint_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    config_snapshot_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    manifest_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now(), index=True)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)

    model: Mapped[RegisteredModel] = relationship(back_populates="versions")
    artifacts: Mapped[list["ModelVersionArtifact"]] = relationship(back_populates="version", cascade="all, delete-orphan")


class ModelVersionArtifact(Base):
    __tablename__ = "model_version_artifacts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(80), index=True)
    path: Mapped[str] = mapped_column(String(1024))
    label: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now(), index=True)

    version: Mapped[ModelVersion] = relationship(back_populates="artifacts")


Index("ix_registered_models_owner_updated", RegisteredModel.owner_user_id, RegisteredModel.updated_at)
Index("ix_model_versions_model_created", ModelVersion.model_id, ModelVersion.created_at)
Index("ix_model_version_artifacts_version_created", ModelVersionArtifact.version_id, ModelVersionArtifact.created_at)
