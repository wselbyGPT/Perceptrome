import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..db import Base


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    actor_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), index=True, nullable=True)
    target_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), index=True, nullable=True)
    action: Mapped[str] = mapped_column(String(128), index=True)
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(512), nullable=True)
    metadata_json: Mapped[str] = mapped_column(String, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now(), index=True)

    actor_user = relationship("User", back_populates="actor_audit_events", foreign_keys=[actor_user_id])
    target_user = relationship("User", back_populates="target_audit_events", foreign_keys=[target_user_id])


Index("ix_audit_events_actor_created", AuditEvent.actor_user_id, AuditEvent.created_at)
Index("ix_audit_events_target_created", AuditEvent.target_user_id, AuditEvent.created_at)
Index("ix_audit_events_action_created", AuditEvent.action, AuditEvent.created_at)
