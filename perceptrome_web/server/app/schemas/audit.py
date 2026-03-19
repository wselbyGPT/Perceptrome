from datetime import datetime

from pydantic import BaseModel, Field


class AuditEventOut(BaseModel):
    id: str
    actor_user_id: str | None = None
    actor_email: str | None = None
    target_user_id: str | None = None
    target_email: str | None = None
    action: str
    ip_address: str | None = None
    user_agent: str | None = None
    metadata: dict = Field(default_factory=dict)
    created_at: datetime


class AuditEventListOut(BaseModel):
    events: list[AuditEventOut]
    total: int
