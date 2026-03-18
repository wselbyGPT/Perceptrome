import json
import logging
import sys
from collections import Counter
from datetime import datetime
from typing import Any

from fastapi import Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import AuditEvent
from ..schemas import AuditEventOut

_auth_logger = logging.getLogger("perceptrome.auth")
_auth_metrics: Counter[str] = Counter()


def utcnow() -> datetime:
    main = sys.modules.get('app.main')
    override = getattr(main, '_utcnow', None) if main else None
    if callable(override):
        return override()
    return datetime.utcnow()


def metric_inc(name: str) -> None:
    _auth_metrics[name] += 1


def structured_auth_log(event: str, **fields: Any) -> None:
    payload = {'event': event, **fields}
    _auth_logger.warning(json.dumps(payload, sort_keys=True))


def create_audit_event(db: Session, *, action: str, actor_user_id: str | None = None, target_user_id: str | None = None, request: Request | None = None, metadata: dict[str, Any] | None = None) -> AuditEvent:
    event = AuditEvent(
        action=action,
        actor_user_id=actor_user_id,
        target_user_id=target_user_id,
        ip_address=request.client.host if request and request.client else None,
        user_agent=request.headers.get('user-agent') if request else None,
        metadata_json=json.dumps(metadata or {}, sort_keys=True, default=str),
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


def list_audit_events(db: Session, limit: int = 100) -> list[AuditEventOut]:
    rows = db.execute(select(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(max(1, min(limit, 500)))).scalars().all()
    return [AuditEventOut(id=row.id, actor_user_id=row.actor_user_id, target_user_id=row.target_user_id, action=row.action, ip_address=row.ip_address, user_agent=row.user_agent, metadata=_loads(row.metadata_json), created_at=row.created_at) for row in rows]


def _loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}
