import json
import logging
import sys
from collections import Counter
from datetime import datetime
from typing import Any

from fastapi import Request
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, aliased

from ..models import AuditEvent, User
from ..schemas import AuditEventListOut, AuditEventOut

_auth_logger = logging.getLogger("perceptrome.auth")
_auth_metrics: Counter[str] = Counter()


class AuditActions:
    USER_CREATED = "admin.user_created"
    USER_UPDATED = "admin.user_updated"
    INVITE_CREATED = "admin.invite_created"
    INVITE_REVOKED = "admin.invite_revoked"
    ROLE_CHANGED = "admin.user_role_changed"
    USER_ACTIVATED = "admin.user_activated"
    USER_SUSPENDED = "admin.user_suspended"
    PASSWORD_RESET_FORCED = "admin.user_force_reset"
    VERIFICATION_RESENT = "admin.user_verification_resent"
    SESSION_REVOKED = "admin.user_sessions_revoked"
    AUTH_PASSWORD_RESET = "auth.password_reset"
    AUTH_SESSION_REVOKED = "auth.session_revoked"
    AUTH_OTHER_SESSIONS_REVOKED = "auth.other_sessions_revoked"
    PROFILE_UPDATED = "auth.profile_updated"
    PASSWORD_CHANGED = "auth.password_changed"


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


def list_audit_events(
    db: Session,
    *,
    limit: int = 100,
    action: str | None = None,
    actor: str | None = None,
    target: str | None = None,
    search: str | None = None,
) -> AuditEventListOut:
    actor_user = aliased(User)
    target_user = aliased(User)
    stmt = select(AuditEvent, actor_user.email, target_user.email).outerjoin(actor_user, AuditEvent.actor_user_id == actor_user.id).outerjoin(target_user, AuditEvent.target_user_id == target_user.id)
    count_stmt = select(func.count()).select_from(AuditEvent).outerjoin(actor_user, AuditEvent.actor_user_id == actor_user.id).outerjoin(target_user, AuditEvent.target_user_id == target_user.id)

    filters = []
    if action:
        filters.append(AuditEvent.action == action.strip())
    if actor:
        actor_value = actor.strip().lower()
        filters.append(or_(func.lower(func.coalesce(actor_user.email, '')).like(f"%{actor_value}%"), AuditEvent.actor_user_id == actor.strip()))
    if target:
        target_value = target.strip().lower()
        filters.append(or_(func.lower(func.coalesce(target_user.email, '')).like(f"%{target_value}%"), AuditEvent.target_user_id == target.strip()))
    if search:
        search_value = f"%{search.strip().lower()}%"
        filters.append(or_(
            func.lower(AuditEvent.action).like(search_value),
            func.lower(func.coalesce(actor_user.email, '')).like(search_value),
            func.lower(func.coalesce(target_user.email, '')).like(search_value),
            func.lower(func.coalesce(AuditEvent.metadata_json, '')).like(search_value),
        ))

    for filter_clause in filters:
        stmt = stmt.where(filter_clause)
        count_stmt = count_stmt.where(filter_clause)

    rows = db.execute(stmt.order_by(AuditEvent.created_at.desc()).limit(max(1, min(limit, 500)))).all()
    total = int(db.execute(count_stmt).scalar_one())
    return AuditEventListOut(events=[_row_to_out(row) for row in rows], total=total)


def _row_to_out(row: Any) -> AuditEventOut:
    event, actor_email, target_email = row
    return AuditEventOut(id=event.id, actor_user_id=event.actor_user_id, actor_email=actor_email, target_user_id=event.target_user_id, target_email=target_email, action=event.action, ip_address=event.ip_address, user_agent=event.user_agent, metadata=_loads(event.metadata_json), created_at=event.created_at)


def _loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}
