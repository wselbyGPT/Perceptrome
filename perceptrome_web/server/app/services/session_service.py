import json
from datetime import timedelta
from typing import Any

from fastapi import HTTPException, Request, Response, WebSocket
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.config import settings
from ..core.db import SessionLocal
from ..core.security import hash_session_token, make_session_token
from ..models import User, UserSession
from . import audit_service


def set_session_cookie(response: Response, raw_token: str) -> None:
    response.set_cookie(key=settings.session_cookie_name, value=raw_token, httponly=True, secure=settings.cookie_secure, samesite=settings.cookie_samesite, max_age=settings.session_ttl_hours * 3600, path='/', domain=settings.cookie_domain)


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=settings.session_cookie_name, path='/', domain=settings.cookie_domain)


def issue_session(db: Session, user: User, request: Request) -> str:
    raw = make_session_token()
    sess = UserSession(user_id=user.id, token_hash=hash_session_token(raw), expires_at=audit_service.utcnow() + timedelta(hours=settings.session_ttl_hours), ip_address=request.client.host if request.client else None, user_agent=request.headers.get('user-agent'))
    db.add(sess)
    user.last_login_at = audit_service.utcnow()
    db.commit()
    return raw


def revoke_session_by_cookie(db: Session, raw_cookie: str | None) -> None:
    if not raw_cookie:
        return
    sess = db.execute(select(UserSession).where(UserSession.token_hash == hash_session_token(raw_cookie)).where(UserSession.revoked_at.is_(None))).scalar_one_or_none()
    if sess:
        sess.revoked_at = audit_service.utcnow()
        db.commit()


def revoke_other_sessions(db: Session, user: User, *, current_cookie: str | None = None) -> None:
    now = audit_service.utcnow()
    current_hash = hash_session_token(current_cookie) if current_cookie else None
    sessions = db.execute(select(UserSession).where(UserSession.user_id == user.id).where(UserSession.revoked_at.is_(None)).where(UserSession.expires_at > now)).scalars().all()
    for sess in sessions:
        if sess.token_hash != current_hash:
            sess.revoked_at = now
    db.commit()


def revoke_all_active_sessions(db: Session, user: User) -> None:
    now = audit_service.utcnow()
    sessions = db.execute(select(UserSession).where(UserSession.user_id == user.id).where(UserSession.revoked_at.is_(None)).where(UserSession.expires_at > now)).scalars().all()
    for sess in sessions:
        sess.revoked_at = now
    db.commit()


def websocket_auth_user(websocket: WebSocket, db: Session) -> User | None:
    raw = websocket.cookies.get(settings.session_cookie_name)
    if not raw:
        return None
    sess = db.execute(select(UserSession).where(UserSession.token_hash == hash_session_token(raw)).where(UserSession.revoked_at.is_(None)).where(UserSession.expires_at > audit_service.utcnow())).scalar_one_or_none()
    if not sess:
        return None
    user = db.get(User, sess.user_id)
    if not user or not user.is_active:
        return None
    return user


def emit_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload)
