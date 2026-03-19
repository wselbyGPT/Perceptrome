import json
from datetime import timedelta
from typing import Any

from fastapi import HTTPException, Request, Response, WebSocket, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.config import settings
from ..core.security import hash_session_token, make_session_token
from ..models import User, UserSession
from ..schemas import SessionOut
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


def list_sessions(db: Session, user: User, *, current_cookie: str | None = None) -> list[SessionOut]:
    current_hash = hash_session_token(current_cookie) if current_cookie else None
    sessions = db.execute(select(UserSession).where(UserSession.user_id == user.id).order_by(UserSession.created_at.desc())).scalars().all()
    return [
        SessionOut(
            id=sess.id,
            created_at=sess.created_at,
            expires_at=sess.expires_at,
            revoked_at=sess.revoked_at,
            ip_address=sess.ip_address,
            user_agent=sess.user_agent,
            is_current=bool(current_hash and sess.token_hash == current_hash),
        )
        for sess in sessions
    ]


def revoke_session_by_cookie(db: Session, raw_cookie: str | None) -> None:
    if not raw_cookie:
        return
    sess = db.execute(select(UserSession).where(UserSession.token_hash == hash_session_token(raw_cookie)).where(UserSession.revoked_at.is_(None))).scalar_one_or_none()
    if sess:
        sess.revoked_at = audit_service.utcnow()
        db.commit()


def revoke_session_by_id(db: Session, user: User, session_id: str, *, current_cookie: str | None = None) -> bool:
    sess = db.execute(select(UserSession).where(UserSession.id == session_id).where(UserSession.user_id == user.id)).scalar_one_or_none()
    if not sess:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Session not found')
    if sess.revoked_at is None:
        sess.revoked_at = audit_service.utcnow()
        db.commit()
    return bool(current_cookie and sess.token_hash == hash_session_token(current_cookie))


def revoke_other_sessions(db: Session, user: User, *, current_cookie: str | None = None) -> int:
    now = audit_service.utcnow()
    current_hash = hash_session_token(current_cookie) if current_cookie else None
    sessions = db.execute(select(UserSession).where(UserSession.user_id == user.id).where(UserSession.revoked_at.is_(None)).where(UserSession.expires_at > now)).scalars().all()
    revoked_count = 0
    for sess in sessions:
        if sess.token_hash != current_hash:
            sess.revoked_at = now
            revoked_count += 1
    db.commit()
    return revoked_count


def revoke_all_active_sessions(db: Session, user: User) -> int:
    now = audit_service.utcnow()
    sessions = db.execute(select(UserSession).where(UserSession.user_id == user.id).where(UserSession.revoked_at.is_(None)).where(UserSession.expires_at > now)).scalars().all()
    revoked_count = 0
    for sess in sessions:
        sess.revoked_at = now
        revoked_count += 1
    db.commit()
    return revoked_count


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
