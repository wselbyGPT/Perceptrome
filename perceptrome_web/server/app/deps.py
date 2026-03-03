# server/app/deps.py
from datetime import datetime
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .db import SessionLocal
from .models import User, UserSession
from .security import hash_session_token


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _utcnow_naive() -> datetime:
    return datetime.utcnow()


def _get_user_from_session_cookie(request: Request, db: Session) -> User | None:
    raw = request.cookies.get(settings.session_cookie_name)
    if not raw:
        return None

    token_hash = hash_session_token(raw)
    now = _utcnow_naive()

    stmt = (
        select(UserSession)
        .where(UserSession.token_hash == token_hash)
        .where(UserSession.revoked_at.is_(None))
        .where(UserSession.expires_at > now)
    )
    sess = db.execute(stmt).scalar_one_or_none()
    if not sess:
        return None

    user = db.get(User, sess.user_id)
    if not user or not user.is_active:
        return None
    return user


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    user = _get_user_from_session_cookie(request, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user


def get_current_user_strict(user: User = Depends(get_current_user)) -> User:
    if user.must_change_password:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Password change required",
        )
    return user


def require_role(*allowed_roles: str):
    def _dep(user: User = Depends(get_current_user_strict)) -> User:
        if user.role not in allowed_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
        return user

    return _dep
