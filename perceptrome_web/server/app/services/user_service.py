from sqlalchemy import select
from sqlalchemy.orm import Session

from fastapi import HTTPException

from ..core.security import hash_password
from ..models import User
from ..schemas import AdminUserOut, UserOut
from . import audit_service


def normalize_identity(email: str, username: str | None) -> tuple[str, str | None]:
    return email.lower().strip(), username.strip() if username else None


def ensure_unique_user_fields(db: Session, *, email: str, username: str | None) -> None:
    if db.execute(select(User).where(User.email == email)).scalar_one_or_none():
        raise HTTPException(status_code=409, detail='Email already registered')
    if username and db.execute(select(User).where(User.username == username)).scalar_one_or_none():
        raise HTTPException(status_code=409, detail='Username already taken')


def create_registered_user(db: Session, *, email: str, password: str, username: str | None) -> User:
    email, username = normalize_identity(email, username)
    ensure_unique_user_fields(db, email=email, username=username)
    user = User(email=email, username=username, password_hash=hash_password(password), role='user', is_active=True, must_change_password=False, email_verified_at=None)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def create_admin_user(db: Session, *, email: str, password: str, username: str | None, role: str, is_active: bool, must_change_password: bool) -> User:
    email, username = normalize_identity(email, username)
    role = role.strip().lower()
    if role not in {'admin', 'user'}:
        raise HTTPException(status_code=400, detail='Invalid role')
    ensure_unique_user_fields(db, email=email, username=username)
    user = User(email=email, username=username, password_hash=hash_password(password), role=role, is_active=is_active, must_change_password=must_change_password, email_verified_at=audit_service.utcnow())
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def list_admin_users(db: Session) -> list[AdminUserOut]:
    users = db.execute(select(User).order_by(User.created_at.desc())).scalars().all()
    return [AdminUserOut.from_model(u) for u in users]


def user_out(user: User) -> UserOut:
    return UserOut.from_model(user)
