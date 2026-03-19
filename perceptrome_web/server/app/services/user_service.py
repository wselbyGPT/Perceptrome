from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from fastapi import HTTPException

from ..core.security import hash_password
from ..models import User
from ..schemas import AdminUserOut, UserOut
from . import audit_service, auth_service, session_service

_VALID_ROLES = {'admin', 'user'}


def normalize_identity(email: str, username: str | None) -> tuple[str, str | None]:
    return email.lower().strip(), username.strip() if username else None


def _normalize_username(username: str | None) -> str | None:
    username = username.strip() if username else None
    return username or None


def _normalize_role(role: str | None) -> str | None:
    if role is None:
        return None
    normalized = role.strip().lower()
    if normalized not in _VALID_ROLES:
        raise HTTPException(status_code=400, detail='Invalid role')
    return normalized


def ensure_unique_user_fields(db: Session, *, email: str, username: str | None) -> None:
    if db.execute(select(User).where(User.email == email)).scalar_one_or_none():
        raise HTTPException(status_code=409, detail='Email already registered')
    if username and db.execute(select(User).where(User.username == username)).scalar_one_or_none():
        raise HTTPException(status_code=409, detail='Username already taken')


def _ensure_unique_username_for_update(db: Session, *, user_id: str, username: str | None) -> None:
    if not username:
        return
    existing = db.execute(select(User).where(User.username == username).where(User.id != user_id)).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=409, detail='Username already taken')


def get_user_or_404(db: Session, user_id: str) -> User:
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail='User not found')
    return user


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
    role = _normalize_role(role)
    ensure_unique_user_fields(db, email=email, username=username)
    user = User(email=email, username=username, password_hash=hash_password(password), role=role, is_active=is_active, must_change_password=must_change_password, email_verified_at=audit_service.utcnow())
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def list_admin_users(
    db: Session,
    *,
    search: str | None = None,
    role: str | None = None,
    state: str | None = None,
    verification: str | None = None,
    must_change_password: bool | None = None,
) -> tuple[list[AdminUserOut], int]:
    stmt = select(User)
    count_stmt = select(func.count()).select_from(User)
    filters = []

    if search:
        term = f"%{search.strip().lower()}%"
        filters.append(or_(func.lower(User.email).like(term), func.lower(func.coalesce(User.username, '')).like(term), User.id.like(f"%{search.strip()}%")))

    normalized_role = _normalize_role(role) if role else None
    if normalized_role:
        filters.append(User.role == normalized_role)

    if state == 'active':
        filters.append(User.is_active.is_(True))
    elif state == 'suspended':
        filters.append(User.is_active.is_(False))
    elif state and state != 'all':
        raise HTTPException(status_code=400, detail='Invalid account state filter')

    if verification == 'verified':
        filters.append(User.email_verified_at.is_not(None))
    elif verification == 'pending':
        filters.append(User.email_verified_at.is_(None))
    elif verification and verification != 'all':
        raise HTTPException(status_code=400, detail='Invalid verification filter')

    if must_change_password is not None:
        filters.append(User.must_change_password.is_(must_change_password))

    for filter_clause in filters:
        stmt = stmt.where(filter_clause)
        count_stmt = count_stmt.where(filter_clause)

    stmt = stmt.order_by(User.created_at.desc())
    users = db.execute(stmt).scalars().all()
    total = int(db.execute(count_stmt).scalar_one())
    return [AdminUserOut.from_model(u) for u in users], total


def update_admin_user(db: Session, *, user_id: str, username: str | None, role: str | None, is_active: bool | None, must_change_password: bool | None) -> User:
    user = get_user_or_404(db, user_id)
    normalized_username = _normalize_username(username)
    normalized_role = _normalize_role(role)
    _ensure_unique_username_for_update(db, user_id=user.id, username=normalized_username)

    user.username = normalized_username
    if normalized_role is not None:
        user.role = normalized_role
    if is_active is not None:
        user.is_active = is_active
    if must_change_password is not None:
        user.must_change_password = must_change_password

    db.commit()
    db.refresh(user)
    return user


def suspend_user(db: Session, *, user_id: str) -> tuple[User, int]:
    user = get_user_or_404(db, user_id)
    user.is_active = False
    revoked_count = session_service.revoke_all_active_sessions(db, user)
    db.refresh(user)
    return user, revoked_count


def activate_user(db: Session, *, user_id: str) -> User:
    user = get_user_or_404(db, user_id)
    user.is_active = True
    db.commit()
    db.refresh(user)
    return user


def admin_force_reset(db: Session, *, user_id: str) -> tuple[User, int]:
    user = get_user_or_404(db, user_id)
    user.must_change_password = True
    db.commit()
    revoked_count = session_service.revoke_all_active_sessions(db, user)
    db.refresh(user)
    return user, revoked_count


def resend_user_verification(db: Session, *, user_id: str) -> User:
    user = get_user_or_404(db, user_id)
    auth_service.resend_verification(db, user.email)
    db.refresh(user)
    return user


def revoke_user_sessions(db: Session, *, user_id: str) -> tuple[User, int]:
    user = get_user_or_404(db, user_id)
    revoked_count = session_service.revoke_all_active_sessions(db, user)
    db.refresh(user)
    return user, revoked_count


def user_out(user: User) -> UserOut:
    return UserOut.from_model(user)
