from datetime import timedelta
from urllib.parse import urlencode

from fastapi import HTTPException, status
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ..core.config import settings
from ..core.security import hash_session_token, make_session_token
from ..models import UserInvitation
from ..schemas import UserInvitationListOut, UserInvitationOut
from . import audit_service
from .user_service import _normalize_role


INVITATION_ACCEPT_BASE_URL = "http://localhost:5173/accept-invite"


def create_invitation(db: Session, *, email: str, role: str, invited_by_user_id: str, reissue: bool = True) -> tuple[UserInvitation, str]:
    normalized_email = email.lower().strip()
    normalized_role = _normalize_role(role) or "user"
    now = audit_service.utcnow()

    existing_active = db.execute(
        select(UserInvitation)
        .where(UserInvitation.email == normalized_email)
        .where(UserInvitation.accepted_at.is_(None))
        .where(UserInvitation.revoked_at.is_(None))
        .where(UserInvitation.expires_at > now)
        .order_by(UserInvitation.created_at.desc())
    ).scalar_one_or_none()

    if existing_active and not reissue:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An active invitation already exists for this email")

    if existing_active:
        existing_active.revoked_at = now

    raw_token = make_session_token()
    invitation = UserInvitation(
        email=normalized_email,
        role=normalized_role,
        invited_by_user_id=invited_by_user_id,
        token_hash=hash_session_token(raw_token),
        expires_at=now + timedelta(hours=settings.invitation_ttl_hours),
    )
    db.add(invitation)
    db.commit()
    db.refresh(invitation)
    return invitation, raw_token


def revoke_invitation(db: Session, *, invitation_id: str) -> UserInvitation:
    invitation = db.get(UserInvitation, invitation_id)
    if not invitation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invitation not found")
    if invitation.accepted_at is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Accepted invitations cannot be revoked")
    if invitation.revoked_at is None:
        invitation.revoked_at = audit_service.utcnow()
        db.commit()
        db.refresh(invitation)
    return invitation


def list_invitations(
    db: Session,
    *,
    status_filter: str | None = None,
    role: str | None = None,
    search: str | None = None,
    limit: int = 200,
) -> UserInvitationListOut:
    stmt = select(UserInvitation)
    count_stmt = select(func.count()).select_from(UserInvitation)
    filters = []

    if role:
        filters.append(UserInvitation.role == (_normalize_role(role) or role))
    if search:
        term = f"%{search.strip().lower()}%"
        filters.append(or_(func.lower(UserInvitation.email).like(term), UserInvitation.id.like(f"%{search.strip()}%")))

    normalized_status = (status_filter or "").strip().lower()
    now = audit_service.utcnow()
    if normalized_status == "pending":
        filters.append(UserInvitation.accepted_at.is_(None))
        filters.append(UserInvitation.revoked_at.is_(None))
        filters.append(UserInvitation.expires_at > now)
    elif normalized_status == "accepted":
        filters.append(UserInvitation.accepted_at.is_not(None))
    elif normalized_status == "revoked":
        filters.append(UserInvitation.revoked_at.is_not(None))
    elif normalized_status == "expired":
        filters.append(UserInvitation.accepted_at.is_(None))
        filters.append(UserInvitation.revoked_at.is_(None))
        filters.append(UserInvitation.expires_at <= now)
    elif normalized_status and normalized_status != "all":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid invitation status filter")

    for filter_clause in filters:
        stmt = stmt.where(filter_clause)
        count_stmt = count_stmt.where(filter_clause)

    rows = db.execute(stmt.order_by(UserInvitation.created_at.desc()).limit(max(1, min(limit, 500)))).scalars().all()
    total = int(db.execute(count_stmt).scalar_one())
    return UserInvitationListOut(invitations=[invitation_out(row) for row in rows], total=total)


def invitation_status(invitation: UserInvitation) -> str:
    if invitation.accepted_at is not None:
        return "accepted"
    if invitation.revoked_at is not None:
        return "revoked"
    if invitation.expires_at <= audit_service.utcnow():
        return "expired"
    return "pending"


def invitation_out(invitation: UserInvitation, *, raw_token: str | None = None) -> UserInvitationOut:
    invite_url = None
    token_preview = None
    if raw_token:
        invite_url = f"{INVITATION_ACCEPT_BASE_URL}?{urlencode({'token': raw_token})}"
        token_preview = raw_token
    return UserInvitationOut(
        id=invitation.id,
        email=invitation.email,
        role=invitation.role,
        invited_by_user_id=invitation.invited_by_user_id,
        expires_at=invitation.expires_at,
        accepted_at=invitation.accepted_at,
        revoked_at=invitation.revoked_at,
        created_at=invitation.created_at,
        status=invitation_status(invitation),
        invite_url=invite_url,
        token_preview=token_preview,
    )
