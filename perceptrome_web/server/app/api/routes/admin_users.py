from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session

from ...deps import get_db, require_role
from ...models import User
from ...schemas import AdminCreateUserRequest, AdminUserActionOut, AdminUserListOut, AdminUserOut, AdminUserUpdateRequest
from ...services import audit_service, user_service

router = APIRouter(prefix='/api/admin/users', tags=['admin-users'])


@router.get('', response_model=AdminUserListOut)
def list_users(
    search: str | None = Query(default=None),
    role: str | None = Query(default=None),
    state: str | None = Query(default=None),
    verification: str | None = Query(default=None),
    must_change_password: bool | None = Query(default=None),
    _admin: User = Depends(require_role('admin')),
    db: Session = Depends(get_db),
):
    users, total = user_service.list_admin_users(
        db,
        search=search,
        role=role,
        state=state,
        verification=verification,
        must_change_password=must_change_password,
    )
    return AdminUserListOut(users=users, total=total)


@router.post('', response_model=AdminUserOut)
def create_user_admin(payload: AdminCreateUserRequest, _admin: User = Depends(require_role('admin')), db: Session = Depends(get_db)):
    return AdminUserOut.from_model(user_service.create_admin_user(db, email=payload.email, password=payload.password, username=payload.username, role=payload.role, is_active=payload.is_active, must_change_password=payload.must_change_password))


@router.patch('/{user_id}', response_model=AdminUserOut)
def update_user_admin(payload: AdminUserUpdateRequest, user_id: str, request: Request, admin: User = Depends(require_role('admin')), db: Session = Depends(get_db)):
    user = user_service.update_admin_user(db, user_id=user_id, username=payload.username, role=payload.role, is_active=payload.is_active, must_change_password=payload.must_change_password)
    audit_service.create_audit_event(db, action='admin.user_updated', actor_user_id=admin.id, target_user_id=user.id, request=request, metadata=payload.model_dump())
    return AdminUserOut.from_model(user)


@router.post('/{user_id}/suspend', response_model=AdminUserActionOut)
def suspend_user_admin(user_id: str, request: Request, admin: User = Depends(require_role('admin')), db: Session = Depends(get_db)):
    user, revoked_count = user_service.suspend_user(db, user_id=user_id)
    audit_service.create_audit_event(db, action='admin.user_suspended', actor_user_id=admin.id, target_user_id=user.id, request=request, metadata={'revoked_session_count': revoked_count})
    return AdminUserActionOut(message='User suspended', user=AdminUserOut.from_model(user), revoked_session_count=revoked_count)


@router.post('/{user_id}/activate', response_model=AdminUserActionOut)
def activate_user_admin(user_id: str, request: Request, admin: User = Depends(require_role('admin')), db: Session = Depends(get_db)):
    user = user_service.activate_user(db, user_id=user_id)
    audit_service.create_audit_event(db, action='admin.user_activated', actor_user_id=admin.id, target_user_id=user.id, request=request)
    return AdminUserActionOut(message='User activated', user=AdminUserOut.from_model(user), revoked_session_count=0)


@router.post('/{user_id}/force-reset', response_model=AdminUserActionOut)
def force_reset_user_admin(user_id: str, request: Request, admin: User = Depends(require_role('admin')), db: Session = Depends(get_db)):
    user, revoked_count = user_service.admin_force_reset(db, user_id=user_id)
    audit_service.create_audit_event(db, action='admin.user_force_reset', actor_user_id=admin.id, target_user_id=user.id, request=request, metadata={'revoked_session_count': revoked_count})
    return AdminUserActionOut(message='Password reset enforced', user=AdminUserOut.from_model(user), revoked_session_count=revoked_count)


@router.post('/{user_id}/resend-verification', response_model=AdminUserActionOut)
def resend_verification_user_admin(user_id: str, request: Request, admin: User = Depends(require_role('admin')), db: Session = Depends(get_db)):
    user = user_service.resend_user_verification(db, user_id=user_id)
    audit_service.create_audit_event(db, action='admin.user_verification_resent', actor_user_id=admin.id, target_user_id=user.id, request=request)
    return AdminUserActionOut(message='Verification email resent', user=AdminUserOut.from_model(user), revoked_session_count=0)


@router.post('/{user_id}/revoke-sessions', response_model=AdminUserActionOut)
def revoke_sessions_user_admin(user_id: str, request: Request, admin: User = Depends(require_role('admin')), db: Session = Depends(get_db)):
    user, revoked_count = user_service.revoke_user_sessions(db, user_id=user_id)
    audit_service.create_audit_event(db, action='admin.user_sessions_revoked', actor_user_id=admin.id, target_user_id=user.id, request=request, metadata={'revoked_session_count': revoked_count})
    return AdminUserActionOut(message=f'Revoked {revoked_count} active session(s)', user=AdminUserOut.from_model(user), revoked_session_count=revoked_count)
