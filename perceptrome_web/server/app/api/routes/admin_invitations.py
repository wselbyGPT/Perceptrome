from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session

from ...deps import get_db, require_role
from ...models import User
from ...schemas import UserInvitationActionOut, UserInvitationCreateRequest, UserInvitationListOut, UserInvitationOut
from ...services import audit_service, invitation_service

router = APIRouter(prefix='/api/admin/invitations', tags=['admin-invitations'])


@router.get('', response_model=UserInvitationListOut)
def list_invitations(
    status: str | None = Query(default=None),
    role: str | None = Query(default=None),
    search: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=500),
    _admin: User = Depends(require_role('admin')),
    db: Session = Depends(get_db),
):
    return invitation_service.list_invitations(db, status_filter=status, role=role, search=search, limit=limit)


@router.post('', response_model=UserInvitationOut)
def create_invitation(payload: UserInvitationCreateRequest, request: Request, admin: User = Depends(require_role('admin')), db: Session = Depends(get_db)):
    invitation, raw_token = invitation_service.create_invitation(db, email=payload.email, role=payload.role, invited_by_user_id=admin.id, reissue=payload.reissue)
    audit_service.create_audit_event(db, action=audit_service.AuditActions.INVITE_CREATED, actor_user_id=admin.id, request=request, metadata={'invitation_id': invitation.id, 'email': invitation.email, 'role': invitation.role, 'reissue': payload.reissue, 'status': invitation_service.invitation_status(invitation)})
    return invitation_service.invitation_out(invitation, raw_token=raw_token)


@router.post('/{invitation_id}/revoke', response_model=UserInvitationActionOut)
def revoke_invitation(invitation_id: str, request: Request, admin: User = Depends(require_role('admin')), db: Session = Depends(get_db)):
    invitation = invitation_service.revoke_invitation(db, invitation_id=invitation_id)
    audit_service.create_audit_event(db, action=audit_service.AuditActions.INVITE_REVOKED, actor_user_id=admin.id, request=request, metadata={'invitation_id': invitation.id, 'email': invitation.email})
    return UserInvitationActionOut(message='Invitation revoked', invitation=invitation_service.invitation_out(invitation))
