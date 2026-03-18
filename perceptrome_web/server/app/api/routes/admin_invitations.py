from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...deps import get_db, require_role
from ...models import User, UserInvitation
from ...schemas import UserInvitationOut

router = APIRouter(prefix='/api/admin/invitations', tags=['admin-invitations'])


@router.get('', response_model=list[UserInvitationOut])
def list_invitations(_admin: User = Depends(require_role('admin')), db: Session = Depends(get_db)):
    rows = db.execute(select(UserInvitation).order_by(UserInvitation.created_at.desc())).scalars().all()
    return [UserInvitationOut.model_validate(row, from_attributes=True) for row in rows]
