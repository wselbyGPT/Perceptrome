from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...deps import get_db, require_role
from ...models import User
from ...schemas import AdminCreateUserRequest, AdminUserOut
from ...services import user_service

router = APIRouter(prefix='/api/admin/users', tags=['admin-users'])


@router.get('', response_model=list[AdminUserOut])
def list_users(_admin: User = Depends(require_role('admin')), db: Session = Depends(get_db)):
    return user_service.list_admin_users(db)


@router.post('', response_model=AdminUserOut)
def create_user_admin(payload: AdminCreateUserRequest, _admin: User = Depends(require_role('admin')), db: Session = Depends(get_db)):
    return AdminUserOut.from_model(user_service.create_admin_user(db, email=payload.email, password=payload.password, username=payload.username, role=payload.role, is_active=payload.is_active, must_change_password=payload.must_change_password))
