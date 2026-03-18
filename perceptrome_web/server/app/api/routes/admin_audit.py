from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...deps import get_db, require_role
from ...models import User
from ...schemas import AuditEventOut
from ...services import audit_service

router = APIRouter(prefix='/api/admin/audit', tags=['admin-audit'])


@router.get('', response_model=list[AuditEventOut])
def list_audit_events(_admin: User = Depends(require_role('admin')), db: Session = Depends(get_db), limit: int = 100):
    return audit_service.list_audit_events(db, limit=limit)
