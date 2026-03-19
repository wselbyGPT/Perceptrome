from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ...deps import get_db, require_role
from ...models import User
from ...schemas import AuditEventListOut
from ...services import audit_service

router = APIRouter(prefix='/api/admin/audit', tags=['admin-audit'])


@router.get('', response_model=AuditEventListOut)
def list_audit_events(
    limit: int = Query(default=100, ge=1, le=500),
    action: str | None = Query(default=None),
    actor: str | None = Query(default=None),
    target: str | None = Query(default=None),
    search: str | None = Query(default=None),
    _admin: User = Depends(require_role('admin')),
    db: Session = Depends(get_db),
):
    return audit_service.list_audit_events(db, limit=limit, action=action, actor=actor, target=target, search=search)
