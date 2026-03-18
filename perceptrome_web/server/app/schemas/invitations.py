from datetime import datetime

from pydantic import BaseModel, EmailStr


class UserInvitationOut(BaseModel):
    id: str
    email: EmailStr
    role: str
    invited_by_user_id: str
    expires_at: datetime
    accepted_at: datetime | None = None
    revoked_at: datetime | None = None
    created_at: datetime
