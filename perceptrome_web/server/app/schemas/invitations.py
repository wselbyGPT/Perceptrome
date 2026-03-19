from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class UserInvitationCreateRequest(BaseModel):
    email: EmailStr
    role: str = Field(default="user")
    reissue: bool = True


class UserInvitationOut(BaseModel):
    id: str
    email: EmailStr
    role: str
    invited_by_user_id: str
    expires_at: datetime
    accepted_at: datetime | None = None
    revoked_at: datetime | None = None
    created_at: datetime
    status: str
    invite_url: str | None = None
    token_preview: str | None = None


class UserInvitationActionOut(BaseModel):
    message: str
    invitation: UserInvitationOut


class UserInvitationListOut(BaseModel):
    invitations: list[UserInvitationOut]
    total: int
