from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class AdminCreateUserRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=256)
    username: str | None = Field(default=None, min_length=3, max_length=64)
    role: str
    is_active: bool = True
    must_change_password: bool = True


class UserOut(BaseModel):
    id: str
    email: EmailStr
    username: str | None
    role: str
    is_active: bool
    must_change_password: bool
    email_verified_at: datetime | None

    @classmethod
    def from_model(cls, u):
        return cls(
            id=u.id,
            email=u.email,
            username=u.username,
            role=u.role,
            is_active=u.is_active,
            must_change_password=u.must_change_password,
            email_verified_at=u.email_verified_at,
        )


class AdminUserOut(UserOut):
    created_at: datetime
    last_login_at: datetime | None
    locked_until: datetime | None
    failed_login_count: int
    account_state: str
    is_locked: bool

    @classmethod
    def from_model(cls, u):
        is_locked = bool(u.locked_until and u.locked_until > datetime.utcnow())
        account_state = "active" if u.is_active else "suspended"
        return cls(
            **UserOut.from_model(u).model_dump(),
            created_at=u.created_at,
            last_login_at=u.last_login_at,
            locked_until=u.locked_until,
            failed_login_count=u.failed_login_count or 0,
            account_state=account_state,
            is_locked=is_locked,
        )
