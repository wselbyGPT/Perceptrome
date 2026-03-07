# server/app/schemas.py
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=256)
    username: str | None = Field(default=None, min_length=3, max_length=64)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)


class VerifyEmailRequest(BaseModel):
    token: str = Field(min_length=8, max_length=512)


class ResendVerificationRequest(BaseModel):
    email: EmailStr


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str = Field(min_length=8, max_length=512)
    new_password: str = Field(min_length=8, max_length=256)


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=12, max_length=256)


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


class MessageOut(BaseModel):
    message: str


class RunStartRequest(BaseModel):
    config: dict = Field(default_factory=dict)


class RunArtifactOut(BaseModel):
    id: int
    phase: str | None = None
    path: str
    label: str | None = None
    download_url: str
    created_at: datetime


class RunOut(BaseModel):
    run_id: str
    user_id: str
    kind: str
    state: str
    message: str | None = None
    config: dict = Field(default_factory=dict)
    result: dict | None = None
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    artifacts: list[RunArtifactOut] = Field(default_factory=list)
