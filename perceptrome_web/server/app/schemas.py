# server/app/schemas.py
from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=256)
    username: str | None = Field(default=None, min_length=3, max_length=64)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8, max_length=256)


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

    @classmethod
    def from_model(cls, u):
        return cls(
            id=u.id,
            email=u.email,
            username=u.username,
            role=u.role,
            is_active=u.is_active,
        )


class MessageOut(BaseModel):
    message: str
