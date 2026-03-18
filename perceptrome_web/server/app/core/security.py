import hashlib
import secrets

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHash, VerifyMismatchError

_password_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return _password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _password_hasher.verify(password_hash, password)
    except (VerifyMismatchError, InvalidHash):
        return False


def make_session_token() -> str:
    return secrets.token_urlsafe(48)


def hash_session_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def password_complexity_error(password: str) -> str | None:
    if len(password) < 12:
        return "Password must be at least 12 characters long"
    if not any(ch.isupper() for ch in password):
        return "Password must include at least one uppercase letter"
    if not any(ch.islower() for ch in password):
        return "Password must include at least one lowercase letter"
    if not any(ch.isdigit() for ch in password):
        return "Password must include at least one number"
    return None
