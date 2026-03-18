from datetime import timedelta
import smtplib
import sys
from email.message import EmailMessage
from urllib.parse import urlencode

from fastapi import HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth_rate_limit import login_attempt_store as default_login_attempt_store
from ..core.config import settings
from ..core.security import (
    hash_password,
    hash_session_token,
    make_session_token,
    password_complexity_error,
    verify_password,
)
from ..models import AuthToken, User
from . import audit_service, session_service, user_service


def _main_module():
    return sys.modules.get("app.main")


def current_login_attempt_store():
    main = _main_module()
    return getattr(main, "login_attempt_store", default_login_attempt_store) if main else default_login_attempt_store


def send_email(recipient: str, subject: str, body: str) -> None:
    if settings.mail_provider.lower() == "smtp":
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = settings.mail_from_email
        msg["To"] = recipient
        msg.set_content(body)
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as smtp:
            if settings.smtp_use_tls:
                smtp.starttls()
            if settings.smtp_username:
                smtp.login(settings.smtp_username, settings.smtp_password or "")
            smtp.send_message(msg)
        return
    print(f"[auth] {subject} to {recipient}: {body}")


def send_verification_email(recipient: str, raw_token: str) -> None:
    main = _main_module()
    override = getattr(main, "_send_verification_email", None) if main else None
    if callable(override) and override is not send_verification_email:
        override(recipient, raw_token)
        return
    link = f"{settings.email_verification_base_url}?{urlencode({'token': raw_token})}"
    send_email(
        recipient,
        "Verify your Perceptrome account",
        "Welcome to Perceptrome!\n\n"
        "Use the following link to verify your email:\n"
        f"{link}\n\n"
        "If you did not sign up, you can ignore this email.",
    )


def send_password_reset_email(recipient: str, raw_token: str) -> None:
    main = _main_module()
    override = getattr(main, "_send_password_reset_email", None) if main else None
    if callable(override) and override is not send_password_reset_email:
        override(recipient, raw_token)
        return
    link = f"{settings.password_reset_base_url}?{urlencode({'token': raw_token})}"
    send_email(
        recipient,
        "Reset your Perceptrome password",
        "We received a request to reset your Perceptrome password.\n\n"
        "Use the following link to reset your password:\n"
        f"{link}\n\n"
        "If you did not request this, you can ignore this email.",
    )


def issue_auth_token(db: Session, user: User, *, purpose: str, ttl_minutes: int) -> str:
    raw = make_session_token()
    token = AuthToken(
        user_id=user.id,
        purpose=purpose,
        token_hash=hash_session_token(raw),
        expires_at=audit_service.utcnow() + timedelta(minutes=ttl_minutes),
    )
    db.add(token)
    db.commit()
    return raw


def issue_email_verification_token(db: Session, user: User) -> str:
    raw = issue_auth_token(
        db,
        user,
        purpose="email_verification",
        ttl_minutes=settings.email_verification_token_ttl_minutes,
    )
    user.email_verification_sent_at = audit_service.utcnow()
    db.commit()
    return raw


def issue_password_reset_token(db: Session, user: User) -> str:
    return issue_auth_token(
        db,
        user,
        purpose="password_reset",
        ttl_minutes=settings.password_reset_token_ttl_minutes,
    )


def register_user(db: Session, *, email: str, password: str, username: str | None) -> User:
    if not settings.allow_self_register:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Self registration is disabled")
    user = user_service.create_registered_user(db, email=email, password=password, username=username)
    raw_token = issue_email_verification_token(db, user)
    send_verification_email(user.email, raw_token)
    db.refresh(user)
    return user


def verify_email_token(db: Session, token_value: str) -> str:
    token = db.execute(
        select(AuthToken)
        .where(AuthToken.purpose == "email_verification")
        .where(AuthToken.token_hash == hash_session_token(token_value))
    ).scalar_one_or_none()
    if not token:
        audit_service.metric_inc("verification_failures")
        raise HTTPException(status_code=400, detail="Invalid verification token")
    if token.used_at is not None:
        audit_service.metric_inc("verification_failures")
        raise HTTPException(status_code=400, detail="Verification token already used")
    if token.expires_at <= audit_service.utcnow():
        audit_service.metric_inc("verification_failures")
        raise HTTPException(status_code=400, detail="Verification token expired")
    user = db.get(User, token.user_id)
    if not user:
        audit_service.metric_inc("verification_failures")
        raise HTTPException(status_code=400, detail="Invalid verification token")

    token.used_at = audit_service.utcnow()
    if user.email_verified_at is not None:
        db.commit()
        return "Email already verified"

    user.email_verified_at = audit_service.utcnow()
    db.commit()
    return "Email verified"


def resend_verification(db: Session, email: str) -> str:
    user = db.execute(select(User).where(User.email == email.lower().strip())).scalar_one_or_none()
    if not user:
        return "If this email is registered, a verification email has been sent"
    if user.email_verified_at is not None:
        return "Email already verified"
    if user.email_verification_sent_at is not None:
        elapsed = (audit_service.utcnow() - user.email_verification_sent_at).total_seconds()
        if elapsed < settings.email_verification_resend_cooldown_seconds:
            raise_auth_429(
                max(1, int(settings.email_verification_resend_cooldown_seconds - elapsed)),
                "verification_resend_cooldown",
            )
    raw_token = issue_email_verification_token(db, user)
    send_verification_email(user.email, raw_token)
    return "Verification email sent"


def forgot_password(db: Session, email: str) -> str:
    user = db.execute(select(User).where(User.email == email.lower().strip())).scalar_one_or_none()
    if user:
        raw_token = issue_password_reset_token(db, user)
        send_password_reset_email(user.email, raw_token)
    return "If this email is registered, a password reset email has been sent"


def reset_password(db: Session, *, token_value: str, new_password: str) -> str:
    token = db.execute(
        select(AuthToken)
        .where(AuthToken.purpose == "password_reset")
        .where(AuthToken.token_hash == hash_session_token(token_value))
    ).scalar_one_or_none()
    if not token:
        raise HTTPException(status_code=400, detail="Invalid password reset token")
    if token.used_at is not None:
        raise HTTPException(status_code=400, detail="Password reset token already used")

    now = audit_service.utcnow()
    if token.expires_at <= now:
        raise HTTPException(status_code=400, detail="Password reset token expired")

    user = db.get(User, token.user_id)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid password reset token")

    complexity_error = password_complexity_error(new_password)
    if complexity_error:
        raise HTTPException(status_code=400, detail=complexity_error)

    user.password_hash = hash_password(new_password)
    user.must_change_password = False
    token.used_at = now
    db.commit()
    session_service.revoke_all_active_sessions(db, user)
    return "Password reset successful"


def login_user(db: Session, *, email: str, password: str, request: Request) -> User:
    email = email.lower().strip()
    ip = request.client.host if request.client else "unknown"
    now = audit_service.utcnow()
    rl_status = current_login_attempt_store().check_and_record(db=db, ip=ip, email=email, now=now)
    if rl_status.limited:
        audit_service.structured_auth_log(
            "login_rate_limited",
            ip=ip,
            email=email,
            scope=rl_status.scope,
            retry_after_seconds=rl_status.retry_after_seconds,
        )
        audit_service.metric_inc("lockouts")
        raise_auth_429(rl_status.retry_after_seconds, f"rate_limit_{rl_status.scope}")

    user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
    if user and user.locked_until and user.locked_until > now:
        retry_after = max(1, int((user.locked_until - now).total_seconds()))
        audit_service.structured_auth_log(
            "login_user_locked",
            ip=ip,
            email=email,
            user_id=user.id,
            retry_after_seconds=retry_after,
        )
        audit_service.metric_inc("lockouts")
        raise_auth_429(retry_after, "user_locked")

    if not user or not user.is_active or not verify_password(password, user.password_hash):
        if user:
            user.failed_login_count = (user.failed_login_count or 0) + 1
            if user.failed_login_count >= settings.login_lockout_threshold:
                user.locked_until = now + timedelta(seconds=settings.login_lockout_seconds)
                audit_service.metric_inc("lockouts")
            elif user.failed_login_count >= 2:
                backoff = min(
                    settings.login_backoff_max_seconds,
                    settings.login_backoff_base_seconds * (2 ** (user.failed_login_count - 2)),
                )
                user.locked_until = now + timedelta(seconds=backoff)
            db.commit()
            audit_service.structured_auth_log(
                "login_failed",
                ip=ip,
                email=email,
                user_id=user.id,
                failed_login_count=user.failed_login_count,
                locked_until=user.locked_until.isoformat() if user.locked_until else None,
            )
        else:
            audit_service.structured_auth_log("login_failed", ip=ip, email=email, user_id=None)
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if user.email_verified_at is None:
        audit_service.structured_auth_log("login_email_unverified", ip=ip, email=email, user_id=user.id)
        raise HTTPException(status_code=403, detail="Email verification required")

    if user.failed_login_count or user.locked_until is not None:
        user.failed_login_count = 0
        user.locked_until = None
        db.commit()
        audit_service.metric_inc("resets")
        audit_service.structured_auth_log("login_failure_state_reset", ip=ip, email=email, user_id=user.id)

    return user


def change_password(db: Session, *, user: User, current_password: str, new_password: str, current_cookie: str | None) -> str:
    if not verify_password(current_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    if current_password == new_password:
        raise HTTPException(status_code=400, detail="New password must be different")
    complexity_error = password_complexity_error(new_password)
    if complexity_error:
        raise HTTPException(status_code=400, detail=complexity_error)

    user.password_hash = hash_password(new_password)
    user.must_change_password = False
    db.commit()
    session_service.revoke_other_sessions(db, user, current_cookie=current_cookie)
    return "Password changed"


def raise_auth_429(retry_after_seconds: int, reason: str) -> None:
    detail = {
        "message": "Too many authentication attempts. Please retry later.",
        "retry_after_seconds": retry_after_seconds,
        "reason": reason,
    }
    raise HTTPException(status_code=429, detail=detail, headers={"Retry-After": str(retry_after_seconds)})
