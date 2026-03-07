# server/app/main.py
import asyncio
from datetime import datetime, timedelta
import json
import logging
import smtplib
from collections import Counter
from email.message import EmailMessage
from urllib.parse import urlencode

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, select, text
from sqlalchemy.orm import Session

from .config import settings
from .auth_rate_limit import login_attempt_store
from .db import Base, engine, SessionLocal
from .deps import get_db, get_current_user, get_current_user_strict, require_role
from .models import AuthToken, User, UserSession
from .schemas import (
    RegisterRequest,
    LoginRequest,
    VerifyEmailRequest,
    ResendVerificationRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    ChangePasswordRequest,
    AdminCreateUserRequest,
    UserOut,
    MessageOut,
)
from perceptrome.jobs import JobEngine, JobEvent, JobSpec

from .security import (
    hash_password,
    verify_password,
    make_session_token,
    hash_session_token,
    password_complexity_error,
)

app = FastAPI(title=settings.app_name)

origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

_auth_logger = logging.getLogger("perceptrome.auth")
_auth_metrics: Counter[str] = Counter()


def _utcnow() -> datetime:
    return datetime.utcnow()


def _set_session_cookie(response: Response, raw_token: str):
    response.set_cookie(
        key=settings.session_cookie_name,
        value=raw_token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        max_age=settings.session_ttl_hours * 3600,
        path="/",
        domain=settings.cookie_domain,
    )


def _clear_session_cookie(response: Response):
    response.delete_cookie(
        key=settings.session_cookie_name,
        path="/",
        domain=settings.cookie_domain,
    )


def _issue_session(db: Session, user: User, request: Request) -> str:
    raw = make_session_token()
    token_hash = hash_session_token(raw)
    expires_at = _utcnow() + timedelta(hours=settings.session_ttl_hours)

    sess = UserSession(
        user_id=user.id,
        token_hash=token_hash,
        expires_at=expires_at,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )
    db.add(sess)
    user.last_login_at = _utcnow()
    db.commit()
    return raw


def _build_verification_link(raw_token: str) -> str:
    return f"{settings.email_verification_base_url}?{urlencode({'token': raw_token})}"


def _build_password_reset_link(raw_token: str) -> str:
    return f"{settings.password_reset_base_url}?{urlencode({'token': raw_token})}"


def _send_email(recipient: str, subject: str, body: str):
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


def _send_verification_email(recipient: str, raw_token: str):
    link = _build_verification_link(raw_token)
    _send_email(
        recipient,
        "Verify your Perceptrome account",
        "Welcome to Perceptrome!\n\n"
        "Use the following link to verify your email:\n"
        f"{link}\n\n"
        "If you did not sign up, you can ignore this email.",
    )


def _send_password_reset_email(recipient: str, raw_token: str):
    link = _build_password_reset_link(raw_token)
    _send_email(
        recipient,
        "Reset your Perceptrome password",
        "We received a request to reset your Perceptrome password.\n\n"
        "Use the following link to reset your password:\n"
        f"{link}\n\n"
        "If you did not request this, you can ignore this email.",
    )


def _issue_auth_token(db: Session, user: User, purpose: str, ttl_minutes: int) -> str:
    raw = make_session_token()
    token = AuthToken(
        user_id=user.id,
        purpose=purpose,
        token_hash=hash_session_token(raw),
        expires_at=_utcnow() + timedelta(minutes=ttl_minutes),
    )
    db.add(token)
    db.commit()
    return raw


def _issue_email_verification_token(db: Session, user: User) -> str:
    raw = _issue_auth_token(db, user, purpose="email_verification", ttl_minutes=settings.email_verification_token_ttl_minutes)
    user.email_verification_sent_at = _utcnow()
    db.commit()
    return raw


def _issue_password_reset_token(db: Session, user: User) -> str:
    return _issue_auth_token(db, user, purpose="password_reset", ttl_minutes=settings.password_reset_token_ttl_minutes)


def _revoke_session_by_cookie(db: Session, raw_cookie: str | None):
    if not raw_cookie:
        return
    token_hash = hash_session_token(raw_cookie)
    stmt = (
        select(UserSession)
        .where(UserSession.token_hash == token_hash)
        .where(UserSession.revoked_at.is_(None))
    )
    sess = db.execute(stmt).scalar_one_or_none()
    if sess:
        sess.revoked_at = _utcnow()
        db.commit()


def _metric_inc(name: str):
    _auth_metrics[name] += 1


def _structured_auth_log(event: str, **fields):
    payload = {"event": event, **fields}
    _auth_logger.warning(json.dumps(payload, sort_keys=True))


def _raise_auth_429(retry_after_seconds: int, reason: str):
    detail = {
        "message": "Too many authentication attempts. Please retry later.",
        "retry_after_seconds": retry_after_seconds,
        "reason": reason,
    }
    raise HTTPException(status_code=429, detail=detail, headers={"Retry-After": str(retry_after_seconds)})


def _ensure_users_columns():
    insp = inspect(engine)
    if "users" not in insp.get_table_names():
        return

    cols = {c["name"] for c in insp.get_columns("users")}
    statements = {
        "must_change_password": "ALTER TABLE users ADD COLUMN must_change_password BOOLEAN NOT NULL DEFAULT 0",
        "email_verified_at": "ALTER TABLE users ADD COLUMN email_verified_at DATETIME NULL",
        "email_verification_sent_at": "ALTER TABLE users ADD COLUMN email_verification_sent_at DATETIME NULL",
        "failed_login_count": "ALTER TABLE users ADD COLUMN failed_login_count INTEGER NOT NULL DEFAULT 0",
        "locked_until": "ALTER TABLE users ADD COLUMN locked_until DATETIME NULL",
    }

    for name, stmt in statements.items():
        if name not in cols:
            with engine.begin() as conn:
                conn.execute(text(stmt))
            print(f"[auth] migrated users.{name}")


def _bootstrap_admin():
    if not settings.bootstrap_admin_email or not settings.bootstrap_admin_password:
        return

    db = SessionLocal()
    try:
        email = settings.bootstrap_admin_email.lower().strip()
        existing = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
        if existing:
            return

        admin = User(
            email=email,
            password_hash=hash_password(settings.bootstrap_admin_password),
            role="admin",
            is_active=True,
            must_change_password=True,
            email_verified_at=_utcnow(),
        )
        db.add(admin)
        db.commit()
        print(f"[auth] bootstrapped admin user: {admin.email}")
    finally:
        db.close()


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    _ensure_users_columns()
    _bootstrap_admin()


@app.get("/api/health")
def health():
    return {"ok": True, "service": "perceptrome-api"}


@app.post("/api/auth/register", response_model=UserOut)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    if not settings.allow_self_register:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Self registration is disabled")

    email = payload.email.lower().strip()
    username = payload.username.strip() if payload.username else None

    if db.execute(select(User).where(User.email == email)).scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    if username and db.execute(select(User).where(User.username == username)).scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Username already taken")

    user = User(
        email=email,
        username=username,
        password_hash=hash_password(payload.password),
        role="user",
        is_active=True,
        must_change_password=False,
        email_verified_at=None,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    raw_token = _issue_email_verification_token(db, user)
    _send_verification_email(user.email, raw_token)

    db.refresh(user)
    return UserOut.from_model(user)


@app.post("/api/auth/verify-email", response_model=MessageOut)
def verify_email(payload: VerifyEmailRequest, db: Session = Depends(get_db)):
    token_hash = hash_session_token(payload.token)
    token = db.execute(
        select(AuthToken)
        .where(AuthToken.purpose == "email_verification")
        .where(AuthToken.token_hash == token_hash)
    ).scalar_one_or_none()

    if not token:
        _metric_inc("verification_failures")
        raise HTTPException(status_code=400, detail="Invalid verification token")

    if token.used_at is not None:
        _metric_inc("verification_failures")
        raise HTTPException(status_code=400, detail="Verification token already used")

    if token.expires_at <= _utcnow():
        _metric_inc("verification_failures")
        raise HTTPException(status_code=400, detail="Verification token expired")

    user = db.get(User, token.user_id)
    if not user:
        _metric_inc("verification_failures")
        raise HTTPException(status_code=400, detail="Invalid verification token")

    token.used_at = _utcnow()
    if user.email_verified_at is not None:
        db.commit()
        return MessageOut(message="Email already verified")

    user.email_verified_at = _utcnow()
    db.commit()
    return MessageOut(message="Email verified")


@app.post("/api/auth/resend-verification", response_model=MessageOut)
def resend_verification(payload: ResendVerificationRequest, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()
    user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()

    if not user:
        return MessageOut(message="If this email is registered, a verification email has been sent")

    if user.email_verified_at is not None:
        return MessageOut(message="Email already verified")

    if user.email_verification_sent_at is not None:
        elapsed = (_utcnow() - user.email_verification_sent_at).total_seconds()
        if elapsed < settings.email_verification_resend_cooldown_seconds:
            _raise_auth_429(max(1, int(settings.email_verification_resend_cooldown_seconds - elapsed)), "verification_resend_cooldown")

    raw_token = _issue_email_verification_token(db, user)
    _send_verification_email(user.email, raw_token)
    return MessageOut(message="Verification email sent")


@app.post("/api/auth/forgot-password", response_model=MessageOut)
def forgot_password(payload: ForgotPasswordRequest, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()
    user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()

    if user:
        raw_token = _issue_password_reset_token(db, user)
        _send_password_reset_email(user.email, raw_token)

    return MessageOut(message="If this email is registered, a password reset email has been sent")


@app.post("/api/auth/reset-password", response_model=MessageOut)
def reset_password(payload: ResetPasswordRequest, db: Session = Depends(get_db)):
    token_hash = hash_session_token(payload.token)
    token = db.execute(
        select(AuthToken)
        .where(AuthToken.purpose == "password_reset")
        .where(AuthToken.token_hash == token_hash)
    ).scalar_one_or_none()

    if not token:
        raise HTTPException(status_code=400, detail="Invalid password reset token")

    if token.used_at is not None:
        raise HTTPException(status_code=400, detail="Password reset token already used")

    now = _utcnow()
    if token.expires_at <= now:
        raise HTTPException(status_code=400, detail="Password reset token expired")

    user = db.get(User, token.user_id)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid password reset token")

    complexity_error = password_complexity_error(payload.new_password)
    if complexity_error:
        raise HTTPException(status_code=400, detail=complexity_error)

    user.password_hash = hash_password(payload.new_password)
    user.must_change_password = False
    token.used_at = now

    active_sessions = db.execute(
        select(UserSession)
        .where(UserSession.user_id == user.id)
        .where(UserSession.revoked_at.is_(None))
        .where(UserSession.expires_at > now)
    ).scalars().all()
    for sess in active_sessions:
        sess.revoked_at = now

    db.commit()
    return MessageOut(message="Password reset successful")


@app.post("/api/auth/login", response_model=UserOut)
def login(payload: LoginRequest, request: Request, response: Response, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()
    ip = request.client.host if request.client else "unknown"
    now = _utcnow()
    rl_status = login_attempt_store.check_and_record(db=db, ip=ip, email=email, now=now)
    if rl_status.limited:
        _structured_auth_log("login_rate_limited", ip=ip, email=email, scope=rl_status.scope, retry_after_seconds=rl_status.retry_after_seconds)
        _metric_inc("lockouts")
        _raise_auth_429(rl_status.retry_after_seconds, f"rate_limit_{rl_status.scope}")

    user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()

    if user and user.locked_until and user.locked_until > now:
        retry_after = max(1, int((user.locked_until - now).total_seconds()))
        _structured_auth_log("login_user_locked", ip=ip, email=email, user_id=user.id, retry_after_seconds=retry_after)
        _metric_inc("lockouts")
        _raise_auth_429(retry_after, "user_locked")

    if not user or not user.is_active or not verify_password(payload.password, user.password_hash):
        if user:
            user.failed_login_count = (user.failed_login_count or 0) + 1
            if user.failed_login_count >= settings.login_lockout_threshold:
                user.locked_until = now + timedelta(seconds=settings.login_lockout_seconds)
                _metric_inc("lockouts")
            elif user.failed_login_count >= 2:
                backoff = min(
                    settings.login_backoff_max_seconds,
                    settings.login_backoff_base_seconds * (2 ** (user.failed_login_count - 2)),
                )
                user.locked_until = now + timedelta(seconds=backoff)
            db.commit()
            _structured_auth_log(
                "login_failed",
                ip=ip,
                email=email,
                user_id=user.id,
                failed_login_count=user.failed_login_count,
                locked_until=user.locked_until.isoformat() if user.locked_until else None,
            )
        else:
            _structured_auth_log("login_failed", ip=ip, email=email, user_id=None)
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if user.email_verified_at is None:
        _structured_auth_log("login_email_unverified", ip=ip, email=email, user_id=user.id)
        raise HTTPException(status_code=403, detail="Email verification required")

    if user.failed_login_count or user.locked_until is not None:
        user.failed_login_count = 0
        user.locked_until = None
        db.commit()
        _metric_inc("resets")
        _structured_auth_log("login_failure_state_reset", ip=ip, email=email, user_id=user.id)

    _revoke_session_by_cookie(db, request.cookies.get(settings.session_cookie_name))

    raw_session = _issue_session(db, user, request)
    _set_session_cookie(response, raw_session)
    return UserOut.from_model(user)


@app.post("/api/auth/logout", response_model=MessageOut)
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    _revoke_session_by_cookie(db, request.cookies.get(settings.session_cookie_name))
    _clear_session_cookie(response)
    return MessageOut(message="Logged out")


@app.get("/api/auth/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return UserOut.from_model(user)


@app.post("/api/auth/change-password", response_model=MessageOut)
def change_password(
    payload: ChangePasswordRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not verify_password(payload.current_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    if payload.current_password == payload.new_password:
        raise HTTPException(status_code=400, detail="New password must be different")

    complexity_error = password_complexity_error(payload.new_password)
    if complexity_error:
        raise HTTPException(status_code=400, detail=complexity_error)

    now = _utcnow()
    user.password_hash = hash_password(payload.new_password)
    user.must_change_password = False

    current_cookie = request.cookies.get(settings.session_cookie_name)
    current_session_hash = hash_session_token(current_cookie) if current_cookie else None
    active_sessions = db.execute(
        select(UserSession)
        .where(UserSession.user_id == user.id)
        .where(UserSession.revoked_at.is_(None))
        .where(UserSession.expires_at > now)
    ).scalars().all()
    for sess in active_sessions:
        if sess.token_hash != current_session_hash:
            sess.revoked_at = now

    db.commit()
    return MessageOut(message="Password changed")


@app.post("/api/runs/start")
def start_run(user: User = Depends(get_current_user_strict)):
    spec = JobSpec(kind="generate_plasmid", config_path="config/stream_config.yaml", params={"length_bp": 512, "output": "generated/web_api_run.fasta"})
    result = JobEngine().run(spec)
    return {"ok": result.ok, "message": result.message, "user_id": user.id, "role": user.role, "data": result.data}


@app.get("/api/admin/users")
def list_users(
    _admin: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    users = db.execute(select(User).order_by(User.created_at.desc())).scalars().all()
    return [UserOut.from_model(u).model_dump() for u in users]


@app.post("/api/admin/users", response_model=UserOut)
def create_user_admin(
    payload: AdminCreateUserRequest,
    _admin: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    email = payload.email.lower().strip()
    username = payload.username.strip() if payload.username else None
    role = payload.role.strip().lower()

    if role not in {"admin", "user"}:
        raise HTTPException(status_code=400, detail="Invalid role")

    if db.execute(select(User).where(User.email == email)).scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    if username and db.execute(select(User).where(User.username == username)).scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Username already taken")

    user = User(
        email=email,
        username=username,
        password_hash=hash_password(payload.password),
        role=role,
        is_active=payload.is_active,
        must_change_password=payload.must_change_password,
        email_verified_at=_utcnow(),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return UserOut.from_model(user)


def _ws_auth_user(websocket: WebSocket, db: Session) -> User | None:
    raw = websocket.cookies.get(settings.session_cookie_name)
    if not raw:
        return None

    token_hash = hash_session_token(raw)
    now = _utcnow()

    sess = (
        db.execute(
            select(UserSession)
            .where(UserSession.token_hash == token_hash)
            .where(UserSession.revoked_at.is_(None))
            .where(UserSession.expires_at > now)
        )
        .scalar_one_or_none()
    )
    if not sess:
        return None

    user = db.get(User, sess.user_id)
    if not user or not user.is_active:
        return None
    return user


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    db = SessionLocal()
    try:
        user = _ws_auth_user(websocket, db)
        if not user:
            await websocket.close(code=4401)
            return

        if user.must_change_password:
            await websocket.close(code=4403)  # password change required
            return

        await websocket.accept()

        await websocket.send_text(
            json.dumps(
                {
                    "type": "status",
                    "status": f"authenticated as {user.email}",
                    "user_id": user.id,
                    "role": user.role,
                }
            )
        )

        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "start_run":
                cfg = msg.get("config", {}) or {}
                spec = JobSpec(
                    kind=str(cfg.get("kind", "generate_plasmid")),
                    config_path=str(cfg.get("config_path", "config/stream_config.yaml")),
                    params=dict(cfg.get("params", {"length_bp": 512, "output": "generated/web_ws_run.fasta"})),
                )
                await websocket.send_text(json.dumps({"type": "status", "status": f"run accepted for {user.email}", "progress": 0.0}))

                queue: list[dict] = []

                def _sink(ev: JobEvent):
                    queue.append({"type": "log", "line": f"[{ev.stage}] {ev.message}", "data": ev.data})

                result = await asyncio.to_thread(lambda: JobEngine(event_sink=_sink).run(spec))
                for entry in queue:
                    await websocket.send_text(json.dumps(entry))
                await websocket.send_text(json.dumps({"type": "result", "result": {"ok": result.ok, "message": result.message, "data": result.data}}))
                await websocket.send_text(json.dumps({"type": "status", "status": ("completed" if result.ok else "error"), "progress": (1.0 if result.ok else 0.0)}))

            elif msg.get("type") == "stop_run":
                await websocket.send_text(json.dumps({"type": "run_stopped", "message": "Run stopped"}))

            elif msg.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "ts": msg.get("ts")}))

            else:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "status",
                            "status": f"echo ({user.role})",
                            "data": msg,
                        }
                    )
                )

    except WebSocketDisconnect:
        pass
    finally:
        db.close()
