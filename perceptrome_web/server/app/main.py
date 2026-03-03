# server/app/main.py
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json
import smtplib
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
from .db import Base, engine, SessionLocal
from .deps import get_db, get_current_user, get_current_user_strict, require_role
from .models import EmailVerificationToken, User, UserSession
from .schemas import (
    RegisterRequest,
    LoginRequest,
    VerifyEmailRequest,
    ResendVerificationRequest,
    ChangePasswordRequest,
    AdminCreateUserRequest,
    UserOut,
    MessageOut,
)
from .security import hash_password, verify_password, make_session_token, hash_session_token

app = FastAPI(title=settings.app_name)

origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

_login_attempts: dict[str, deque[float]] = defaultdict(deque)


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


def _send_verification_email(recipient: str, raw_token: str):
    link = _build_verification_link(raw_token)

    if settings.mail_provider.lower() == "smtp":
        msg = EmailMessage()
        msg["Subject"] = "Verify your Perceptrome account"
        msg["From"] = settings.mail_from_email
        msg["To"] = recipient
        msg.set_content(
            "Welcome to Perceptrome!\n\n"
            "Use the following link to verify your email:\n"
            f"{link}\n\n"
            "If you did not sign up, you can ignore this email."
        )

        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as smtp:
            if settings.smtp_use_tls:
                smtp.starttls()
            if settings.smtp_username:
                smtp.login(settings.smtp_username, settings.smtp_password or "")
            smtp.send_message(msg)
        return

    print(f"[auth] verification email to {recipient}: {link}")


def _issue_email_verification_token(db: Session, user: User) -> str:
    raw = make_session_token()
    token = EmailVerificationToken(
        user_id=user.id,
        token_hash=hash_session_token(raw),
        expires_at=_utcnow() + timedelta(minutes=settings.email_verification_token_ttl_minutes),
    )
    user.email_verification_sent_at = _utcnow()
    db.add(token)
    db.commit()
    return raw


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


def _is_rate_limited(key: str) -> bool:
    now_ts = _utcnow().timestamp()
    q = _login_attempts[key]
    window = settings.login_rate_limit_window_seconds

    while q and (now_ts - q[0]) > window:
        q.popleft()

    if len(q) >= settings.login_rate_limit_max_attempts:
        return True

    q.append(now_ts)
    return False


def _ensure_users_columns():
    insp = inspect(engine)
    if "users" not in insp.get_table_names():
        return

    cols = {c["name"] for c in insp.get_columns("users")}
    statements = {
        "must_change_password": "ALTER TABLE users ADD COLUMN must_change_password BOOLEAN NOT NULL DEFAULT 0",
        "email_verified_at": "ALTER TABLE users ADD COLUMN email_verified_at DATETIME NULL",
        "email_verification_sent_at": "ALTER TABLE users ADD COLUMN email_verification_sent_at DATETIME NULL",
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
    token = db.execute(select(EmailVerificationToken).where(EmailVerificationToken.token_hash == token_hash)).scalar_one_or_none()

    if not token:
        raise HTTPException(status_code=400, detail="Invalid verification token")

    if token.used_at is not None:
        raise HTTPException(status_code=400, detail="Verification token already used")

    if token.expires_at <= _utcnow():
        raise HTTPException(status_code=400, detail="Verification token expired")

    user = db.get(User, token.user_id)
    if not user:
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
            raise HTTPException(status_code=429, detail="Please wait before requesting another verification email")

    raw_token = _issue_email_verification_token(db, user)
    _send_verification_email(user.email, raw_token)
    return MessageOut(message="Verification email sent")


@app.post("/api/auth/login", response_model=UserOut)
def login(payload: LoginRequest, request: Request, response: Response, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()
    ip = request.client.host if request.client else "unknown"
    rl_key = f"{ip}:{email}"

    if _is_rate_limited(rl_key):
        raise HTTPException(status_code=429, detail="Too many login attempts. Try again soon.")

    user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()

    if not user or not user.is_active or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if user.email_verified_at is None:
        raise HTTPException(status_code=403, detail="Email verification required")

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
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not verify_password(payload.current_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    if payload.current_password == payload.new_password:
        raise HTTPException(status_code=400, detail="New password must be different")

    user.password_hash = hash_password(payload.new_password)
    user.must_change_password = False
    db.commit()
    return MessageOut(message="Password changed")


@app.post("/api/runs/start")
def start_run(user: User = Depends(get_current_user_strict)):
    return {
        "ok": True,
        "message": f"Run started by {user.email}",
        "user_id": user.id,
        "role": user.role,
    }


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
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "status",
                            "status": f"run accepted for {user.email}",
                            "progress": 0.0,
                        }
                    )
                )
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "log",
                            "line": f"Starting run with config: {json.dumps(msg.get('config', {}))}",
                        }
                    )
                )
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "status",
                            "status": "running",
                            "progress": 0.5,
                        }
                    )
                )
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "result",
                            "result": {
                                "ok": True,
                                "owner": user.email,
                                "role": user.role,
                                "echo_config": msg.get("config", {}),
                            },
                        }
                    )
                )
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "status",
                            "status": "completed",
                            "progress": 1.0,
                        }
                    )
                )

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
