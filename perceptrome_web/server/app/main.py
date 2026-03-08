# server/app/main.py
import asyncio
from datetime import datetime, timedelta
import hashlib
import json
import logging
import smtplib
import threading
from dataclasses import dataclass, field
from collections import Counter
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, urlencode
from uuid import uuid4

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
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, select, text
from sqlalchemy.orm import Session

from .config import settings
from .auth_rate_limit import login_attempt_store
from .db import Base, engine, SessionLocal
from .deps import get_db, get_current_user, get_current_user_strict, require_role
from .models import AuthToken, Run, RunArtifact, User, UserSession
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
    RunArtifactOut,
    RunOut,
    RunStartRequest,
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

RunState = Literal["queued", "running", "completed", "failed", "canceled"]


@dataclass(slots=True)
class RunRecord:
    run_id: str
    spec: JobSpec
    state: RunState = "queued"
    cancel_event: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None


_run_lock = threading.Lock()
_runs: dict[str, RunRecord] = {}

VALID_JOB_KINDS = {"train_one", "stream", "generate_plasmid", "generate_protein", "validate_plasmid", "pretrain"}
REPLAY_DESCRIPTOR_VERSION = 1


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _json_loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _artifact_download_url(run_id: str, artifact_id: int) -> str:
    return f"/api/runs/{run_id}/artifacts/{artifact_id}/download"


def _run_to_out(run: Run) -> RunOut:
    artifacts = [
        RunArtifactOut(
            id=a.id,
            phase=a.phase,
            path=a.path,
            label=a.label,
            download_url=_artifact_download_url(run.run_id, a.id),
            created_at=a.created_at,
        )
        for a in sorted(run.artifacts, key=lambda item: item.created_at)
    ]
    result_obj = _json_loads(run.result_json)
    return RunOut(
        run_id=run.run_id,
        user_id=run.user_id,
        kind=run.kind,
        state=run.state,
        message=run.message,
        config=_json_loads(run.config_json),
        result=result_obj or None,
        submitted_at=run.submitted_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        artifacts=artifacts,
    )


def _find_run(db: Session, run_id: str) -> Run | None:
    return db.execute(select(Run).where(Run.run_id == run_id)).scalar_one_or_none()


def _save_run_submission(db: Session, user: User, run_id: str, spec: JobSpec, cfg: dict[str, Any]) -> Run:
    run = _find_run(db, run_id)
    if run and run.state in {"queued", "running"}:
        raise HTTPException(status_code=409, detail=f"Run already active: {run_id}")
    if not run:
        run = Run(run_id=run_id, user_id=user.id, kind=spec.kind)
        db.add(run)
    run.kind = spec.kind
    run.state = "queued"
    run.config_json = _json_dumps(cfg)
    run.result_json = None
    run.message = "queued"
    run.submitted_at = _utcnow()
    run.started_at = None
    run.finished_at = None
    db.commit()
    db.refresh(run)
    return run


def _mark_run_started(db: Session, run_id: str):
    run = _find_run(db, run_id)
    if not run:
        return
    run.state = "running"
    run.started_at = _utcnow()
    run.message = "running"
    db.commit()


def _record_artifact(db: Session, run_id: str, path: str, phase: str | None = None, label: str | None = None) -> RunArtifact | None:
    run = _find_run(db, run_id)
    if not run:
        return None
    artifact = RunArtifact(run_id=run.id, path=path, phase=phase, label=label)
    db.add(artifact)
    db.commit()
    db.refresh(artifact)
    return artifact


def _finalize_run(db: Session, run_id: str, state: RunState, result: dict[str, Any], message: str | None = None):
    run = _find_run(db, run_id)
    if not run:
        return
    run.state = state
    run.message = message or state
    run.result_json = _json_dumps(result)
    run.finished_at = _utcnow()
    db.commit()


def _assert_run_access(run: Run, user: User):
    if user.role != "admin" and run.user_id != user.id:
        raise HTTPException(status_code=403, detail="Forbidden")


def _extract_manifest_uri(data: dict[str, Any], run_id: str) -> str | None:
    manifest_path = data.get("manifest_path")
    if isinstance(manifest_path, str) and manifest_path:
        return f"/api/runs/{run_id}/artifacts/download-by-path?path={quote(manifest_path)}"
    return None


def _parse_job_spec(config: dict[str, Any]) -> tuple[str, JobSpec]:
    cfg = config or {}
    run_id = str(cfg.get("run_id") or cfg.get("manifest_id") or f"web_{uuid4().hex}")
    kind = str(cfg.get("kind", "generate_plasmid"))
    config_path = str(cfg.get("config_path", "config/stream_config.yaml"))
    if kind not in VALID_JOB_KINDS:
        raise HTTPException(status_code=400, detail=f"Unsupported run kind: {kind}")
    if not config_path:
        raise HTTPException(status_code=400, detail="config_path is required")

    reserved = {"run_id", "manifest_id", "kind", "config_path", "params"}
    params = {k: v for k, v in cfg.items() if k not in reserved}
    params.update(dict(cfg.get("params") or {}))
    params.setdefault("run_id", run_id)
    params.setdefault("manifest_id", run_id)

    spec = JobSpec(kind=kind, config_path=config_path, params=params)
    return run_id, spec


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _extract_seed_info(params: dict[str, Any]) -> dict[str, Any]:
    seed_keys = {
        "seed",
        "random_seed",
        "rng_seed",
        "np_seed",
        "torch_seed",
        "python_seed",
    }
    return {k: params[k] for k in sorted(seed_keys) if k in params}


def _extract_required_parent_artifacts(manifest_path: Path) -> list[dict[str, Any]]:
    payload = _load_json_file(manifest_path)
    required: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()

    def _append_parent(ref: dict[str, Any]):
        raw_path = ref.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            return
        resolved = Path(raw_path).expanduser().resolve()
        if resolved in seen_paths:
            return
        seen_paths.add(resolved)
        exists = resolved.exists() and resolved.is_file()
        required.append(
            {
                "path": str(resolved),
                "sha256": _sha256_file(resolved) if exists else None,
                "size_bytes": resolved.stat().st_size if exists else None,
                "artifact_id": ref.get("artifact_id"),
                "relation": ref.get("relation"),
            }
        )

    run = payload.get("run")
    if isinstance(run, dict):
        for parent in run.get("parents") or []:
            if isinstance(parent, dict):
                _append_parent(parent)

    for artifact in payload.get("artifacts") or []:
        if not isinstance(artifact, dict):
            continue
        for parent in artifact.get("parents") or []:
            if isinstance(parent, dict):
                _append_parent(parent)
    return required


def _build_replay_descriptor(*, cfg: dict[str, Any], spec: JobSpec, result_data: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    manifest_path_value = result_data.get("manifest_path")
    manifest_path = Path(str(manifest_path_value)).expanduser().resolve() if isinstance(manifest_path_value, str) and manifest_path_value else None
    explicit_params = dict(spec.params)
    descriptor: dict[str, Any] = {
        "schema_version": REPLAY_DESCRIPTOR_VERSION,
        "run_kind": spec.kind,
        "config_path": spec.config_path,
        "resolved_config_snapshot": result_data.get("config_snapshot"),
        "explicit_params": explicit_params,
        "seed_info": _extract_seed_info(explicit_params),
        "required_input_artifacts": _extract_required_parent_artifacts(manifest_path) if manifest_path and manifest_path.exists() else [],
        "source_run_config": cfg,
    }
    descriptor_path: str | None = None
    if manifest_path:
        descriptor_path = str(manifest_path.with_name(f"{manifest_path.stem}.replay.json"))
        target = Path(descriptor_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(descriptor, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return descriptor, descriptor_path


def _descriptor_hash(descriptor: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _validate_required_inputs(required_inputs: list[dict[str, Any]]) -> None:
    for item in required_inputs:
        path_value = item.get("path")
        expected_hash = item.get("sha256")
        if not isinstance(path_value, str) or not path_value:
            raise HTTPException(status_code=400, detail="Replay descriptor has an invalid required artifact path")
        target = Path(path_value).expanduser().resolve()
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=409, detail=f"Required artifact missing for replay: {target}")
        if isinstance(expected_hash, str) and expected_hash:
            actual_hash = _sha256_file(target)
            if actual_hash != expected_hash:
                raise HTTPException(status_code=409, detail=f"Required artifact hash mismatch for replay: {target}")


def _update_manifest_metadata(manifest_path: str, metadata: dict[str, Any]) -> None:
    if not manifest_path:
        return
    target = Path(manifest_path).expanduser().resolve()
    if not target.exists() or not target.is_file():
        return
    payload = _load_json_file(target)
    provenance = payload.get("provenance_metadata")
    if not isinstance(provenance, dict):
        provenance = {}
    replay_meta = provenance.get("replay")
    if not isinstance(replay_meta, dict):
        replay_meta = {}
    replay_meta.update(metadata)
    provenance["replay"] = replay_meta
    payload["provenance_metadata"] = provenance
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _execute_run(*, cfg: dict[str, Any], user: User, db: Session, manifest_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    run_id, spec = _parse_job_spec(cfg)
    record = _upsert_run(run_id, spec)
    _save_run_submission(db, user, run_id, spec, cfg)
    record.state = "running"
    _mark_run_started(db, run_id)
    result = JobEngine(cancel_event=record.cancel_event).run(spec)
    final_state: RunState = "completed" if result.ok else ("canceled" if result.exit_code == 130 else "failed")
    result_data = dict(result.data or {})
    manifest_path = result_data.get("manifest_path") if isinstance(result_data.get("manifest_path"), str) else None
    manifest_uri = _extract_manifest_uri(result_data, run_id)
    descriptor, descriptor_path = _build_replay_descriptor(cfg=cfg, spec=spec, result_data=result_data)
    descriptor_hash = _descriptor_hash(descriptor)
    final_payload = {
        "run_id": run_id,
        "ok": bool(result.ok),
        "state": final_state,
        "message": result.message,
        "manifest_path": manifest_path,
        "manifest_uri": manifest_uri,
        "replay_descriptor": descriptor,
        "replay_descriptor_path": descriptor_path,
        "replay_descriptor_hash": descriptor_hash,
        **result_data,
    }
    with _run_lock:
        record.state = final_state
        record.result = final_payload
    if manifest_path:
        _record_artifact(db, run_id, manifest_path, phase="manifest", label="Run manifest")
    if descriptor_path:
        _record_artifact(db, run_id, descriptor_path, phase="provenance", label="Replay descriptor")
    if manifest_path and manifest_metadata:
        _update_manifest_metadata(manifest_path, manifest_metadata)
    _finalize_run(db, run_id, final_state, final_payload, result.message)
    return {"ok": result.ok, "message": result.message, "user_id": user.id, "role": user.role, "result": final_payload}


def _load_run_replay_descriptor(source_run: Run) -> tuple[dict[str, Any], str]:
    result_payload = _json_loads(source_run.result_json)
    descriptor = result_payload.get("replay_descriptor")
    descriptor_path = result_payload.get("replay_descriptor_path")
    if isinstance(descriptor, dict):
        return descriptor, _descriptor_hash(descriptor)
    if isinstance(descriptor_path, str) and descriptor_path:
        descriptor_file = Path(descriptor_path).expanduser().resolve()
        if not descriptor_file.exists() or not descriptor_file.is_file():
            raise HTTPException(status_code=400, detail="Replay descriptor file is missing")
        descriptor_payload = _load_json_file(descriptor_file)
        return descriptor_payload, _descriptor_hash(descriptor_payload)
    raise HTTPException(status_code=400, detail="Replay descriptor is unavailable for this run")


def _upsert_run(run_id: str, spec: JobSpec) -> RunRecord:
    with _run_lock:
        existing = _runs.get(run_id)
        if existing and existing.state in {"queued", "running"}:
            raise HTTPException(status_code=409, detail=f"Run already active: {run_id}")
        record = RunRecord(run_id=run_id, spec=spec)
        _runs[run_id] = record
    return record


def _stop_run_record(run_id: str) -> bool:
    with _run_lock:
        record = _runs.get(run_id)
        if not record:
            return False
        record.cancel_event.set()
        if record.state == "queued":
            record.state = "canceled"
            record.result = {"run_id": run_id, "canceled": True, "manifest_path": None, "manifest_uri": None}
    return True


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
def start_run(payload: RunStartRequest | None = None, user: User = Depends(get_current_user_strict), db: Session = Depends(get_db)):
    cfg = dict((payload.config if payload else {}) or {})
    return _execute_run(cfg=cfg, user=user, db=db)


@app.post("/api/runs/{run_id}/replay")
def replay_run(run_id: str, user: User = Depends(get_current_user_strict), db: Session = Depends(get_db)):
    source_run = _find_run(db, run_id)
    if not source_run:
        raise HTTPException(status_code=404, detail="Run not found")
    _assert_run_access(source_run, user)

    descriptor, descriptor_hash = _load_run_replay_descriptor(source_run)
    required_inputs = descriptor.get("required_input_artifacts")
    if not isinstance(required_inputs, list):
        raise HTTPException(status_code=400, detail="Replay descriptor is invalid: required_input_artifacts must be a list")
    if any(not isinstance(item, dict) for item in required_inputs):
        raise HTTPException(status_code=400, detail="Replay descriptor is invalid: each required_input_artifacts entry must be an object")
    _validate_required_inputs(required_inputs)

    explicit_params = descriptor.get("explicit_params")
    if not isinstance(explicit_params, dict):
        raise HTTPException(status_code=400, detail="Replay descriptor is invalid: explicit_params must be an object")

    replay_run_id = f"replay_{uuid4().hex}"
    replay_cfg = {
        "run_id": replay_run_id,
        "manifest_id": replay_run_id,
        "kind": descriptor.get("run_kind"),
        "config_path": descriptor.get("config_path"),
        "params": explicit_params,
    }
    metadata = {"replayed_from_run_id": run_id, "descriptor_hash": descriptor_hash}
    return _execute_run(cfg=replay_cfg, user=user, db=db, manifest_metadata=metadata)


@app.get("/api/runs", response_model=list[RunOut])
def list_runs(user: User = Depends(get_current_user_strict), db: Session = Depends(get_db), limit: int = 50):
    q = select(Run).order_by(Run.submitted_at.desc()).limit(max(1, min(limit, 200)))
    if user.role != "admin":
        q = q.where(Run.user_id == user.id)
    runs = db.execute(q).scalars().all()
    return [_run_to_out(run) for run in runs]


@app.get("/api/runs/{run_id}", response_model=RunOut)
def get_run(run_id: str, user: User = Depends(get_current_user_strict), db: Session = Depends(get_db)):
    run = _find_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    _assert_run_access(run, user)
    return _run_to_out(run)


@app.get("/api/runs/{run_id}/artifacts", response_model=list[RunArtifactOut])
def list_run_artifacts(run_id: str, user: User = Depends(get_current_user_strict), db: Session = Depends(get_db)):
    run = _find_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    _assert_run_access(run, user)
    return _run_to_out(run).artifacts


@app.get("/api/runs/{run_id}/artifacts/{artifact_id}/download")
def download_artifact(run_id: str, artifact_id: int, user: User = Depends(get_current_user_strict), db: Session = Depends(get_db)):
    run = _find_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    _assert_run_access(run, user)
    artifact = db.execute(select(RunArtifact).where(RunArtifact.id == artifact_id).where(RunArtifact.run_id == run.id)).scalar_one_or_none()
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    target = Path(artifact.path).expanduser().resolve()
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found")
    return FileResponse(path=target, filename=target.name)


@app.get("/api/runs/{run_id}/artifacts/download-by-path")
def download_artifact_by_path(run_id: str, path: str, user: User = Depends(get_current_user_strict), db: Session = Depends(get_db)):
    run = _find_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    _assert_run_access(run, user)
    target = Path(path).expanduser().resolve()
    known_paths = {Path(a.path).expanduser().resolve() for a in run.artifacts}
    if target not in known_paths:
        raise HTTPException(status_code=403, detail="Artifact path not permitted")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found")
    return FileResponse(path=target, filename=target.name)


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
                cfg = dict(msg.get("config", {}) or {})
                run_id, spec = _parse_job_spec(cfg)
                try:
                    record = _upsert_run(run_id, spec)
                    _save_run_submission(db, user, run_id, spec, cfg)
                except HTTPException as exc:
                    await websocket.send_text(json.dumps({"type": "error", "run_id": run_id, "state": "failed", "message": str(exc.detail)}))
                    continue

                await websocket.send_text(json.dumps({"type": "status", "run_id": run_id, "state": "queued", "status": "queued", "phase": "queued", "progress": 0.0}))

                queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
                loop = asyncio.get_running_loop()

                def _sink(ev: JobEvent):
                    payload: dict[str, Any] = {"type": "log", "run_id": run_id, "phase": ev.stage, "line": ev.message, "data": ev.data}
                    if ev.stage == "train" and "loss" in ev.data:
                        payload = {"type": "metric", "run_id": run_id, "phase": ev.stage, "name": "loss", "value": float(ev.data["loss"]), "step": ev.data.get("epoch")}
                    if ev.stage == "validate" and ev.data:
                        payload = {"type": "validation-summary", "run_id": run_id, "phase": ev.stage, "summary": ev.data}
                    if "progress" in ev.data and isinstance(ev.data.get("progress"), (int, float)):
                        payload = {"type": "progress", "run_id": run_id, "state": "running", "phase": ev.stage, "progress": float(ev.data["progress"])}
                    elif ev.stage in {"start", "done", "stream_step", "generate", "validate", "pretrain", "encode", "train", "manifest"}:
                        payload = {"type": "phase", "run_id": run_id, "state": "running", "phase": ev.stage, "status": ev.message, "data": ev.data}
                    if "path" in ev.data and isinstance(ev.data.get("path"), str):
                        with SessionLocal() as tx:
                            artifact = _record_artifact(tx, run_id, ev.data["path"], phase=ev.stage)
                        artifact_payload = {"path": ev.data["path"]}
                        if artifact:
                            artifact_payload["download_url"] = _artifact_download_url(run_id, artifact.id)
                            loop.call_soon_threadsafe(queue.put_nowait, {"type": "checkpoint", "run_id": run_id, "phase": ev.stage, "path": ev.data["path"], "download_url": artifact_payload["download_url"]})
                        loop.call_soon_threadsafe(queue.put_nowait, {"type": "artifact-available", "run_id": run_id, "state": "running", "artifact": artifact_payload, "phase": ev.stage})
                    loop.call_soon_threadsafe(queue.put_nowait, payload)

                async def _forward_events(task: asyncio.Task[Any]):
                    while True:
                        if task.done() and queue.empty():
                            break
                        try:
                            item = await asyncio.wait_for(queue.get(), timeout=0.1)
                        except asyncio.TimeoutError:
                            continue
                        await websocket.send_text(json.dumps(item))

                record.state = "running"
                _mark_run_started(db, run_id)
                await websocket.send_text(json.dumps({"type": "status", "run_id": run_id, "state": "running", "status": "running", "phase": "start", "progress": 0.01}))

                run_task = asyncio.create_task(asyncio.to_thread(lambda: JobEngine(event_sink=_sink, cancel_event=record.cancel_event).run(spec)))
                forward_task = asyncio.create_task(_forward_events(run_task))
                result = await run_task
                await forward_task

                final_state: RunState = "completed" if result.ok else ("canceled" if result.exit_code == 130 else "failed")
                result_data = dict(result.data or {})
                manifest_path = result_data.get("manifest_path") if isinstance(result_data.get("manifest_path"), str) else None
                manifest_uri = _extract_manifest_uri(result_data, run_id)
                final_result = {
                    "run_id": run_id,
                    "ok": bool(result.ok),
                    "state": final_state,
                    "message": result.message,
                    "manifest_path": manifest_path,
                    "manifest_uri": manifest_uri,
                    **result_data,
                }
                with _run_lock:
                    record.state = final_state
                    record.result = final_result

                if manifest_path:
                    _record_artifact(db, run_id, manifest_path, phase="manifest", label="Run manifest")
                _finalize_run(db, run_id, final_state, final_result, result.message)

                await websocket.send_text(json.dumps({"type": "result", "run_id": run_id, "state": final_state, "result": final_result}))
                await websocket.send_text(json.dumps({"type": "status", "run_id": run_id, "state": final_state, "status": final_state, "phase": "done", "progress": (1.0 if final_state == "completed" else 0.0)}))

            elif msg.get("type") == "stop_run":
                run_id = str(msg.get("run_id") or "")
                if not run_id:
                    with _run_lock:
                        active = [rid for rid, rec in _runs.items() if rec.state in {"queued", "running"}]
                    run_id = active[-1] if active else ""
                if not run_id or not _stop_run_record(run_id):
                    await websocket.send_text(json.dumps({"type": "error", "state": "failed", "message": "run_id not found"}))
                else:
                    _finalize_run(db, run_id, "canceled", {"run_id": run_id, "canceled": True}, "Cancellation requested")
                    await websocket.send_text(json.dumps({"type": "run_stopped", "run_id": run_id, "state": "canceled", "message": "Cancellation requested"}))

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
