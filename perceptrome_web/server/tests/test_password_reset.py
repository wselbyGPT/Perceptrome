from datetime import datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app import main as main_module
from app.deps import get_db
from app.main import app
from tests.db_utils import apply_migrations
from app.models import AuthToken, User, UserSession
from app.security import hash_password, hash_session_token


def setup_client(monkeypatch, tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False}, future=True)
    testing_session_local = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)
    apply_migrations(f"sqlite:///{db_path}")

    def override_get_db():
        db = testing_session_local()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    issued_tokens = []

    def fake_send_password_reset_email(_recipient: str, raw_token: str):
        issued_tokens.append(raw_token)

    monkeypatch.setattr(main_module, "_send_password_reset_email", fake_send_password_reset_email)

    client = TestClient(app)
    return client, testing_session_local, issued_tokens


def test_forgot_password_enumeration_resistance(monkeypatch, tmp_path):
    client, db_factory, issued_tokens = setup_client(monkeypatch, tmp_path)

    with db_factory() as db:
        user = User(
            email="known@example.com",
            password_hash=hash_password("verysecure123"),
            role="user",
            is_active=True,
            must_change_password=False,
            email_verified_at=datetime.utcnow(),
        )
        db.add(user)
        db.commit()

    known = client.post("/api/auth/forgot-password", json={"email": "known@example.com"})
    unknown = client.post("/api/auth/forgot-password", json={"email": "unknown@example.com"})

    assert known.status_code == 200
    assert unknown.status_code == 200
    assert known.json() == unknown.json()
    assert len(issued_tokens) == 1


def test_password_reset_token_replay_prevention(monkeypatch, tmp_path):
    client, db_factory, issued_tokens = setup_client(monkeypatch, tmp_path)

    with db_factory() as db:
        user = User(
            email="replay@example.com",
            password_hash=hash_password("verysecure123"),
            role="user",
            is_active=True,
            must_change_password=True,
            email_verified_at=datetime.utcnow(),
        )
        db.add(user)
        db.commit()

    client.post("/api/auth/forgot-password", json={"email": "replay@example.com"})
    token = issued_tokens[-1]

    first = client.post("/api/auth/reset-password", json={"token": token, "new_password": "anEvenBetterPass123"})
    second = client.post("/api/auth/reset-password", json={"token": token, "new_password": "anotherStrongPass123"})

    assert first.status_code == 200
    assert second.status_code == 400
    assert second.json()["detail"] == "Password reset token already used"


def test_password_reset_token_expiration(monkeypatch, tmp_path):
    client, db_factory, issued_tokens = setup_client(monkeypatch, tmp_path)

    with db_factory() as db:
        user = User(
            email="expired@example.com",
            password_hash=hash_password("verysecure123"),
            role="user",
            is_active=True,
            must_change_password=False,
            email_verified_at=datetime.utcnow(),
        )
        db.add(user)
        db.commit()

    client.post("/api/auth/forgot-password", json={"email": "expired@example.com"})
    token = issued_tokens[-1]

    with db_factory() as db:
        token_hash = hash_session_token(token)
        auth_token = db.execute(select(AuthToken).where(AuthToken.token_hash == token_hash)).scalar_one()
        auth_token.expires_at = datetime.utcnow() - timedelta(minutes=1)
        db.commit()

    reset = client.post("/api/auth/reset-password", json={"token": token, "new_password": "anEvenBetterPass123"})
    assert reset.status_code == 400
    assert reset.json()["detail"] == "Password reset token expired"


def test_password_reset_revokes_active_sessions(monkeypatch, tmp_path):
    client, db_factory, issued_tokens = setup_client(monkeypatch, tmp_path)

    with db_factory() as db:
        user = User(
            email="sessions@example.com",
            password_hash=hash_password("verysecure123"),
            role="user",
            is_active=True,
            must_change_password=True,
            email_verified_at=datetime.utcnow(),
        )
        db.add(user)
        db.commit()

        active_session = UserSession(
            user_id=user.id,
            token_hash="active-token-hash",
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        expired_session = UserSession(
            user_id=user.id,
            token_hash="expired-token-hash",
            expires_at=datetime.utcnow() - timedelta(hours=1),
        )
        db.add(active_session)
        db.add(expired_session)
        db.commit()

    client.post("/api/auth/forgot-password", json={"email": "sessions@example.com"})
    token = issued_tokens[-1]
    reset = client.post("/api/auth/reset-password", json={"token": token, "new_password": "anEvenBetterPass123"})

    assert reset.status_code == 200

    with db_factory() as db:
        user = db.execute(select(User).where(User.email == "sessions@example.com")).scalar_one()
        sessions = db.execute(select(UserSession).where(UserSession.user_id == user.id)).scalars().all()
        active = next(s for s in sessions if s.token_hash == "active-token-hash")
        expired = next(s for s in sessions if s.token_hash == "expired-token-hash")

        assert active.revoked_at is not None
        assert expired.revoked_at is None
        assert user.must_change_password is False
