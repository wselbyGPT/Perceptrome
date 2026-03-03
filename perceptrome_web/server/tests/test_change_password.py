from datetime import datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.db import Base
from app.deps import get_db
from app.main import app
from app.models import User, UserSession
from app.security import hash_password, hash_session_token, verify_password


def setup_client(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False}, future=True)
    testing_session_local = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = testing_session_local()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    client = TestClient(app)
    return client, testing_session_local


def test_change_password_user_initiated_revokes_other_active_sessions(tmp_path):
    client, db_factory = setup_client(tmp_path)

    current_raw = "current-session-token"
    other_raw = "other-session-token"

    with db_factory() as db:
        user = User(
            email="normal-user@example.com",
            password_hash=hash_password("OldPassword123"),
            role="user",
            is_active=True,
            must_change_password=False,
            email_verified_at=datetime.utcnow(),
        )
        db.add(user)
        db.commit()

        current_session = UserSession(
            user_id=user.id,
            token_hash=hash_session_token(current_raw),
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        other_session = UserSession(
            user_id=user.id,
            token_hash=hash_session_token(other_raw),
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        db.add(current_session)
        db.add(other_session)
        db.commit()

    client.cookies.set(settings.session_cookie_name, current_raw)
    response = client.post(
        "/api/auth/change-password",
        json={
            "current_password": "OldPassword123",
            "new_password": "NewPassword123",
        },
    )

    assert response.status_code == 200

    with db_factory() as db:
        user = db.execute(select(User).where(User.email == "normal-user@example.com")).scalar_one()
        assert user.must_change_password is False
        assert verify_password("NewPassword123", user.password_hash)

        sessions = db.execute(select(UserSession).where(UserSession.user_id == user.id)).scalars().all()
        current = next(s for s in sessions if s.token_hash == hash_session_token(current_raw))
        other = next(s for s in sessions if s.token_hash == hash_session_token(other_raw))

        assert current.revoked_at is None
        assert other.revoked_at is not None


def test_change_password_rejects_weak_password(tmp_path):
    client, db_factory = setup_client(tmp_path)

    current_raw = "single-session-token"

    with db_factory() as db:
        user = User(
            email="weak-check@example.com",
            password_hash=hash_password("OldPassword123"),
            role="user",
            is_active=True,
            must_change_password=False,
            email_verified_at=datetime.utcnow(),
        )
        db.add(user)
        db.commit()

        db.add(
            UserSession(
                user_id=user.id,
                token_hash=hash_session_token(current_raw),
                expires_at=datetime.utcnow() + timedelta(hours=1),
            )
        )
        db.commit()

    client.cookies.set(settings.session_cookie_name, current_raw)
    response = client.post(
        "/api/auth/change-password",
        json={
            "current_password": "OldPassword123",
            "new_password": "alllowercase123",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Password must include at least one uppercase letter"
