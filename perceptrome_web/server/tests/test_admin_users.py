from datetime import datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app import main as main_module
from app.deps import get_db
from app.main import app
from app.models import User, UserSession
from tests.db_utils import apply_migrations


def setup_client(monkeypatch, tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False}, future=True)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)
    apply_migrations(f"sqlite:///{db_path}")

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(main_module.settings, "allow_self_register", True)
    monkeypatch.setattr(main_module.settings, "email_verification_resend_cooldown_seconds", 0)
    monkeypatch.setattr(main_module, "_send_verification_email", lambda _recipient, _token: None)

    client = TestClient(app)
    return client, TestingSessionLocal


def create_admin_and_login(client: TestClient, db_factory):
    register = client.post("/api/auth/register", json={"email": "admin@example.com", "password": "verysecure123", "username": "adminuser"})
    assert register.status_code == 200

    with db_factory() as db:
        admin = db.execute(select(User).where(User.email == "admin@example.com")).scalar_one()
        admin.role = "admin"
        admin.email_verified_at = datetime.utcnow()
        db.commit()

    login = client.post("/api/auth/login", json={"email": "admin@example.com", "password": "verysecure123"})
    assert login.status_code == 200


def test_admin_user_filters_and_actions(monkeypatch, tmp_path):
    client, db_factory = setup_client(monkeypatch, tmp_path)
    create_admin_and_login(client, db_factory)

    created = client.post(
        "/api/admin/users",
        json={
            "email": "member@example.com",
            "password": "temporary123",
            "username": "member1",
            "role": "user",
            "is_active": True,
            "must_change_password": False,
        },
    )
    assert created.status_code == 200
    user_id = created.json()["id"]

    with db_factory() as db:
        user = db.get(User, user_id)
        user.email_verified_at = None
        user.last_login_at = datetime(2024, 1, 2, 3, 4, 5)
        user.failed_login_count = 3
        user.locked_until = datetime.utcnow() + timedelta(minutes=30)
        db.add(UserSession(user_id=user.id, token_hash="session-a", expires_at=datetime.utcnow() + timedelta(hours=1)))
        db.add(UserSession(user_id=user.id, token_hash="session-b", expires_at=datetime.utcnow() + timedelta(hours=1)))
        db.commit()

    filtered = client.get("/api/admin/users", params={"search": "member", "verification": "pending", "must_change_password": "false"})
    assert filtered.status_code == 200
    payload = filtered.json()
    assert payload["total"] == 1
    assert payload["users"][0]["id"] == user_id
    assert payload["users"][0]["last_login_at"] == "2024-01-02T03:04:05"
    assert payload["users"][0]["is_locked"] is True

    updated = client.patch(f"/api/admin/users/{user_id}", json={"role": "admin", "must_change_password": True, "username": "member-renamed", "is_active": True})
    assert updated.status_code == 200
    assert updated.json()["role"] == "admin"
    assert updated.json()["must_change_password"] is True
    assert updated.json()["username"] == "member-renamed"

    suspended = client.post(f"/api/admin/users/{user_id}/suspend")
    assert suspended.status_code == 200
    assert suspended.json()["user"]["is_active"] is False
    assert suspended.json()["revoked_session_count"] == 2

    activated = client.post(f"/api/admin/users/{user_id}/activate")
    assert activated.status_code == 200
    assert activated.json()["user"]["is_active"] is True

    resent = client.post(f"/api/admin/users/{user_id}/resend-verification")
    assert resent.status_code == 200
    assert resent.json()["message"] == "Verification email resent"

    revoke_again = client.post(f"/api/admin/users/{user_id}/revoke-sessions")
    assert revoke_again.status_code == 200
    assert revoke_again.json()["revoked_session_count"] == 0

    force_reset = client.post(f"/api/admin/users/{user_id}/force-reset")
    assert force_reset.status_code == 200
    assert force_reset.json()["user"]["must_change_password"] is True

    with db_factory() as db:
        refreshed = db.get(User, user_id)
        assert refreshed.role == "admin"
        assert refreshed.username == "member-renamed"
        assert refreshed.must_change_password is True
        sessions = db.execute(select(UserSession).where(UserSession.user_id == user_id)).scalars().all()
        assert all(session.revoked_at is not None for session in sessions)
